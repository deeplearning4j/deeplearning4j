/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.autodiff.samediff.execution;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

/**
 * Compiles a {@link ForwardExecutionDAG} and concrete placeholder shapes into an
 * {@link ExecutionPlan} with pre-computed shapes, liveness-based buffer reuse,
 * and a contiguous memory layout.
 *
 * <p>The compilation pipeline mirrors ONNX Runtime's allocation planner:</p>
 * <ol>
 *   <li><b>Shape propagation</b> — Walk ops in topological order, calling
 *       {@code calculateOutputShape} with lightweight OpContexts to determine
 *       all intermediate tensor shapes.</li>
 *   <li><b>Liveness analysis</b> — Count uses per variable; a variable is dead
 *       after its last consumer executes.</li>
 *   <li><b>Buffer assignment with reuse</b> — Maintain a freelist of dead buffers.
 *       When allocating a new intermediate, first check if a freed buffer of
 *       compatible size exists.  If so, reuse it (same offset in pool).</li>
 *   <li><b>Layout</b> — All ALLOCATE/OUTPUT buffers are packed into a contiguous
 *       byte range with 64-byte alignment.  The total is {@link ExecutionPlan#totalHostBytes}.</li>
 * </ol>
 *
 * <p>Ops that require input <em>values</em> (not just shapes) to compute output
 * shapes (e.g., Reshape with a data-dependent shape tensor) are flagged as
 * {@link OpSlot#isRequiresDynamicShapeInference()}.  The executor will fall back
 * to runtime shape inference for those slots only.</p>
 *
 * @see ExecutionPlan
 * @see PlanExecutor
 */
@Slf4j
public class ExecutionPlanCompiler {

    private static final Set<String> SHAPE_DEPENDENT_OPS = new HashSet<>(Arrays.asList(
            "reshape", "reshape_no_copy", "create", "fill", "expand_dims",
            "constant_of_shape", "broadcast_to", "one_hot", "eye",
            "linspace", "range"
    ));

    private ExecutionPlanCompiler() {}

    /**
     * Compile an execution plan from a DAG and placeholder shapes.
     *
     * @param sd               the SameDiff graph
     * @param dag              the forward execution DAG (topological order, dependencies)
     * @param placeholderArrays actual placeholder arrays (used for shape + as shape-value sources)
     * @param requestedOutputs the output variable names this plan should produce
     * @return a compiled, immutable ExecutionPlan
     */
    public static ExecutionPlan compile(SameDiff sd,
                                        ForwardExecutionDAG dag,
                                        Map<String, INDArray> placeholderArrays,
                                        Set<String> requestedOutputs) {

        List<ExecutionNode> topoOrder = dag.getExecutionOrder();

        // ---- Pass 1: Shape propagation ----
        // Maps variable name -> (shape, dataType). For intermediates this is computed
        // by calling calculateOutputShape; for placeholders/constants it is known.
        Map<String, long[]> knownShapes = new LinkedHashMap<>();
        Map<String, DataType> knownTypes = new LinkedHashMap<>();
        Set<String> dynamicShapeOps = new HashSet<>();

        seedKnownShapes(sd, dag, placeholderArrays, knownShapes, knownTypes);
        propagateShapes(sd, dag, topoOrder, placeholderArrays, knownShapes, knownTypes, dynamicShapeOps);

        // ---- Pass 2: Liveness analysis (use counting) ----
        Map<String, Integer> useCount = computeUseCounts(sd, topoOrder);

        // ---- Pass 3: Buffer assignment with freelist reuse ----
        List<BufferAllocation> allocations = new ArrayList<>();
        Map<String, BufferAllocation> bufferMap = new LinkedHashMap<>();
        Deque<BufferAllocation> freelist = new ArrayDeque<>();
        long currentOffset = 0;
        int nextIndex = 0;

        // Assign buffers for each op's outputs in topological order
        for (ExecutionNode node : topoOrder) {
            if (node.getNodeType() == ExecutionNode.ExecutionNodeType.PLACEHOLDER_SET ||
                node.getNodeType() == ExecutionNode.ExecutionNodeType.VARIABLE_INIT) {
                // These are pre-existing — no workspace allocation needed
                for (String output : node.getOutputVariables()) {
                    SDVariable var = sd.getVariable(output);
                    if (var == null) continue;
                    BufferAllocKind kind;
                    if (var.getVariableType() == VariableType.PLACEHOLDER) {
                        kind = BufferAllocKind.PRE_EXISTING;
                    } else if (var.getVariableType() == VariableType.CONSTANT) {
                        kind = BufferAllocKind.CONSTANT;
                    } else {
                        kind = BufferAllocKind.VARIABLE;
                    }
                    long[] shape = knownShapes.getOrDefault(output, new long[0]);
                    DataType dt = knownTypes.getOrDefault(output, DataType.FLOAT);
                    BufferAllocation alloc = new BufferAllocation(
                            nextIndex++, 0, 0, dt, shape, -1, kind, output);
                    allocations.add(alloc);
                    bufferMap.put(output, alloc);
                }
                continue;
            }

            SameDiffOp sdOp = sd.getOps().get(node.getOperationName());
            if (sdOp == null) continue;

            // Decrement use counts for inputs — free dead buffers
            for (String input : node.getInputVariables()) {
                Integer remaining = useCount.get(input);
                if (remaining == null) continue;
                remaining = remaining - 1;
                useCount.put(input, remaining);
                if (remaining <= 0) {
                    BufferAllocation buf = bufferMap.get(input);
                    if (buf != null && (buf.getKind() == BufferAllocKind.ALLOCATE ||
                                        buf.getKind() == BufferAllocKind.REUSE)) {
                        freelist.addFirst(buf);
                    }
                }
            }

            // Allocate outputs
            List<String> outputNames = node.getOutputVariables();
            for (int i = 0; i < outputNames.size(); i++) {
                String output = outputNames.get(i);
                long[] shape = knownShapes.get(output);
                DataType dt = knownTypes.getOrDefault(output, DataType.FLOAT);

                if (shape == null) {
                    // Shape unknown (dynamic op) — allocate a placeholder slot
                    BufferAllocation alloc = new BufferAllocation(
                            nextIndex++, 0, 0, dt, new long[0], -1,
                            BufferAllocKind.ALLOCATE, output);
                    allocations.add(alloc);
                    bufferMap.put(output, alloc);
                    continue;
                }

                long sizeBytes = dt.width() * numElements(shape);
                if (sizeBytes <= 0) sizeBytes = dt.width(); // scalar

                boolean isOutput = requestedOutputs.contains(output);
                if (isOutput) {
                    // Graph outputs can't be reused — must persist
                    long aligned = ExecutionPlan.align(sizeBytes);
                    BufferAllocation alloc = new BufferAllocation(
                            nextIndex++, currentOffset, sizeBytes, dt, shape, -1,
                            BufferAllocKind.OUTPUT, output);
                    allocations.add(alloc);
                    bufferMap.put(output, alloc);
                    currentOffset += aligned;
                } else {
                    // Try to reuse from freelist
                    BufferAllocation reused = findReusable(freelist, sizeBytes, dt);
                    if (reused != null) {
                        BufferAllocation alloc = new BufferAllocation(
                                nextIndex++, reused.getOffsetInPool(), sizeBytes, dt, shape,
                                reused.getIndex(), BufferAllocKind.REUSE, output);
                        allocations.add(alloc);
                        bufferMap.put(output, alloc);
                    } else {
                        long aligned = ExecutionPlan.align(sizeBytes);
                        BufferAllocation alloc = new BufferAllocation(
                                nextIndex++, currentOffset, sizeBytes, dt, shape, -1,
                                BufferAllocKind.ALLOCATE, output);
                        allocations.add(alloc);
                        bufferMap.put(output, alloc);
                        currentOffset += aligned;
                    }
                }
            }
        }

        // ---- Pass 4: Build OpSlots ----
        List<OpSlot> slots = buildOpSlots(sd, topoOrder, bufferMap, knownShapes, knownTypes, dynamicShapeOps);

        // ---- Compute shape signature ----
        long shapeSignature = computeShapeSignature(placeholderArrays);

        return new ExecutionPlan(shapeSignature, slots, currentOffset, bufferMap, allocations, requestedOutputs);
    }

    // ---- Shape propagation helpers ----

    private static void seedKnownShapes(SameDiff sd, ForwardExecutionDAG dag,
                                         Map<String, INDArray> placeholderArrays,
                                         Map<String, long[]> knownShapes,
                                         Map<String, DataType> knownTypes) {
        // Seed placeholders
        if (placeholderArrays != null) {
            for (Map.Entry<String, INDArray> entry : placeholderArrays.entrySet()) {
                INDArray arr = entry.getValue();
                if (arr != null) {
                    knownShapes.put(entry.getKey(), arr.shape());
                    knownTypes.put(entry.getKey(), arr.dataType());
                }
            }
        }

        // Seed constants and variables
        for (String constName : dag.getConstants()) {
            SDVariable var = sd.getVariable(constName);
            if (var != null && var.getArr() != null) {
                knownShapes.put(constName, var.getArr().shape());
                knownTypes.put(constName, var.getArr().dataType());
            }
        }
        for (String varName : dag.getVariables()) {
            SDVariable var = sd.getVariable(varName);
            if (var != null && var.getArr() != null) {
                knownShapes.put(varName, var.getArr().shape());
                knownTypes.put(varName, var.getArr().dataType());
            }
        }
    }

    private static void propagateShapes(SameDiff sd, ForwardExecutionDAG dag,
                                         List<ExecutionNode> topoOrder,
                                         Map<String, INDArray> placeholderArrays,
                                         Map<String, long[]> knownShapes,
                                         Map<String, DataType> knownTypes,
                                         Set<String> dynamicShapeOps) {
        for (ExecutionNode node : topoOrder) {
            if (node.getNodeType() == ExecutionNode.ExecutionNodeType.PLACEHOLDER_SET ||
                node.getNodeType() == ExecutionNode.ExecutionNodeType.VARIABLE_INIT) {
                continue;
            }

            SameDiffOp sdOp = sd.getOps().get(node.getOperationName());
            if (sdOp == null) continue;
            DifferentialFunction fn = sdOp.getOp();
            if (fn == null) continue;

            // Check if all input shapes are known
            boolean allInputsKnown = true;
            for (String input : node.getInputVariables()) {
                if (!knownShapes.containsKey(input)) {
                    allInputsKnown = false;
                    break;
                }
            }

            if (!allInputsKnown) {
                // Can't compute shapes yet — mark as dynamic
                dynamicShapeOps.add(node.getOperationName());
                log.debug("Marking op {} as dynamic: not all input shapes known", node.getOperationName());
                continue;
            }

            // Check if this is a shape-dependent op (needs input values, not just shapes)
            boolean isShapeDependent = SHAPE_DEPENDENT_OPS.contains(fn.opName());

            if (isShapeDependent) {
                // For shape-dependent ops, we can still compute shapes if the shape-producing
                // inputs are constants or placeholders (whose values we have)
                boolean canResolve = canResolveShapeDependentOp(sd, node, placeholderArrays);
                if (!canResolve) {
                    dynamicShapeOps.add(node.getOperationName());
                    log.debug("Marking op {} ({}) as dynamic: shape-dependent with non-constant inputs",
                              node.getOperationName(), fn.opName());
                    continue;
                }
            }

            // Build a shape-inference OpContext
            try {
                List<DataBuffer> outShapes = computeOutputShapes(sd, fn, node, placeholderArrays, knownShapes, knownTypes);
                if (outShapes != null && !outShapes.isEmpty()) {
                    List<String> outputNames = node.getOutputVariables();
                    for (int i = 0; i < Math.min(outShapes.size(), outputNames.size()); i++) {
                        long[] shapeInfo = outShapes.get(i).asLong();
                        knownShapes.put(outputNames.get(i), Shape.shape(shapeInfo));
                        knownTypes.put(outputNames.get(i), Shape.dataType(shapeInfo));
                    }
                } else {
                    dynamicShapeOps.add(node.getOperationName());
                    log.debug("Marking op {} as dynamic: calculateOutputShape returned null/empty",
                              node.getOperationName());
                }
            } catch (Exception e) {
                dynamicShapeOps.add(node.getOperationName());
                log.debug("Marking op {} as dynamic: shape calculation failed: {}",
                          node.getOperationName(), e.getMessage());
            }
        }
    }

    private static boolean canResolveShapeDependentOp(SameDiff sd, ExecutionNode node,
                                                       Map<String, INDArray> placeholderArrays) {
        // A shape-dependent op can be resolved if the shape-producing inputs
        // (typically index 1+) are constants or placeholders with known values
        List<String> inputs = node.getInputVariables();
        for (int i = 1; i < inputs.size(); i++) {
            String input = inputs.get(i);
            SDVariable var = sd.getVariable(input);
            if (var == null) return false;
            if (var.getVariableType() == VariableType.CONSTANT && var.getArr() != null) continue;
            if (var.getVariableType() == VariableType.PLACEHOLDER && placeholderArrays != null &&
                placeholderArrays.containsKey(input)) continue;
            // This input is a computed intermediate — can't resolve statically
            return false;
        }
        return true;
    }

    private static List<DataBuffer> computeOutputShapes(SameDiff sd, DifferentialFunction fn,
                                                         ExecutionNode node,
                                                         Map<String, INDArray> placeholderArrays,
                                                         Map<String, long[]> knownShapes,
                                                         Map<String, DataType> knownTypes) {
        // Build a temporary OpContext with input arrays for shape inference
        org.nd4j.linalg.api.ops.OpContext ctx = Nd4j.getExecutioner().buildContext();
        try {
            List<String> inputNames = node.getInputVariables();
            List<INDArray> inputArrays = new ArrayList<>(inputNames.size());
            for (String inputName : inputNames) {
                INDArray arr = resolveInputArray(sd, inputName, placeholderArrays, knownShapes, knownTypes);
                inputArrays.add(arr);
            }
            ctx.setInputArrays(inputArrays);

            // Set arguments for DynamicCustomOps
            if (fn instanceof DynamicCustomOp) {
                DynamicCustomOp dynOp = (DynamicCustomOp) fn;
                ctx.setIArguments(dynOp.iArgs());
                ctx.setTArguments(dynOp.tArgs());
                ctx.setDArguments(dynOp.dArgs());
                ctx.setBArguments(dynOp.bArgs());
                ctx.setSArguments(dynOp.sArgs());
            }

            return fn.calculateOutputShape(ctx);
        } finally {
            try {
                ctx.close();
            } catch (Exception e) {
                // ignore cleanup errors
            }
        }
    }

    private static INDArray resolveInputArray(SameDiff sd, String varName,
                                               Map<String, INDArray> placeholderArrays,
                                               Map<String, long[]> knownShapes,
                                               Map<String, DataType> knownTypes) {
        // Try to get actual array (for constants/variables/placeholders)
        SDVariable var = sd.getVariable(varName);
        if (var != null) {
            if (var.getVariableType() == VariableType.CONSTANT || var.getVariableType() == VariableType.VARIABLE) {
                INDArray arr = var.getArr();
                if (arr != null) return arr;
            }
            if (var.getVariableType() == VariableType.PLACEHOLDER && placeholderArrays != null) {
                INDArray arr = placeholderArrays.get(varName);
                if (arr != null) return arr;
            }
        }

        // Create a dummy array with the known shape for shape inference only
        long[] shape = knownShapes.get(varName);
        DataType dt = knownTypes.getOrDefault(varName, DataType.FLOAT);
        if (shape != null && shape.length > 0) {
            return Nd4j.zeros(dt, shape);
        }

        // Scalar fallback
        return Nd4j.scalar(dt, 0);
    }

    // ---- Liveness analysis ----

    private static Map<String, Integer> computeUseCounts(SameDiff sd, List<ExecutionNode> topoOrder) {
        Map<String, Integer> useCount = new HashMap<>();
        for (ExecutionNode node : topoOrder) {
            for (String input : node.getInputVariables()) {
                useCount.merge(input, 1, Integer::sum);
            }
        }
        return useCount;
    }

    // ---- Freelist buffer reuse ----

    private static BufferAllocation findReusable(Deque<BufferAllocation> freelist, long needed, DataType dt) {
        // Search for a buffer with compatible size (>= needed, same data type for alignment)
        // Prefer best-fit (smallest buffer that is large enough)
        BufferAllocation bestFit = null;
        long bestFitSize = Long.MAX_VALUE;

        Iterator<BufferAllocation> it = freelist.iterator();
        while (it.hasNext()) {
            BufferAllocation candidate = it.next();
            long candidateAligned = ExecutionPlan.align(candidate.getSizeInBytes());
            long neededAligned = ExecutionPlan.align(needed);
            if (candidateAligned >= neededAligned && candidateAligned < bestFitSize) {
                bestFit = candidate;
                bestFitSize = candidateAligned;
                // Perfect fit — take it immediately
                if (candidateAligned == neededAligned) break;
            }
        }

        if (bestFit != null) {
            freelist.remove(bestFit);
        }
        return bestFit;
    }

    // ---- OpSlot construction ----

    private static List<OpSlot> buildOpSlots(SameDiff sd, List<ExecutionNode> topoOrder,
                                              Map<String, BufferAllocation> bufferMap,
                                              Map<String, long[]> knownShapes,
                                              Map<String, DataType> knownTypes,
                                              Set<String> dynamicShapeOps) {
        List<OpSlot> slots = new ArrayList<>();

        for (ExecutionNode node : topoOrder) {
            if (node.getNodeType() == ExecutionNode.ExecutionNodeType.PLACEHOLDER_SET ||
                node.getNodeType() == ExecutionNode.ExecutionNodeType.VARIABLE_INIT) {
                continue; // Not executable ops
            }

            SameDiffOp sdOp = sd.getOps().get(node.getOperationName());
            if (sdOp == null) continue;
            DifferentialFunction fn = sdOp.getOp();
            if (fn == null) continue;

            boolean isCustomOp = fn instanceof CustomOp;
            boolean isDynamic = dynamicShapeOps.contains(node.getOperationName());

            // Input buffer indices
            List<String> inputNames = node.getInputVariables();
            int[] inputIndices = new int[inputNames.size()];
            String[] inputVarNames = inputNames.toArray(new String[0]);
            for (int i = 0; i < inputNames.size(); i++) {
                BufferAllocation alloc = bufferMap.get(inputNames.get(i));
                if (alloc != null && alloc.getKind() != BufferAllocKind.PRE_EXISTING &&
                    alloc.getKind() != BufferAllocKind.CONSTANT &&
                    alloc.getKind() != BufferAllocKind.VARIABLE) {
                    inputIndices[i] = alloc.getIndex();
                } else {
                    inputIndices[i] = -1; // Resolved externally
                }
            }

            // Output buffer indices
            List<String> outputNames = node.getOutputVariables();
            int[] outputIndices = new int[outputNames.size()];
            String[] outputVarNames = outputNames.toArray(new String[0]);
            long[][] outputShapes = new long[outputNames.size()][];
            DataType[] outputDataTypes = new DataType[outputNames.size()];
            for (int i = 0; i < outputNames.size(); i++) {
                BufferAllocation alloc = bufferMap.get(outputNames.get(i));
                if (alloc != null) {
                    outputIndices[i] = alloc.getIndex();
                    outputShapes[i] = alloc.getShape();
                    outputDataTypes[i] = alloc.getDataType();
                } else {
                    outputIndices[i] = -1;
                    outputShapes[i] = knownShapes.getOrDefault(outputNames.get(i), new long[0]);
                    outputDataTypes[i] = knownTypes.getOrDefault(outputNames.get(i), DataType.FLOAT);
                }
            }

            // Op arguments
            long[] iArgs = new long[0];
            double[] tArgs = new double[0];
            boolean[] bArgs = new boolean[0];
            DataType[] dArgs = new DataType[0];
            if (isCustomOp) {
                DynamicCustomOp dynOp = (DynamicCustomOp) fn;
                iArgs = dynOp.iArgs();
                tArgs = dynOp.tArgs();
                bArgs = dynOp.bArgs();
                dArgs = dynOp.dArgs();
            }

            OpSlot slot = OpSlot.builder()
                    .opName(node.getOperationName())
                    .op(fn)
                    .customOp(isCustomOp)
                    .inputBufferIndices(inputIndices)
                    .inputVarNames(inputVarNames)
                    .outputBufferIndices(outputIndices)
                    .outputVarNames(outputVarNames)
                    .outputShapes(outputShapes)
                    .outputDataTypes(outputDataTypes)
                    .iArgs(iArgs)
                    .tArgs(tArgs)
                    .bArgs(bArgs)
                    .dArgs(dArgs)
                    .requiresDynamicShapeInference(isDynamic)
                    .build();
            slots.add(slot);
        }

        return slots;
    }

    // ---- Utilities ----

    static long computeShapeSignature(Map<String, INDArray> placeholderArrays) {
        if (placeholderArrays == null || placeholderArrays.isEmpty()) return 0L;

        // Deterministic ordering by name
        TreeMap<String, INDArray> sorted = new TreeMap<>(placeholderArrays);
        long h = 1;
        for (Map.Entry<String, INDArray> entry : sorted.entrySet()) {
            h = h * 31 + entry.getKey().hashCode();
            INDArray arr = entry.getValue();
            if (arr != null) {
                h = h * 31 + arr.dataType().ordinal();
                for (long dim : arr.shape()) {
                    h = h * 31 + dim;
                }
            }
        }
        return h;
    }

    private static long numElements(long[] shape) {
        if (shape == null || shape.length == 0) return 1;
        long prod = 1;
        for (long dim : shape) {
            if (dim > 0) prod *= dim;
        }
        return prod;
    }
}

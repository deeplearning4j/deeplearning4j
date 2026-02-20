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
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.BaseReduceOp;
import org.nd4j.linalg.api.ops.BaseScalarOp;
import org.nd4j.linalg.api.ops.BaseTransformBoolOp;
import org.nd4j.linalg.api.ops.BaseTransformFloatOp;
import org.nd4j.linalg.api.ops.BaseTransformSameOp;
import org.nd4j.linalg.api.ops.BaseTransformStrictOp;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

/**
 * Compiles a {@link DynamicShapePlan} from a {@link ForwardExecutionDAG}.
 *
 * <p>The compilation pipeline:</p>
 * <ol>
 *   <li>Filter execution order to actual ops (skip VARIABLE_INIT, PLACEHOLDER_SET)</li>
 *   <li>Build index maps for constants/variables/placeholders (external inputs)</li>
 *   <li>Assign sequential flat output slot indices per op output variable</li>
 *   <li>Build input wiring: look up each input variable in varToOutputIdx / external index maps</li>
 *   <li>Compute liveness: for each output slot, find last consumer step</li>
 *   <li>Build releaseAtStep[][] from liveness analysis</li>
 *   <li>Pre-allocate OpContext pool</li>
 *   <li>Return null if control flow ops detected (fall back to standard path)</li>
 * </ol>
 *
 * @see DynamicShapePlan
 * @see DynamicShapeSlot
 * @see DynamicShapePlanExecutor
 */
@Slf4j
public class DynamicShapePlanCompiler {

    // Ops with data-dependent output shapes that can't be cached
    private static final Set<String> DATA_DEPENDENT_OUTPUT_OPS = Set.of("where", "unique");

    /**
     * Ops whose output shape depends on input tensor VALUES, not just input shapes.
     * For these ops, the shape cache key must include INT/LONG input values.
     * For all other ops, INT/LONG values are excluded from the key, avoiding expensive
     * CUDA D2H sync and eliminating false cache misses from value changes.
     */
    private static final Set<String> VALUE_DEPENDENT_SHAPE_OPS = Set.of(
            "reshape", "reshape_no_copy",
            "expand_dims", "squeeze",
            "tile", "repeat",
            "slice", "strided_slice",
            "create", "fill",
            "onehot", "lin_space", "linspace", "range", "eye",
            "pad", "mirror_pad",
            "broadcast_to",
            "unique",
            "split_v", "split",
            "gather", "reduce_sum", "reduce_mean"
    );

    /**
     * Ops known to fully write every element of their output buffer. These ops can
     * safely skip zeroing of reused buffers, saving a CUDA kernel launch per allocation.
     *
     * <p>Conservative whitelist — only ops with absolute write guarantees are included.
     * View/shape ops (reshape, permute, transpose) are EXCLUDED because with
     * shapeFunctionOverride=true, C++ may not copy data to the pre-allocated output
     * buffer. Gather/concat/split/stack are EXCLUDED because they have complex memory
     * access patterns with potential edge cases.</p>
     */
    private static final Set<String> FULLY_WRITING_OPS = Set.of(
            // Matrix ops — BLAS contractually writes C[i,j] = sum(A[i,k]*B[k,j]) for all (i,j)
            "matmul", "mmul", "batched_gemm", "tensormmul", "xw_plus_b",
            // Elementwise binary — output[i] = f(a[i], b[i]) for every i (with broadcasting)
            "add", "subtract", "multiply", "divide", "floormod", "floordiv",
            "reversedivide", "reversesubtract", "squaredsubtract",
            "add_scalar", "subtract_scalar", "multiply_scalar", "divide_scalar",
            "pow", "min_pairwise", "max_pairwise", "atan2",
            // Elementwise unary — output[i] = f(input[i]) for every i
            "abs", "neg", "exp", "log", "log1p", "sqrt", "rsqrt", "square", "reciprocal",
            "ceil", "floor", "round", "sign", "erf", "erfc",
            // Activation functions — elementwise, writes every element
            "relu", "relu6", "leakyrelu", "elu", "selu", "gelu", "sigmoid", "tanh",
            "softsign", "softplus", "swish", "mish", "hard_sigmoid", "hardtanh",
            // Comparison ops — elementwise boolean output
            "equals", "not_equals", "less", "less_equal", "greater", "greater_equal",
            "boolean_and", "boolean_or", "boolean_not", "boolean_xor",
            // Reduction ops — fully computes every output element from input
            "reduce_sum", "reduce_mean", "reduce_max", "reduce_min", "reduce_prod",
            "reduce_norm1", "reduce_norm2", "reduce_logsumexp", "reduce_variance", "reduce_stdev",
            "sum", "mean", "max", "min", "prod", "norm1", "norm2", "normmax",
            "argmax", "argmin",
            // Softmax — normalizes every element across axis
            "softmax", "log_softmax",
            // Type conversion — converts every element
            "cast",
            // Clip — elementwise
            "clipbyvalue",
            // Data movement — each writes every element of output (views are zero-cost on GPU)
            "gather", "concat", "stack", "split", "unstack", "slice", "strided_slice",
            "reshape", "permute", "transpose", "expand_dims", "squeeze",
            // Shape/metadata — small scalar outputs, always fully written
            "shape_of", "size_at", "rank",
            // Array creation — allocates and fills entire output
            "create", "ones", "zeros", "fill", "range", "linspace",
            "ones_like", "zeros_like", "ones_as", "zeroslike",
            // Copy/tile — fully writes output
            "assign", "tile",
            // Indexing — write every element at target locations
            "set_scalar", "scatter_nd", "scatter_update",
            // Selection — writes every element (conditional)
            "where", "select",
            // Normalization — fully computed output
            "rms_norm", "layer_norm", "batch_norm",
            // Fused ops from GraphOptimizer
            "swish_mul", "dot_product_attention_v2", "kv_scatter", "token_sample",
            // Attention
            "onnx_multi_head_attention"
    );

    private DynamicShapePlanCompiler() {}

    /**
     * Compile a DynamicShapePlan from a ForwardExecutionDAG.
     *
     * @param sd             the SameDiff graph
     * @param dag            the forward execution DAG
     * @param requestedOutputs the output variable names
     * @return a compiled DynamicShapePlan, or null if control flow ops are detected
     */
    public static DynamicShapePlan compile(SameDiff sd, ForwardExecutionDAG dag,
                                            Set<String> requestedOutputs) {
        List<ExecutionNode> executionOrder = dag.getFrameAwareExecutionOrder();

        // Step 1: Filter to actual ops, detect control flow
        List<ExecutionNode> opNodes = new ArrayList<>();
        for (ExecutionNode node : executionOrder) {
            if (node.getNodeType() == ExecutionNode.ExecutionNodeType.VARIABLE_INIT ||
                    node.getNodeType() == ExecutionNode.ExecutionNodeType.PLACEHOLDER_SET) {
                continue;
            }
            if (node.getNodeType() == ExecutionNode.ExecutionNodeType.CONTROL_FLOW_OP) {
                log.debug("Control flow op detected ({}), falling back to standard path",
                        node.getOperationName());
                return null;
            }
            // Check for control flow ops by opName as well
            SameDiffOp sdOp = sd.getOps().get(node.getOperationName());
            if (sdOp != null && sdOp.getOp() != null) {
                String opNameLower = sdOp.getOp().opName().toLowerCase();
                if (opNameLower.equals("switch") || opNameLower.equals("merge") ||
                        opNameLower.equals("enter") || opNameLower.equals("exit") ||
                        opNameLower.equals("next_iteration") || opNameLower.equals("loop_cond")) {
                    log.debug("Control flow op detected by name ({}), falling back to standard path",
                            opNameLower);
                    return null;
                }
            }
            opNodes.add(node);
        }

        // Diagnostic tracking for needsZeroedOutput
        Map<String, Integer> needsZeroedOutputOps = new java.util.TreeMap<>();
        Map<String, Integer> skipsZeroedOutputOps = new java.util.TreeMap<>();

        // Step 2: Build external input index maps
        // External inputs = constants + variables + placeholders
        // They are referenced by negative indices: -(externalIndex + 1)
        List<String> externalInputKeys = new ArrayList<>();
        Map<String, Integer> externalIndexMap = new HashMap<>();

        for (String constName : dag.getConstants()) {
            int idx = externalInputKeys.size();
            externalInputKeys.add(constName);
            externalIndexMap.put(constName, idx);
        }
        for (String varName : dag.getVariables()) {
            if (!externalIndexMap.containsKey(varName)) {
                int idx = externalInputKeys.size();
                externalInputKeys.add(varName);
                externalIndexMap.put(varName, idx);
            }
        }
        for (String phName : dag.getRequiredPlaceholders()) {
            if (!externalIndexMap.containsKey(phName)) {
                int idx = externalInputKeys.size();
                externalInputKeys.add(phName);
                externalIndexMap.put(phName, idx);
            }
        }

        // Step 3: Assign sequential flat output slot indices per op output variable
        Map<String, Integer> varToOutputSlot = new HashMap<>();
        int nextSlotIndex = 0;

        for (ExecutionNode node : opNodes) {
            for (String outputVar : node.getOutputVariables()) {
                varToOutputSlot.put(outputVar, nextSlotIndex++);
            }
        }
        int totalOutputSlots = nextSlotIndex;

        // Step 4: Build DynamicShapeSlots with input wiring
        DynamicShapeSlot[] slots = new DynamicShapeSlot[opNodes.size()];
        // Track which step each output slot is produced at (for liveness)
        int[] slotProducerStep = new int[totalOutputSlots];
        Arrays.fill(slotProducerStep, -1);
        // Track last consumer step for each output slot
        int[] slotLastConsumerStep = new int[totalOutputSlots];
        Arrays.fill(slotLastConsumerStep, -1);

        for (int stepIdx = 0; stepIdx < opNodes.size(); stepIdx++) {
            ExecutionNode node = opNodes.get(stepIdx);
            SameDiffOp sdOp = sd.getOps().get(node.getOperationName());
            if (sdOp == null || sdOp.getOp() == null) {
                log.warn("Null op for node {}, falling back to standard path", node.getOperationName());
                return null;
            }
            DifferentialFunction op = sdOp.getOp();
            String opName = op.opName();
            if (opName == null) {
                opName = "unknown";
            }
            String opNameLower = opName.toLowerCase(Locale.ROOT);
            boolean isCustomOp = op instanceof CustomOp;

            // Detect legacy op type and opNum for ops not registered as
            // DeclarableOp in C++ (legacy transform, scalar, pairwise ops).
            // These will be constructed as LegacyOp wrappers in C++.
            int legacyOpType = DynamicShapeSlot.LEGACY_NONE;
            int legacyOpNum = -1;
            if (!isCustomOp) {
                if (op instanceof BaseTransformStrictOp) {
                    legacyOpType = DynamicShapeSlot.LEGACY_TRANSFORM_STRICT;
                    legacyOpNum = ((BaseTransformStrictOp) op).opNum();
                } else if (op instanceof BaseTransformSameOp) {
                    legacyOpType = DynamicShapeSlot.LEGACY_TRANSFORM_SAME;
                    legacyOpNum = ((BaseTransformSameOp) op).opNum();
                } else if (op instanceof BaseTransformFloatOp) {
                    legacyOpType = DynamicShapeSlot.LEGACY_TRANSFORM_FLOAT;
                    legacyOpNum = ((BaseTransformFloatOp) op).opNum();
                } else if (op instanceof BaseTransformBoolOp) {
                    legacyOpType = DynamicShapeSlot.LEGACY_TRANSFORM_BOOL;
                    legacyOpNum = ((BaseTransformBoolOp) op).opNum();
                } else if (op instanceof BaseScalarOp) {
                    legacyOpType = DynamicShapeSlot.LEGACY_SCALAR;
                    legacyOpNum = ((BaseScalarOp) op).opNum();
                }
            }

            // Build input wiring
            List<String> inputVars = node.getInputVariables();
            int numInputs = inputVars.size();
            int[] inputSourceIndices = new int[numInputs];
            byte[] inputSourceTypes = new byte[numInputs];
            String[] inputVarNames = new String[numInputs];

            boolean hasIntLongInputs = false;
            // Where with 3 inputs (condition, x, y) is element-wise select with static output shape.
            // Only Where with 1 input (coordinate extraction) has data-dependent variable-length output.
            boolean isDataDependent = DATA_DEPENDENT_OUTPUT_OPS.contains(opNameLower)
                    && !(opNameLower.equals("where") && numInputs == 3);

            for (int i = 0; i < numInputs; i++) {
                String inputVar = inputVars.get(i);
                inputVarNames[i] = inputVar;

                Integer outputSlot = varToOutputSlot.get(inputVar);
                if (outputSlot != null) {
                    // This input comes from another op's output
                    inputSourceIndices[i] = outputSlot;
                    inputSourceTypes[i] = DynamicShapeSlot.SOURCE_OP_OUTPUT;
                    // Update last consumer
                    if (stepIdx > slotLastConsumerStep[outputSlot]) {
                        slotLastConsumerStep[outputSlot] = stepIdx;
                    }
                } else {
                    // External input
                    Integer externalIdx = externalIndexMap.get(inputVar);
                    if (externalIdx == null) {
                        // Variable not in any map - might be from a suffix pattern
                        // Try stripping output index suffix (e.g., "op_name:1" -> "op_name")
                        String baseName = inputVar.contains(":") ?
                                inputVar.substring(0, inputVar.lastIndexOf(':')) : inputVar;
                        externalIdx = externalIndexMap.get(baseName);
                        outputSlot = varToOutputSlot.get(baseName);
                        if (outputSlot != null) {
                            inputSourceIndices[i] = outputSlot;
                            inputSourceTypes[i] = DynamicShapeSlot.SOURCE_OP_OUTPUT;
                            if (stepIdx > slotLastConsumerStep[outputSlot]) {
                                slotLastConsumerStep[outputSlot] = stepIdx;
                            }
                            continue;
                        }
                    }
                    if (externalIdx == null) {
                        // Last resort: add it as external
                        externalIdx = externalInputKeys.size();
                        externalInputKeys.add(inputVar);
                        externalIndexMap.put(inputVar, externalIdx);
                    }
                    inputSourceIndices[i] = -(externalIdx + 1);

                    // Determine source type
                    if (dag.getConstants().contains(inputVar)) {
                        inputSourceTypes[i] = DynamicShapeSlot.SOURCE_CONSTANT;
                    } else if (dag.getVariables().contains(inputVar)) {
                        inputSourceTypes[i] = DynamicShapeSlot.SOURCE_VARIABLE;
                    } else {
                        inputSourceTypes[i] = DynamicShapeSlot.SOURCE_PLACEHOLDER;
                    }
                }

                // Check for INT/LONG inputs (for sync flag)
                if (!hasIntLongInputs) {
                    org.nd4j.autodiff.samediff.SDVariable sdVar = sd.getVariable(inputVar);
                    if (sdVar != null) {
                        DataType dt = sdVar.dataType();
                        if (dt == DataType.INT || dt == DataType.LONG) {
                            hasIntLongInputs = true;
                        }
                    }
                }
            }

            // Build output wiring
            List<String> outputVars = node.getOutputVariables();
            int numOutputs = outputVars.size();
            int[] outputSlotIndices = new int[numOutputs];
            String[] outputVarNames = new String[numOutputs];
            for (int i = 0; i < numOutputs; i++) {
                String outputVar = outputVars.get(i);
                outputVarNames[i] = outputVar;
                Integer slot = varToOutputSlot.get(outputVar);
                outputSlotIndices[i] = slot != null ? slot : -1;
                if (slot != null) {
                    slotProducerStep[slot] = stepIdx;
                }
            }

            // Freeze op arguments
            long[] iArgs = new long[0];
            double[] tArgs = new double[0];
            boolean[] bArgs = new boolean[0];
            DataType[] dArgs = new DataType[0];
            if (isCustomOp && op instanceof DynamicCustomOp) {
                DynamicCustomOp dynOp = (DynamicCustomOp) op;
                iArgs = dynOp.iArgs();
                tArgs = dynOp.tArgs();
                bArgs = dynOp.bArgs();
                dArgs = dynOp.dArgs();
            } else if (op instanceof BaseScalarOp) {
                // Scalar ops store their scalar value separately (not as tArgs).
                // The C++ custom op equivalents (e.g., relu) expect it as tArg[0].
                BaseScalarOp scalarOp = (BaseScalarOp) op;
                if (scalarOp.scalar() != null) {
                    tArgs = new double[]{scalarOp.scalar().getDouble(0)};
                }
            }

            // Reduce ops store dimensions and keepDims separately from iArgs/bArgs.
            // The C++ custom op equivalents (e.g., reduce_sum) expect dimensions as
            // iArgs and keepDims as bArgs[0].
            if (op instanceof BaseReduceOp) {
                BaseReduceOp reduceOp = (BaseReduceOp) op;
                long[] dims = reduceOp.dimensionsArr();
                // If dimensions array is empty, try to get from dimensionz INDArray
                if (dims == null || dims.length == 0) {
                    INDArray dimensionz = reduceOp.dimensions();
                    if (dimensionz != null && !dimensionz.isEmpty()) {
                        dims = dimensionz.toLongVector();
                        // Also update the op's dimensions array so shape inference works correctly
                        reduceOp.setDimensions(dims);
                    }
                }
                if (dims != null && dims.length > 0) {
                    iArgs = dims;
                }
                bArgs = new boolean[]{reduceOp.isKeepDims()};
            }

            // Determine if all INT/LONG inputs are from external sources (constants/vars/placeholders).
            // When true, syncIntLongInputs() can skip commit() since external inputs don't need
            // GPU stream synchronization — they were loaded from CPU or synced before plan start.
            boolean allIntLongExternal = true;
            for (int i = 0; i < numInputs; i++) {
                if (inputSourceTypes[i] == DynamicShapeSlot.SOURCE_OP_OUTPUT) {
                    org.nd4j.autodiff.samediff.SDVariable sdVar = sd.getVariable(inputVarNames[i]);
                    if (sdVar != null) {
                        DataType dt = sdVar.dataType();
                        if (dt == DataType.INT || dt == DataType.LONG || dt == DataType.BOOL) {
                            allIntLongExternal = false;
                            break;
                        }
                    }
                }
            }

            // Determine if this op needs dynamic shape inference
            boolean requiresDynamic = false;
            if (op instanceof org.nd4j.linalg.api.ops.impl.shape.tensorops.BaseTensorOp) {
                requiresDynamic = true;
            }

            // Determine if this op needs zeroed output buffers.
            // Default: true (safe). Only skip for ops known to fully write every output element.
            boolean needsZeroedOutput = !FULLY_WRITING_OPS.contains(opNameLower) || isDataDependent;
            if (needsZeroedOutput) {
                needsZeroedOutputOps.merge(opNameLower, 1, Integer::sum);
            } else {
                skipsZeroedOutputOps.merge(opNameLower, 1, Integer::sum);
            }

            // Determine if output shape depends on input values (not just shapes).
            // When true, INT/LONG input values are included in the shape cache key.
            // When false, only input shapes + dtypes are used, avoiding expensive CUDA D2H syncs.
            boolean shapeDependsOnValues = VALUE_DEPENDENT_SHAPE_OPS.contains(opNameLower) || isDataDependent;

            // Pre-compute opName hash for shape key computation (avoids String.hashCode per step)
            long opNameHash = opName.hashCode() * 0x9E3779B97F4A7C15L;

            slots[stepIdx] = DynamicShapeSlot.builder()
                    .opName(opName)
                    .op(op)
                    .customOp(isCustomOp)
                    .legacyOpType(legacyOpType)
                    .legacyOpNum(legacyOpNum)
                    .inputSourceIndices(inputSourceIndices)
                    .inputSourceTypes(inputSourceTypes)
                    .inputVarNames(inputVarNames)
                    .outputSlotIndices(outputSlotIndices)
                    .outputVarNames(outputVarNames)
                    .iArgs(iArgs)
                    .tArgs(tArgs)
                    .bArgs(bArgs)
                    .dArgs(dArgs)
                    .needsIntLongSync(hasIntLongInputs || isDataDependent)
                    .allIntLongInputsExternal(allIntLongExternal)
                    .isDataDependent(isDataDependent)
                    .requiresDynamicShapeInference(requiresDynamic)
                    .needsZeroedOutput(needsZeroedOutput)
                    .outputShapeDependsOnInputValues(shapeDependsOnValues)
                    .stepIndex(stepIdx)
                    .opNameHash(opNameHash)
                    .inputArraysBuffer(new org.nd4j.linalg.api.ndarray.INDArray[numInputs])
                    .build();
        }

        // Step 5: Mark output slots that are final outputs as never releasable
        Set<Integer> finalOutputSlots = new HashSet<>();
        for (String outputName : requestedOutputs) {
            Integer slot = varToOutputSlot.get(outputName);
            if (slot != null) {
                finalOutputSlots.add(slot);
                // Don't release final outputs
                slotLastConsumerStep[slot] = Integer.MAX_VALUE;
            }
        }

        // Step 6: Build releaseAtStep[][] from liveness analysis
        // For each step, collect which output slots become dead after that step
        Map<Integer, List<Integer>> releaseMap = new HashMap<>();
        for (int slotIdx = 0; slotIdx < totalOutputSlots; slotIdx++) {
            int lastStep = slotLastConsumerStep[slotIdx];
            if (lastStep >= 0 && lastStep < opNodes.size() && !finalOutputSlots.contains(slotIdx)) {
                releaseMap.computeIfAbsent(lastStep, k -> new ArrayList<>()).add(slotIdx);
            }
        }

        int[][] releaseAtStep = new int[opNodes.size()][];
        for (int step = 0; step < opNodes.size(); step++) {
            List<Integer> toRelease = releaseMap.get(step);
            if (toRelease != null && !toRelease.isEmpty()) {
                releaseAtStep[step] = toRelease.stream().mapToInt(Integer::intValue).toArray();
            } else {
                releaseAtStep[step] = new int[0];
            }
        }

        // Step 6b: Compute dependency graph for async parallel execution.
        // For each step, determine which previous steps must complete before it can start
        // (predecessors), and which future steps depend on it (successors).
        // Also compute consumer counts per output slot for release tracking.
        int numSteps = opNodes.size();
        int[] predecessorCounts = new int[numSteps];
        int[][] predecessorsArr = new int[numSteps][];
        int[] consumerCounts = new int[totalOutputSlots];

        // Build predecessor edges: for each step's inputs, find the producing step
        @SuppressWarnings("unchecked")
        Set<Integer>[] predSets = new Set[numSteps];
        @SuppressWarnings("unchecked")
        List<Integer>[] succLists = new List[numSteps];
        for (int i = 0; i < numSteps; i++) {
            predSets[i] = new HashSet<>();
            succLists[i] = new ArrayList<>();
        }

        for (int stepIdx = 0; stepIdx < numSteps; stepIdx++) {
            DynamicShapeSlot slot = slots[stepIdx];
            int[] srcIndices = slot.getInputSourceIndices();
            for (int srcIdx : srcIndices) {
                if (srcIdx >= 0) {
                    // This input comes from output slot srcIdx
                    int producerStep = slotProducerStep[srcIdx];
                    if (producerStep >= 0 && producerStep != stepIdx) {
                        predSets[stepIdx].add(producerStep);
                    }
                    // Increment consumer count for this output slot
                    consumerCounts[srcIdx]++;
                }
            }
        }

        // Build successor lists and predecessor arrays
        List<Integer> rootSlotList = new ArrayList<>();
        for (int stepIdx = 0; stepIdx < numSteps; stepIdx++) {
            Set<Integer> preds = predSets[stepIdx];
            predecessorCounts[stepIdx] = preds.size();
            predecessorsArr[stepIdx] = preds.stream().mapToInt(Integer::intValue).toArray();
            if (preds.isEmpty()) {
                rootSlotList.add(stepIdx);
            }
            for (int pred : preds) {
                succLists[pred].add(stepIdx);
            }
        }

        int[][] successorsArr = new int[numSteps][];
        for (int i = 0; i < numSteps; i++) {
            successorsArr[i] = succLists[i].stream().mapToInt(Integer::intValue).toArray();
        }
        int[] rootSlots = rootSlotList.stream().mapToInt(Integer::intValue).toArray();

        // Mark final output slots as having MAX consumer count so they're never freed by async path
        for (int slotIdx : finalOutputSlots) {
            consumerCounts[slotIdx] = Integer.MAX_VALUE;
        }

        // Step 7: OpContext pool — executor uses a small rotating pool instead of
        // pre-allocating one per op (avoids native heap corruption from bulk close).
        OpContext[] opContextPool = new OpContext[0];

        // Step 8: Build output name → slot index map for O(1) output collection.
        // LinkedHashMap preserves insertion order (= requestedOutputs iteration order)
        // so serialize() writes indices in the same order as getRequestedOutputs().
        Map<String, Integer> outputNameToSlotIndex = new java.util.LinkedHashMap<>();
        for (String outputName : requestedOutputs) {
            Integer slot = varToOutputSlot.get(outputName);
            if (slot != null) {
                outputNameToSlotIndex.put(outputName, slot);
            }
        }

        log.debug("DynamicShapePlan compiled: {} ops, {} output slots, {} external inputs, {} final outputs, {} root slots",
                slots.length, totalOutputSlots, externalInputKeys.size(), requestedOutputs.size(), rootSlots.length);

        // Log needsZeroedOutput diagnostics
        int totalNeedsZeroed = needsZeroedOutputOps.values().stream().mapToInt(Integer::intValue).sum();
        int totalSkipsZeroed = skipsZeroedOutputOps.values().stream().mapToInt(Integer::intValue).sum();
        log.info("needsZeroedOutput: {} ops need zeroed output, {} ops skip zeroed output", totalNeedsZeroed, totalSkipsZeroed);
        if (!needsZeroedOutputOps.isEmpty()) {
            log.info("Ops still needing zeroed output: {}", needsZeroedOutputOps);
        }

        // Log full op type histogram for kernel count analysis
        Map<String, Integer> allOpTypes = new java.util.TreeMap<>();
        allOpTypes.putAll(needsZeroedOutputOps);
        for (var entry : skipsZeroedOutputOps.entrySet()) {
            allOpTypes.merge(entry.getKey(), entry.getValue(), Integer::sum);
        }
        log.info("Op type histogram ({} total): {}", slots.length, allOpTypes);

        return new DynamicShapePlan(
                slots,
                totalOutputSlots,
                releaseAtStep,
                opContextPool,
                externalInputKeys.toArray(new String[0]),
                requestedOutputs,
                outputNameToSlotIndex,
                false,
                predecessorCounts,
                predecessorsArr,
                successorsArr,
                consumerCounts,
                rootSlots
        );
    }
}

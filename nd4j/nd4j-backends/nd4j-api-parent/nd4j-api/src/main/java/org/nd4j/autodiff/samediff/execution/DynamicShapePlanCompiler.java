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
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.OpContext;
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
    private static final Set<String> DATA_DEPENDENT_OUTPUT_OPS = Set.of("Where", "unique");

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
            "unique"
    );

    /**
     * Ops known to fully write every element of their output buffer. These ops can
     * safely skip zeroing of reused buffers, saving a CUDA kernel launch per allocation.
     *
     * <p>Conservative whitelist — any op NOT in this set will have its output zeroed.
     * This is the safe default since stale buffer data in partially-written outputs
     * can cascade into native heap corruption.</p>
     */
    private static final Set<String> FULLY_WRITING_OPS = Set.of(
            // Elementwise arithmetic
            "add", "subtract", "multiply", "divide", "floormod", "floordiv",
            "reversedivide", "reversesubtract", "squaredsubtract",
            "add_scalar", "subtract_scalar", "multiply_scalar", "divide_scalar",
            // Elementwise math
            "abs", "neg", "exp", "log", "log1p", "sqrt", "rsqrt", "square", "reciprocal",
            "ceil", "floor", "round", "sign", "erf", "erfc",
            "pow", "min_pairwise", "max_pairwise", "atan2",
            // Activation functions
            "relu", "relu6", "leakyrelu", "elu", "selu", "gelu", "sigmoid", "tanh",
            "softsign", "softplus", "swish", "mish", "hard_sigmoid", "hardtanh",
            // Comparison ops
            "equals", "not_equals", "less", "less_equal", "greater", "greater_equal",
            "boolean_and", "boolean_or", "boolean_not", "boolean_xor",
            // Reduction ops
            "reduce_sum", "reduce_mean", "reduce_max", "reduce_min", "reduce_prod",
            "reduce_norm1", "reduce_norm2", "reduce_logsumexp", "reduce_variance", "reduce_stdev",
            "sum", "mean", "max", "min", "prod", "norm1", "norm2", "normmax",
            "argmax", "argmin",
            // Matrix ops
            "matmul", "mmul", "batched_gemm", "tensormmul",
            // Normalization
            "layer_norm", "layer_norm_bp",
            // Softmax
            "softmax", "log_softmax",
            // Type conversion
            "cast",
            // Shape ops (zero-copy or fully write)
            "reshape", "permute", "transpose", "expand_dims", "squeeze",
            "shape_of", "size", "rank", "reshape_no_copy",
            // Tensor creation (fully write)
            "ones_as", "zeros_as", "fill", "range", "create", "linspace",
            "eye", "ones_like", "zeros_like",
            // Slice/gather/concat (fully write)
            "concat", "stack", "unstack", "gather", "gather_nd", "split",
            "strided_slice", "slice", "tile", "repeat",
            // Embedding
            "embedding_lookup",
            // Broadcast
            "broadcast_to",
            // Assign
            "assign",
            // Where (select) - fully writes unlike Where (condition)
            "select",
            // Clip
            "clipbyvalue",
            // One-hot
            "onehot"
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
            boolean isCustomOp = op instanceof CustomOp;

            // Build input wiring
            List<String> inputVars = node.getInputVariables();
            int numInputs = inputVars.size();
            int[] inputSourceIndices = new int[numInputs];
            byte[] inputSourceTypes = new byte[numInputs];
            String[] inputVarNames = new String[numInputs];

            boolean hasIntLongInputs = false;
            boolean isDataDependent = DATA_DEPENDENT_OUTPUT_OPS.contains(op.opName());

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
            }

            // Determine if this op needs dynamic shape inference
            boolean requiresDynamic = false;
            if (op instanceof org.nd4j.linalg.api.ops.impl.shape.tensorops.BaseTensorOp) {
                requiresDynamic = true;
            }

            // Determine if this op needs zeroed output buffers.
            // Default: true (safe). Only skip for ops known to fully write every output element.
            boolean needsZeroedOutput = !FULLY_WRITING_OPS.contains(op.opName()) || isDataDependent;

            // Determine if output shape depends on input values (not just shapes).
            // When true, INT/LONG input values are included in the shape cache key.
            // When false, only input shapes + dtypes are used, avoiding expensive CUDA D2H syncs.
            boolean shapeDependsOnValues = VALUE_DEPENDENT_SHAPE_OPS.contains(op.opName()) || isDataDependent;

            // Pre-compute opName hash for shape key computation (avoids String.hashCode per step)
            long opNameHash = node.getOperationName().hashCode() * 0x9E3779B97F4A7C15L;

            slots[stepIdx] = DynamicShapeSlot.builder()
                    .opName(node.getOperationName())
                    .op(op)
                    .customOp(isCustomOp)
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

        // Step 7: OpContext pool — executor uses a small rotating pool instead of
        // pre-allocating one per op (avoids native heap corruption from bulk close).
        OpContext[] opContextPool = new OpContext[0];

        // Step 8: Build output name → slot index map for O(1) output collection
        Map<String, Integer> outputNameToSlotIndex = new HashMap<>();
        for (String outputName : requestedOutputs) {
            Integer slot = varToOutputSlot.get(outputName);
            if (slot != null) {
                outputNameToSlotIndex.put(outputName, slot);
            }
        }

        log.info("DynamicShapePlan compiled: {} ops, {} output slots, {} external inputs, {} final outputs",
                slots.length, totalOutputSlots, externalInputKeys.size(), requestedOutputs.size());

        return new DynamicShapePlan(
                slots,
                totalOutputSlots,
                releaseAtStep,
                opContextPool,
                externalInputKeys.toArray(new String[0]),
                requestedOutputs,
                outputNameToSlotIndex,
                false
        );
    }
}

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

import java.util.Arrays;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.BaseReduceOp;
import org.nd4j.linalg.api.ops.BaseScalarBoolOp;
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
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

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

    // Mirror of libnd4j/include/ops/declarable/OpDescriptor.h trait bitmask.
    // OpTraitTable.cpp is the single source of truth for per-op classification —
    // the Java compiler queries it via JNI (getOpTraits) so Java and C++ agree.
    private static final int OP_TRAIT_VALUE_DEPENDENT_SHAPE = 1 << 8;
    private static final int OP_TRAIT_DATA_DEPENDENT        = 1 << 9;

    // Cache: op-name → trait bitmask. Native table is immutable after initOpTraits();
    // memoise per-name to avoid JNI round-trips during plan compilation.
    private static final Map<String, Integer> OP_TRAIT_CACHE = new ConcurrentHashMap<>();

    private static int opTraitsOf(String opName) {
        if (opName == null) return 0;
        Integer cached = OP_TRAIT_CACHE.get(opName);
        if (cached != null) return cached;
        int traits = NativeOpsHolder.getInstance().getDeviceNativeOps().getOpTraits(opName);
        OP_TRAIT_CACHE.put(opName, traits);
        return traits;
    }

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
        // Use getExecutionOrder() (not getFrameAwareExecutionOrder()) to preserve
        // correct topological ordering for control flow ops. Frame-aware ordering
        // groups Enter ops into child frames, placing them after their Merge consumers.
        List<ExecutionNode> executionOrder = dag.getExecutionOrder();

        // Step 1: Filter to actual ops, classify control flow ops
        List<ExecutionNode> opNodes = new ArrayList<>();
        boolean hasControlFlow = false;
        for (ExecutionNode node : executionOrder) {
            if (node.getNodeType() == ExecutionNode.ExecutionNodeType.VARIABLE_INIT ||
                    node.getNodeType() == ExecutionNode.ExecutionNodeType.PLACEHOLDER_SET) {
                continue;
            }
            if (node.getNodeType() == ExecutionNode.ExecutionNodeType.CONTROL_FLOW_OP) {
                hasControlFlow = true;
            }
            // Also detect CF ops by opName
            SameDiffOp sdOp = sd.getOps().get(node.getOperationName());
            if (sdOp != null && sdOp.getOp() != null) {
                String opNameLower = sdOp.getOp().opName().toLowerCase();
                if (opNameLower.equals("switch") || opNameLower.equals("merge") ||
                        opNameLower.equals("enter") || opNameLower.equals("exit") ||
                        opNameLower.equals("next_iteration") || opNameLower.equals("loop_cond")) {
                    hasControlFlow = true;
                }
            }
            opNodes.add(node);
        }

        // Step 2: Build external input index maps
        // External inputs = constants + variables + placeholders
        // They are referenced by negative indices: -(externalIndex + 1)
        List<String> externalInputKeys = new ArrayList<>();
        List<Byte> externalInputSourceTypes = new ArrayList<>();
        Map<String, Integer> externalIndexMap = new HashMap<>();

        for (String constName : dag.getConstants()) {
            int idx = externalInputKeys.size();
            externalInputKeys.add(constName);
            externalInputSourceTypes.add(DynamicShapeSlot.SOURCE_CONSTANT);
            externalIndexMap.put(constName, idx);
        }
        for (String varName : dag.getVariables()) {
            if (!externalIndexMap.containsKey(varName)) {
                int idx = externalInputKeys.size();
                externalInputKeys.add(varName);
                externalInputSourceTypes.add(DynamicShapeSlot.SOURCE_VARIABLE);
                externalIndexMap.put(varName, idx);
            }
        }
        for (String phName : dag.getRequiredPlaceholders()) {
            if (!externalIndexMap.containsKey(phName)) {
                int idx = externalInputKeys.size();
                externalInputKeys.add(phName);
                externalInputSourceTypes.add(DynamicShapeSlot.SOURCE_PLACEHOLDER);
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
                } else if (op instanceof BaseScalarBoolOp) {
                    legacyOpType = DynamicShapeSlot.LEGACY_SCALAR_BOOL;
                    legacyOpNum = ((BaseScalarBoolOp) op).opNum();
                } else if (op instanceof BaseScalarOp) {
                    legacyOpType = DynamicShapeSlot.LEGACY_SCALAR;
                    legacyOpNum = ((BaseScalarOp) op).opNum();
                }
            }

            // Detect control flow type
            byte controlFlowType = DynamicShapeSlot.CF_NONE;
            if (node.getNodeType() == ExecutionNode.ExecutionNodeType.CONTROL_FLOW_OP ||
                    hasControlFlow) {
                switch (opNameLower) {
                    case "switch":         controlFlowType = DynamicShapeSlot.CF_SWITCH; break;
                    case "merge":          controlFlowType = DynamicShapeSlot.CF_MERGE; break;
                    case "enter":          controlFlowType = DynamicShapeSlot.CF_ENTER; break;
                    case "exit":           controlFlowType = DynamicShapeSlot.CF_EXIT; break;
                    case "next_iteration": controlFlowType = DynamicShapeSlot.CF_NEXT_ITERATION; break;
                    case "loop_cond":      controlFlowType = DynamicShapeSlot.CF_LOOP_COND; break;
                    default: break; // regular op in a graph with CF ops
                }
            }

            // Build input wiring
            List<String> inputVars = node.getInputVariables();
            int numInputs = inputVars.size();
            int[] inputSourceIndices = new int[numInputs];
            byte[] inputSourceTypes = new byte[numInputs];
            String[] inputVarNames = new String[numInputs];

            boolean hasIntLongInputs = false;

            for (int i = 0; i < numInputs; i++) {
                String inputVar = inputVars.get(i);
                inputVarNames[i] = inputVar;

                Integer outputSlot = varToOutputSlot.get(inputVar);
                if (outputSlot != null) {
                    // Guard against self-reference: an op cannot use its own output as input.
                    // This would create a circular dependency (slot N reads from outputSlot[N]).
                    // Check if outputSlot belongs to this step's op outputs.
                    boolean isSelfRef = false;
                    List<String> myOutputs = node.getOutputVariables();
                    for (String myOut : myOutputs) {
                        Integer mySlot = varToOutputSlot.get(myOut);
                        if (mySlot != null && mySlot.equals(outputSlot)) {
                            isSelfRef = true;
                            break;
                        }
                    }
                    if (isSelfRef) {
                        log.warn("DSP compile: self-reference detected at step {} (op {}): " +
                                "input '{}' resolves to own output slot {}. Falling back to standard path.",
                                stepIdx, opName, inputVar, outputSlot);
                        return null;
                    }
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
                            // Self-reference guard for baseName fallback
                            boolean isSelfRef2 = false;
                            List<String> myOutputs2 = node.getOutputVariables();
                            for (String myOut : myOutputs2) {
                                Integer mySlot = varToOutputSlot.get(myOut);
                                if (mySlot != null && mySlot.equals(outputSlot)) {
                                    isSelfRef2 = true;
                                    break;
                                }
                            }
                            if (isSelfRef2) {
                                log.warn("DSP compile: self-reference detected at step {} (op {}): " +
                                        "input '{}' (baseName '{}') resolves to own output slot {}. Falling back.",
                                        stepIdx, opName, inputVar, baseName, outputSlot);
                                return null;
                            }
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

                    // Determine source type — check both inputVar and baseName
                    // (inputVar may have ":0" suffix that constants/variables don't)
                    String baseName2 = inputVar.contains(":") ?
                            inputVar.substring(0, inputVar.lastIndexOf(':')) : inputVar;
                    if (dag.getConstants().contains(inputVar) || dag.getConstants().contains(baseName2)) {
                        inputSourceTypes[i] = DynamicShapeSlot.SOURCE_CONSTANT;
                    } else if (dag.getVariables().contains(inputVar) || dag.getVariables().contains(baseName2)) {
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
            String[] sArgs = new String[0];
            if (isCustomOp && op instanceof DynamicCustomOp) {
                DynamicCustomOp dynOp = (DynamicCustomOp) op;
                iArgs = dynOp.iArgs();
                tArgs = dynOp.tArgs();
                bArgs = dynOp.bArgs();
                dArgs = dynOp.dArgs();
                sArgs = dynOp.sArgs();

                // strided_slice: iArgs are [beginMask, ellipsisMask, endMask, newAxisMask, shrinkAxisMask].
                // When begin/end/strides come as input tensors (numInputs > 1), the Java op may also
                // store them in iArgs[5+]. The C++ op checks iArgs.size() > 5 to decide between
                // static path (read from iArgs) vs dynamic path (read from input tensors).
                // If we pass all iArgs, the C++ op takes the static path with potentially stale values.
                // Cap to 5 mask values so the C++ op correctly reads from input tensors.
                if ("strided_slice".equals(opNameLower) && iArgs.length > 5 && numInputs > 1) {
                    iArgs = Arrays.copyOf(iArgs, 5);
                }
            } else if (op instanceof BaseScalarOp) {
                // Scalar ops store their scalar value separately (not as tArgs).
                // The C++ custom op equivalents (e.g., relu) expect it as tArg[0].
                BaseScalarOp scalarOp = (BaseScalarOp) op;
                if (scalarOp.scalar() != null) {
                    tArgs = new double[]{scalarOp.scalar().getDouble(0)};
                }
            } else if (op instanceof BaseScalarBoolOp) {
                // Scalar bool ops (gt, lt, eq, etc.) also store scalar separately.
                // BaseScalarBoolOp extends BaseOp (NOT BaseScalarOp), so needs its own branch.
                BaseScalarBoolOp scalarBoolOp = (BaseScalarBoolOp) op;
                if (scalarBoolOp.scalar() != null) {
                    tArgs = new double[]{scalarBoolOp.scalar().getDouble(0)};
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

            // Consult the C++ OpTraitTable (single source of truth) rather than the
            // crude "any int/long input → value-dependent" heuristic. Only ops actually
            // carrying VALUE_DEPENDENT_SHAPE or DATA_DEPENDENT traits stamp this flag;
            // e.g. gather takes INT indices but its output shape is derivable from shapes,
            // so it must NOT be flagged value-dependent.
            int opTraits = opTraitsOf(opName);
            boolean shapeDependsOnValues;
            if (opTraits != 0) {
                shapeDependsOnValues = (opTraits &
                        (OP_TRAIT_VALUE_DEPENDENT_SHAPE | OP_TRAIT_DATA_DEPENDENT)) != 0;
            } else {
                // Op not in trait table (e.g. legacy / unregistered) — fall back to the
                // old heuristic so we stay safe on unclassified ops.
                shapeDependsOnValues = hasIntLongInputs;
            }

            DifferentialFunction slotOp = op;

            // Pre-compute opName hash for shape key computation (avoids String.hashCode per step)
            long opNameHash = opName.hashCode() * 0x9E3779B97F4A7C15L;

            slots[stepIdx] = DynamicShapeSlot.builder()
                    .opName(opName)
                    .op(slotOp)
                    .customOp(isCustomOp)
                    .legacyOpType(legacyOpType)
                    .legacyOpNum(legacyOpNum)
                    .controlFlowType(controlFlowType)
                    .inputSourceIndices(inputSourceIndices)
                    .inputSourceTypes(inputSourceTypes)
                    .inputVarNames(inputVarNames)
                    .outputSlotIndices(outputSlotIndices)
                    .outputVarNames(outputVarNames)
                    .iArgs(iArgs)
                    .tArgs(tArgs)
                    .bArgs(bArgs)
                    .dArgs(dArgs)
                    .sArgs(sArgs)
                    .needsIntLongSync(hasIntLongInputs)
                    .allIntLongInputsExternal(allIntLongExternal)
                    .requiresDynamicShapeInference(requiresDynamic)
                    .outputShapeDependsOnInputValues(shapeDependsOnValues)
                    .stepIndex(stepIdx)
                    .opNameHash(opNameHash)
                    .inputArraysBuffer(new org.nd4j.linalg.api.ndarray.INDArray[numInputs])
                    .build();

            // Attention op diagnostics: log all inputs with their sources, types, and shapes
            if ("onnx_multi_head_attention".equals(opNameLower) || "dot_product_attention_v2".equals(opNameLower)) {
                log.info("ATTN_OP_COMPILE: step={} op={} slot={} numInputs={} numOutputs={}",
                        stepIdx, opName, stepIdx, numInputs, numOutputs);
                for (int i = 0; i < numInputs; i++) {
                    int srcIdx = inputSourceIndices[i];
                    String srcType = inputSourceTypes[i] == DynamicShapeSlot.SOURCE_OP_OUTPUT ? "OP_OUTPUT" :
                            inputSourceTypes[i] == DynamicShapeSlot.SOURCE_CONSTANT ? "CONSTANT" :
                            inputSourceTypes[i] == DynamicShapeSlot.SOURCE_VARIABLE ? "VARIABLE" :
                            inputSourceTypes[i] == DynamicShapeSlot.SOURCE_PLACEHOLDER ? "PLACEHOLDER" : "UNKNOWN";
                    String extName = "";
                    String shapeStr = "?";
                    if (srcIdx < 0) {
                        int extIdx = -(srcIdx + 1);
                        extName = extIdx < externalInputKeys.size() ? externalInputKeys.get(extIdx) : "OUT_OF_RANGE";
                        // Try to get the actual array shape
                        INDArray arr = sd.getArrForVarName(extName);
                        if (arr == null) {
                            // Try with :0 suffix stripped
                            String base = extName.contains(":") ? extName.substring(0, extName.lastIndexOf(':')) : extName;
                            arr = sd.getArrForVarName(base);
                        }
                        shapeStr = arr != null ? Arrays.toString(arr.shape()) + " len=" + arr.length() : "NULL";
                    }
                    log.info("ATTN_OP_COMPILE:   input[{}] var='{}' srcIdx={} srcType={} extName='{}' shape={}",
                            i, inputVarNames[i], srcIdx, srcType, extName, shapeStr);
                }
            }
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

        // Step 6c: Detect loop regions and wire control flow
        List<DynamicShapePlan.LoopRegion> loopRegionsList = new ArrayList<>();
        if (hasControlFlow) {
            for (int stepIdx = 0; stepIdx < numSteps; stepIdx++) {
                DynamicShapeSlot slot = slots[stepIdx];
                if (slot.getControlFlowType() != DynamicShapeSlot.CF_MERGE) continue;

                // Find NextIteration that feeds back to this Merge
                int nextIterStep = -1;
                for (int cs = stepIdx + 1; cs < numSteps; cs++) {
                    if (slots[cs].getControlFlowType() == DynamicShapeSlot.CF_NEXT_ITERATION) {
                        int[] niOutputs = slots[cs].getOutputSlotIndices();
                        for (int niOut : niOutputs) {
                            for (int mi : slot.getInputSourceIndices()) {
                                if (mi >= 0 && mi == niOut) { nextIterStep = cs; break; }
                            }
                            if (nextIterStep >= 0) break;
                        }
                        if (nextIterStep >= 0) break;
                    }
                }
                if (nextIterStep < 0) continue; // if-else Merge, not a loop

                int switchStep = -1, exitStep = -1;
                for (int s = stepIdx + 1; s <= nextIterStep; s++) {
                    if (slots[s].getControlFlowType() == DynamicShapeSlot.CF_SWITCH && switchStep < 0)
                        switchStep = s;
                    if (slots[s].getControlFlowType() == DynamicShapeSlot.CF_EXIT)
                        exitStep = s;
                }
                // Exit may be after NextIteration (false branch of Switch)
                if (exitStep < 0) {
                    for (int s = nextIterStep + 1; s < numSteps; s++) {
                        if (slots[s].getControlFlowType() == DynamicShapeSlot.CF_EXIT) {
                            exitStep = s; break;
                        }
                    }
                }
                if (switchStep < 0) continue;

                int bodyStart = switchStep + 1;
                int bodyEnd = nextIterStep;
                int regionIdx = loopRegionsList.size();

                loopRegionsList.add(new DynamicShapePlan.LoopRegion(
                        stepIdx, switchStep, nextIterStep,
                        exitStep >= 0 ? exitStep : nextIterStep + 1,
                        bodyStart, bodyEnd));

                slots[nextIterStep].setLoopBackTarget(stepIdx);
                for (int s = stepIdx; s <= Math.max(nextIterStep, exitStep >= 0 ? exitStep : nextIterStep); s++) {
                    slots[s].setLoopRegionIndex(regionIdx);
                }

                // Defer release of loop body output slots to after Exit
                Set<Integer> loopBodyOutputSlots = new HashSet<>();
                for (int s = stepIdx; s <= nextIterStep; s++) {
                    for (int oi : slots[s].getOutputSlotIndices()) {
                        if (oi >= 0 && !finalOutputSlots.contains(oi))
                            loopBodyOutputSlots.add(oi);
                    }
                }
                for (int s = 0; s < numSteps; s++) {
                    if (releaseAtStep[s].length > 0) {
                        List<Integer> filtered = new ArrayList<>();
                        for (int r : releaseAtStep[s]) {
                            if (!loopBodyOutputSlots.contains(r)) filtered.add(r);
                        }
                        if (filtered.size() != releaseAtStep[s].length)
                            releaseAtStep[s] = filtered.stream().mapToInt(Integer::intValue).toArray();
                    }
                }
                int releaseStep = exitStep >= 0 ? exitStep : nextIterStep;
                if (releaseStep < numSteps) {
                    Set<Integer> combined = new LinkedHashSet<>();
                    for (int r : releaseAtStep[releaseStep]) combined.add(r);
                    combined.addAll(loopBodyOutputSlots);
                    releaseAtStep[releaseStep] = combined.stream().mapToInt(Integer::intValue).toArray();
                }

                log.debug("Loop region {}: merge={}, switch={}, nextIter={}, exit={}, body=[{}-{}]",
                        regionIdx, stepIdx, switchStep, nextIterStep, exitStep, bodyStart, bodyEnd);
            }
        }
        DynamicShapePlan.LoopRegion[] loopRegions = loopRegionsList.isEmpty() ? null :
                loopRegionsList.toArray(new DynamicShapePlan.LoopRegion[0]);

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

        // Log external input discovery for frozen tracking / lineage diagnostics
        int placeholderCount = 0, constantCount = 0, variableCount = 0;
        for (int i = 0; i < externalInputKeys.size(); i++) {
            byte srcType = externalInputSourceTypes.get(i);
            if (srcType == DynamicShapeSlot.SOURCE_PLACEHOLDER) placeholderCount++;
            else if (srcType == DynamicShapeSlot.SOURCE_CONSTANT) constantCount++;
            else if (srcType == DynamicShapeSlot.SOURCE_VARIABLE) variableCount++;
        }
        log.info("EXT_INPUT_DISCOVER: {} total ({} constants, {} variables, {} placeholders)",
                externalInputKeys.size(), constantCount, variableCount, placeholderCount);
        // Log placeholder names (these are the dynamic inputs that change per step)
        for (int i = 0; i < externalInputKeys.size(); i++) {
            if (externalInputSourceTypes.get(i) == DynamicShapeSlot.SOURCE_PLACEHOLDER) {
                log.info("EXT_INPUT_DISCOVER: placeholder idx={} name=\"{}\"", i, externalInputKeys.get(i));
            }
        }
        // Log CONSTANT and VARIABLE ext inputs that have empty/null arrays (these are problematic
        // for attention ops where inputs 4-5 should be past_key/past_value but are empty constants)
        for (int i = 0; i < externalInputKeys.size(); i++) {
            byte srcType = externalInputSourceTypes.get(i);
            if (srcType == DynamicShapeSlot.SOURCE_CONSTANT || srcType == DynamicShapeSlot.SOURCE_VARIABLE) {
                String extName = externalInputKeys.get(i);
                INDArray arr = sd.getArrForVarName(extName);
                if (arr == null) {
                    String base = extName.contains(":") ? extName.substring(0, extName.lastIndexOf(':')) : extName;
                    arr = sd.getArrForVarName(base);
                }
                if (arr == null || arr.isEmpty() || arr.length() == 0) {
                    String typeStr = srcType == DynamicShapeSlot.SOURCE_CONSTANT ? "CONSTANT" : "VARIABLE";
                    log.info("EXT_INPUT_DISCOVER: EMPTY_{} idx={} name=\"{}\" shape={}",
                            typeStr, i, extName,
                            arr != null ? java.util.Arrays.toString(arr.shape()) : "[]");
                }
            }
        }

        // Log op type histogram
        Map<String, Integer> allOpTypes = new java.util.TreeMap<>();
        for (DynamicShapeSlot slot : slots) {
            allOpTypes.merge(slot.getOpName(), 1, Integer::sum);
        }
        log.info("Op type histogram ({} total): {}", slots.length, allOpTypes);

        // Build source types byte array from list
        byte[] extSourceTypes = new byte[externalInputSourceTypes.size()];
        for (int i = 0; i < extSourceTypes.length; i++) {
            extSourceTypes[i] = externalInputSourceTypes.get(i);
        }

        DynamicShapePlan plan = new DynamicShapePlan(
                slots,
                totalOutputSlots,
                releaseAtStep,
                opContextPool,
                externalInputKeys.toArray(new String[0]),
                extSourceTypes,
                requestedOutputs,
                outputNameToSlotIndex,
                hasControlFlow,
                loopRegions,
                predecessorCounts,
                predecessorsArr,
                successorsArr,
                consumerCounts,
                rootSlots
        );

        if (log.isDebugEnabled()) {
            log.debug("Plan structure:\n{}", PlanIntrospection.formatPlan(plan));
        }

        // Device placement planning (disabled by default)
        if (Boolean.getBoolean(ND4JSystemProperties.PLACEMENT_ENABLED)
                || sd.getPlacementStrategy() != null) {
            DevicePlacementPlanner.PlacementStrategy strategy = sd.getPlacementStrategy() != null
                    ? sd.getPlacementStrategy()
                    : DevicePlacementPlanner.PlacementStrategy.valueOf(System.getProperty(
                    ND4JSystemProperties.PLACEMENT_STRATEGY, "SINGLE_DEVICE"));
            DevicePlacementPlanner.PlacementPlan placement = DevicePlacementPlanner.plan(plan, sd, strategy);
            DevicePlacementPlanner.applyPlan(plan, placement);
        }

        return plan;
    }
}

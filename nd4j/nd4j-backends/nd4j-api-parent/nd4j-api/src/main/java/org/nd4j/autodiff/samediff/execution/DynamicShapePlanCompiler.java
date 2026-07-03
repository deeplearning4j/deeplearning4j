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
import org.nd4j.autodiff.samediff.diagnostics.DspDiagnostics;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.BaseBroadcastBoolOp;
import org.nd4j.linalg.api.ops.BaseBroadcastOp;
import org.nd4j.linalg.api.ops.BaseIndexAccumulation;
import org.nd4j.linalg.api.ops.BaseReduceBoolOp;
import org.nd4j.linalg.api.ops.BaseReduceFloatOp;
import org.nd4j.linalg.api.ops.BaseReduceLongOp;
import org.nd4j.linalg.api.ops.BaseReduceOp;
import org.nd4j.linalg.api.ops.BaseReduceSameOp;
import org.nd4j.linalg.api.ops.BaseScalarBoolOp;
import org.nd4j.linalg.api.ops.BaseScalarOp;
import org.nd4j.linalg.api.ops.BaseTransformBoolOp;
import org.nd4j.linalg.api.ops.BaseTransformFloatOp;
import org.nd4j.linalg.api.ops.BaseTransformSameOp;
import org.nd4j.linalg.api.ops.BaseTransformStrictOp;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.shape.tensorops.BaseTensorOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.impl.reduce3.BaseReduce3Op;
import org.nd4j.linalg.api.ops.impl.summarystats.Variance;
import org.nd4j.linalg.api.ops.random.BaseRandomOp;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.ExternalErrorsFunction;
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

        // Pre-check: if the entire SameDiff graph (not just the execution DAG) contains
        // ExternalErrorsFunction, fall back to standard path. ExternalErrorsFunction is a
        // LOGIC op that provides externally-supplied gradients; it is handled exclusively
        // by Java-side InferenceSession logic and must never reach the native DSP executor.
        // It may not appear in the minimal execution DAG (which is built only from ops
        // reachable from the requested outputs), so we must scan the full op map here.
        for (SameDiffOp sdOpFull : sd.getOps().values()) {
            if (sdOpFull.getOp() instanceof ExternalErrorsFunction) {
                log.debug("DSP compile: SameDiff graph contains ExternalErrorsFunction op '{}'. " +
                        "Falling back to standard path.", sdOpFull.getName());
                return null;
            }
        }

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
            // Also detect CF ops and sub-graph delegation ops by opName
            SameDiffOp sdOp = sd.getOps().get(node.getOperationName());
            if (sdOp != null && sdOp.getOp() != null) {
                String opNameLower = sdOp.getOp().opName().toLowerCase();
                if (opNameLower.equals("switch") || opNameLower.equals("merge") ||
                        opNameLower.equals("enter") || opNameLower.equals("exit") ||
                        opNameLower.equals("next_iteration") || opNameLower.equals("loop_cond")) {
                    hasControlFlow = true;
                }
                // invoke delegates to a sub-graph — the native executor has no shape
                // function for it, so DSP compilation must bail out to standard path.
                if (opNameLower.equals("invoke")) {
                    log.debug("DSP compile: graph contains 'invoke' op (sub-graph delegation) " +
                            "which has no native shape function. Falling back to standard path.");
                    return null;
                }
                // Tensor array ops (create_list, stack_list, etc.) are meta-ops handled
                // by special Java-side logic in InferenceSession. The native executor
                // has no shape functions for them.
                if (sdOp.getOp() instanceof BaseTensorOp) {
                    log.debug("DSP compile: graph contains tensor array op '{}'. " +
                            "Falling back to standard path.", opNameLower);
                    return null;
                }
                // Random ops with no input variables carry their shape as constructor
                // arguments. The native executor's shape inference calls INPUT_VARIABLE(0)
                // which fails with "Can't find requested variable by index: 0".
                if (sdOp.getOp() instanceof BaseRandomOp) {
                    List<String> inputs = sdOp.getInputsToOp();
                    if (inputs == null || inputs.isEmpty()) {
                        log.debug("DSP compile: random op '{}' has no input variables. " +
                                "Falling back to standard path.", opNameLower);
                        return null;
                    }
                }
                // ExternalErrorsFunction is a LOGIC op handled by special Java-side logic
                // in InferenceSession. The native executor has no implementation for it.
                // Graphs containing ExternalErrorsFunction (e.g. gradient graphs built with
                // SameDiffUtils.externalErrors) must fall back to the standard path.
                if (sdOp.getOp() instanceof ExternalErrorsFunction) {
                    log.debug("DSP compile: graph contains ExternalErrorsFunction op. " +
                            "Falling back to standard path.");
                    return null;
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

        // Step 3: Assign sequential flat output slot indices per op output variable.
        // Two different ops MUST NOT produce the same output variable name — that
        // would alias their output slots and let one op's buffer be read as another's
        // (e.g. a [512,2] scatter_nd_update buffer flattened into a gather indices
        // input, producing OOB index errors). Detect collisions explicitly and bail
        // to the standard path rather than compiling a silently-aliased plan.
        Map<String, Integer> varToOutputSlot = new HashMap<>();
        // Reverse map only used when a collision fires, to identify the prior producer.
        Map<Integer, String> slotToProducerOp = new HashMap<>();
        int nextSlotIndex = 0;

        for (int producerStep = 0; producerStep < opNodes.size(); producerStep++) {
            ExecutionNode node = opNodes.get(producerStep);
            String producerOpName = node.getOperationName();
            for (String outputVar : node.getOutputVariables()) {
                Integer prevSlot = varToOutputSlot.put(outputVar, nextSlotIndex);
                if (prevSlot != null) {
                    String prevOp = slotToProducerOp.get(prevSlot);
                    log.error("DSP compile: output variable name collision — '{}' was already "
                            + "assigned to slot {} by op '{}', now reassigned to slot {} by op "
                            + "'{}' at step {}. Plan would silently alias slots; falling back "
                            + "to standard execution path.",
                            outputVar, prevSlot, prevOp,
                            nextSlotIndex, producerOpName, producerStep);
                    return null;
                }
                slotToProducerOp.put(nextSlotIndex, producerOpName);
                nextSlotIndex++;
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
                    // BaseTransformSameOp with 2 array inputs (e.g. CompareAndReplace, which
                    // extends BaseTransformSameOp but has both x and y) must use the pairwise
                    // execution path (execPairwiseTransform). With a single input it is a
                    // standard single-input transform (execTransformSame).
                    // NativeOpExecutioner makes this same distinction via op.y() != null.
                    if (node.getInputVariables().size() >= 2) {
                        legacyOpType = DynamicShapeSlot.LEGACY_PAIRWISE_TRANSFORM;
                    } else {
                        legacyOpType = DynamicShapeSlot.LEGACY_TRANSFORM_SAME;
                    }
                    legacyOpNum = ((BaseTransformSameOp) op).opNum();
                } else if (op instanceof BaseTransformFloatOp) {
                    legacyOpType = DynamicShapeSlot.LEGACY_TRANSFORM_FLOAT;
                    legacyOpNum = ((BaseTransformFloatOp) op).opNum();
                } else if (op instanceof BaseTransformBoolOp && ((BaseTransformBoolOp) op).opType() == Op.Type.PAIRWISE_BOOL) {
                    legacyOpType = DynamicShapeSlot.LEGACY_PAIRWISE_BOOL;
                    legacyOpNum = ((BaseTransformBoolOp) op).opNum();
                } else if (op instanceof BaseTransformBoolOp) {
                    legacyOpType = DynamicShapeSlot.LEGACY_TRANSFORM_BOOL;
                    legacyOpNum = ((BaseTransformBoolOp) op).opNum();
                } else if (op instanceof BaseScalarBoolOp) {
                    legacyOpType = DynamicShapeSlot.LEGACY_SCALAR_BOOL;
                    legacyOpNum = ((BaseScalarBoolOp) op).opNum();
                } else if (op instanceof BaseScalarOp) {
                    legacyOpType = DynamicShapeSlot.LEGACY_SCALAR;
                    legacyOpNum = ((BaseScalarOp) op).opNum();
                } else if (op instanceof BaseReduce3Op) {
                    // Must precede BaseReduceFloatOp — BaseReduce3Op extends BaseReduceFloatOp.
                    legacyOpType = DynamicShapeSlot.LEGACY_REDUCE_3;
                    legacyOpNum = ((BaseReduce3Op) op).opNum();
                } else if (op instanceof BaseReduceFloatOp) {
                    legacyOpType = DynamicShapeSlot.LEGACY_REDUCE_FLOAT;
                    legacyOpNum = ((BaseReduceFloatOp) op).opNum();
                } else if (op instanceof BaseReduceSameOp) {
                    legacyOpType = DynamicShapeSlot.LEGACY_REDUCE_SAME;
                    legacyOpNum = ((BaseReduceSameOp) op).opNum();
                } else if (op instanceof BaseReduceBoolOp) {
                    legacyOpType = DynamicShapeSlot.LEGACY_REDUCE_BOOL;
                    legacyOpNum = ((BaseReduceBoolOp) op).opNum();
                } else if (op instanceof BaseReduceLongOp) {
                    legacyOpType = DynamicShapeSlot.LEGACY_REDUCE_LONG;
                    legacyOpNum = ((BaseReduceLongOp) op).opNum();
                } else if (op instanceof Variance) {
                    // Variance (and subclass StandardDeviation) extend BaseReduceOp directly,
                    // bypassing the typed BaseReduce{Float,Same,Bool,Long} branches.
                    legacyOpType = DynamicShapeSlot.LEGACY_STATS;
                    legacyOpNum = ((Variance) op).opNum();
                } else if (op instanceof BaseIndexAccumulation) {
                    legacyOpType = DynamicShapeSlot.LEGACY_INDEX_REDUCE;
                    legacyOpNum = ((BaseIndexAccumulation) op).opNum();
                } else if (op instanceof BaseBroadcastBoolOp) {
                    // Must precede BaseBroadcastOp in case of any hierarchy overlap.
                    legacyOpType = DynamicShapeSlot.LEGACY_BROADCAST_BOOL;
                    legacyOpNum = ((BaseBroadcastBoolOp) op).opNum();
                } else if (op instanceof BaseBroadcastOp) {
                    legacyOpType = DynamicShapeSlot.LEGACY_BROADCAST;
                    legacyOpNum = ((BaseBroadcastOp) op).opNum();
                } else if (op instanceof BaseRandomOp) {
                    legacyOpType = DynamicShapeSlot.LEGACY_RANDOM;
                    legacyOpNum = ((BaseRandomOp) op).opNum();
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
                        // Last resort: add it as external.
                        // Log a warning — this variable's producing op was expected to be
                        // in the plan's execution subgraph, but it wasn't found in
                        // varToOutputSlot. For VariableType.ARRAY variables, this means
                        // ext input resolution will fall back to sd.getVariable().getArr()
                        // which may return a stale/zero-initialized value.
                        Variable varMeta = sd.getVariables().get(inputVar);
                        VariableType vType = varMeta != null ? varMeta.getVariable().getVariableType() : null;
                        String outputOfOp = varMeta != null ? varMeta.getOutputOfOp() : null;
                        if (vType == VariableType.ARRAY || (outputOfOp != null && !outputOfOp.isEmpty())) {
                            log.warn("DSP compile: {} variable '{}' (produced by op '{}') at step {} " +
                                    "(op '{}') input[{}] has no output slot and is not in external maps. " +
                                    "Adding as last-resort external input ext[{}]. This will resolve " +
                                    "via getArr() which may be stale/zero on first execution.",
                                    vType, inputVar, outputOfOp, stepIdx, opName, i, externalInputKeys.size());
                        }
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
                // Also check for string-type inputs (UTF8/UTF16/UTF32): the native executor
                // calls buffer() on all external inputs, which calls BUILD_SINGLE_SELECTOR
                // with SD_COMMON_TYPES_ALL — string types are excluded from that macro and
                // will throw "Unexpected error with data type UTF8". Fall back to standard
                // execution so the string constants pass through the Java-side op as-is.
                org.nd4j.autodiff.samediff.SDVariable sdVar = sd.getVariable(inputVar);
                if (sdVar != null) {
                    DataType dt = sdVar.dataType();
                    if (dt == DataType.INT || dt == DataType.LONG) {
                        hasIntLongInputs = true;
                    }
                    if (dt == DataType.UTF8 || dt == DataType.UTF16 || dt == DataType.UTF32) {
                        log.debug("DSP compile: op '{}' at step {} has string-type input '{}' ({}). " +
                                "Native executor cannot handle string buffers. Falling back to standard path.",
                                opName, stepIdx, inputVar, dt);
                        return null;
                    }
                }
            }

            // Diagnostic: trace gather op wiring so we can identify the producer
            // variable when slot-1268-style OOB index errors fire at runtime.
            if ("gather".equals(opNameLower)) {
                // Resolve the first output's slot index so we can correlate this
                // log line with the C++ slot index that fires errors.
                int firstOutSlot = -1;
                List<String> myOuts = node.getOutputVariables();
                if (!myOuts.isEmpty()) {
                    Integer s = varToOutputSlot.get(myOuts.get(0));
                    if (s != null) firstOutSlot = s;
                }
                StringBuilder sb = new StringBuilder();
                sb.append("DSP_GATHER_COMPILE step=").append(stepIdx);
                sb.append(" outSlot=").append(firstOutSlot);
                sb.append(" op=").append(opName);
                for (int i = 0; i < numInputs; i++) {
                    int srcIdx = inputSourceIndices[i];
                    String srcDesc;
                    if (srcIdx >= 0) {
                        srcDesc = "OP_OUTPUT[slot=" + srcIdx + "]";
                    } else {
                        int extIdx = -(srcIdx + 1);
                        String extKey = extIdx < externalInputKeys.size() ?
                                externalInputKeys.get(extIdx) : "?";
                        byte t = inputSourceTypes[i];
                        String typeStr = t == DynamicShapeSlot.SOURCE_CONSTANT ? "CONSTANT" :
                                t == DynamicShapeSlot.SOURCE_VARIABLE ? "VARIABLE" :
                                t == DynamicShapeSlot.SOURCE_PLACEHOLDER ? "PLACEHOLDER" : "?";
                        srcDesc = typeStr + "[ext=" + extIdx + ",name=" + extKey + "]";
                    }
                    sb.append(" in[").append(i).append("]=")
                            .append(inputVarNames[i]).append("→").append(srcDesc);
                }
                log.info(sb.toString());
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
                // Guard against ops (e.g. BatchMmulBp) that return null from arg accessors
                // when their argument lists are null instead of empty.
                long[] rawIArgs = dynOp.iArgs();
                double[] rawTArgs = dynOp.tArgs();
                boolean[] rawBArgs = dynOp.bArgs();
                DataType[] rawDArgs = dynOp.dArgs();
                String[] rawSArgs = dynOp.sArgs();
                iArgs = rawIArgs != null ? rawIArgs : new long[0];
                tArgs = rawTArgs != null ? rawTArgs : new double[0];
                bArgs = rawBArgs != null ? rawBArgs : new boolean[0];
                dArgs = rawDArgs != null ? rawDArgs : new DataType[0];
                sArgs = rawSArgs != null ? rawSArgs : new String[0];

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

            // Legacy transform ops (BaseTransformSameOp, BaseTransformStrictOp,
            // BaseTransformBoolOp, BaseTransformFloatOp, BaseBroadcastOp, BaseBroadcastBoolOp)
            // may store runtime parameters in extraArgs rather than tArgs.
            // For example, CompareAndSet stores [compare, set, eps, mode] in extraArgs.
            // The C++ LegacyTransformSameOp::validateAndExecute builds ExtraArguments from
            // block.getTArguments() — if tArgs is empty, argumentsAsT() returns nullptr,
            // causing SIGSEGV in op_logic(double, double*).
            // Mirror extraArgs → tArgs so the C++ ExtraArguments is populated.
            if (!isCustomOp && tArgs.length == 0 && op instanceof Op) {
                Object[] extras = ((Op) op).extraArgs();
                if (extras != null && extras.length > 0) {
                    double[] packedT = new double[extras.length];
                    for (int i = 0; i < extras.length; i++) {
                        packedT[i] = (extras[i] instanceof Number)
                            ? ((Number) extras[i]).doubleValue()
                            : 0.0;
                    }
                    tArgs = packedT;
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

                // MatchCondition (BaseReduceLongOp) and other reduce ops that carry
                // numeric extraArgs (e.g., compare/eps/mode for condition reductions) need
                // those mirrored into tArgs so LegacyReduceLongOp::validateAndExecute can
                // pass them to the kernel via ExtraArguments.argumentsAsT(x->dataType()).
                Object[] extras = reduceOp.extraArgs();
                if (extras != null && extras.length > 0 && tArgs.length == 0) {
                    double[] packedT = new double[extras.length];
                    for (int i = 0; i < extras.length; i++) {
                        packedT[i] = (extras[i] instanceof Number)
                            ? ((Number) extras[i]).doubleValue()
                            : 0.0;
                    }
                    tArgs = packedT;
                }

                // LegacyStatsOp (Variance/StandardDeviation) expects iArgs[0] = biasCorrected,
                // iArgs[1..] = dimensions. This differs from other reduce ops which put dims at iArgs[0..].
                if (op instanceof Variance) {
                    Variance varOp = (Variance) op;
                    long biasCorrected = varOp.isBiasCorrected() ? 1L : 0L;
                    long[] packed = new long[iArgs.length + 1];
                    packed[0] = biasCorrected;
                    System.arraycopy(iArgs, 0, packed, 1, iArgs.length);
                    iArgs = packed;
                }
            }

            // BaseIndexAccumulation (FirstIndex, LastIndex, IAMax, IAMin, IMax, IMin) extends
            // BaseOp, not BaseReduceOp, so the branch above misses it. Pack dimensions into iArgs
            // (DSP slot-exec mirrors iArgs into block.getAxis() for LEGACY_INDEX_REDUCE), and
            // extraArgs (compare, eps, mode for FirstIndex/LastIndex) into tArgs.
            if (op instanceof BaseIndexAccumulation) {
                BaseIndexAccumulation idxOp = (BaseIndexAccumulation) op;
                long[] dims = idxOp.dimensionsArr();
                if (dims == null || dims.length == 0) {
                    INDArray dimensionz = idxOp.dimensions();
                    if (dimensionz != null && !dimensionz.isEmpty()) {
                        dims = dimensionz.toLongVector();
                    }
                }
                if (dims != null && dims.length > 0) {
                    iArgs = dims;
                }
                Object[] extras = idxOp.extraArgs();
                if (extras != null && extras.length > 0) {
                    double[] packedT = new double[extras.length];
                    for (int i = 0; i < extras.length; i++) {
                        packedT[i] = (extras[i] instanceof Number)
                            ? ((Number) extras[i]).doubleValue()
                            : 0.0;
                    }
                    tArgs = packedT;
                }
                bArgs = new boolean[]{idxOp.isKeepDims()};
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
            if (op instanceof BaseTensorOp) {
                requiresDynamic = true;
            }

            // Consult the C++ OpTraitTable (single source of truth) rather than the
            // Value-dependent shape classification: only use VALUE_DEPENDENT_SHAPE trait.
            // DATA_DEPENDENT is orthogonal — it means the op's result depends on data,
            // not that the output SHAPE depends on data. argmax/argmin are data-dependent
            // but have shapes fully determined by input shapes + axis iArgs.
            // Ops whose shape function reads input tensor VALUES carry
            // VALUE_DEPENDENT_SHAPE in the trait table (reshape, fill, range, slice, etc.).
            int opTraits = opTraitsOf(opName);
            boolean shapeDependsOnValues;
            if (opTraits != 0) {
                shapeDependsOnValues = (opTraits & OP_TRAIT_VALUE_DEPENDENT_SHAPE) != 0;
            } else {
                // Op not in trait table (e.g. legacy / unregistered) — fall back to the
                // old heuristic so we stay safe on unclassified ops.
                shapeDependsOnValues = hasIntLongInputs;
            }

            // Per-instance demotion for ops with VALUE_DEPENDENT_SHAPE trait that
            // don't actually read input values in specific configurations.
            if (shapeDependsOnValues) {
                if ("reshape".equals(opName) || "reshape_no_copy".equals(opName)) {
                    // VALUE_DEPENDENT_SHAPE only when width > 1: shape read from INPUT(1).
                    // Single-input reshape reads target shape from iArgs.
                    if (numInputs <= 1) {
                        shapeDependsOnValues = false;
                    }
                }
            }
            // Per-instance promotion for ops with DATA_DEPENDENT that genuinely have
            // value-dependent shape in certain configurations:
            if (!shapeDependsOnValues) {
                if ("concat".equals(opName)) {
                    // Axis read from last input array when isAxisInLastArr (bArgs[0]==true).
                    boolean isAxisInLastArr = bArgs.length > 0 && bArgs[0];
                    if (isAxisInLastArr) {
                        shapeDependsOnValues = true;
                    }
                } else if ("expand_dims".equals(opName)) {
                    // Axis from INPUT(1)->e<>() when no INT_ARG present.
                    if (iArgs.length == 0) {
                        shapeDependsOnValues = true;
                    }
                } else if (("argmax".equals(opName) || "argmin".equals(opName)) && numInputs > 1) {
                    // 2-input form reads axes from INPUT_VARIABLE(1).
                    shapeDependsOnValues = true;
                }
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

        // ═══════════════════════════════════════════════════════════════════
        // Post-build structural validation — catch wiring bugs at compile time.
        // ═══════════════════════════════════════════════════════════════════
        {
            int selfRefErrors = 0;
            int forwardRefWarnings = 0;
            for (int stepIdx = 0; stepIdx < slots.length; stepIdx++) {
                DynamicShapeSlot slot = slots[stepIdx];
                if (slot == null) continue;
                int[] srcIndices = slot.getInputSourceIndices();
                int[] outSlots = slot.getOutputSlotIndices();
                if (srcIndices == null || outSlots == null) continue;

                for (int i = 0; i < srcIndices.length; i++) {
                    int srcIdx = srcIndices[i];
                    if (srcIdx < 0) continue;  // external input

                    // Self-reference: input reads from this step's own output
                    for (int outSlot : outSlots) {
                        if (outSlot == srcIdx) {
                            selfRefErrors++;
                            log.error("DSP VALIDATION: SELF-REFERENCE at step {} ({}): "
                                    + "input[{}] srcIdx={} == own outputSlot. "
                                    + "inputVar='{}' outputVars={}",
                                    stepIdx, slot.getOpName(), i, srcIdx,
                                    slot.getInputVarNames() != null && i < slot.getInputVarNames().length
                                            ? slot.getInputVarNames()[i] : "?",
                                    slot.getOutputVarNames() != null
                                            ? Arrays.toString(slot.getOutputVarNames()) : "?");
                        }
                    }

                    // Forward reference: input from a later step (non-CF only)
                    int producerStep = srcIdx < slotProducerStep.length
                            ? slotProducerStep[srcIdx] : -1;
                    if (producerStep > stepIdx
                            && slot.getControlFlowType() == DynamicShapeSlot.CF_NONE) {
                        forwardRefWarnings++;
                        log.warn("DSP VALIDATION: FORWARD-REF at step {} ({}): "
                                + "input[{}] srcIdx={} produced by later step {} ({})",
                                stepIdx, slot.getOpName(), i, srcIdx,
                                producerStep,
                                slots[producerStep] != null
                                        ? slots[producerStep].getOpName() : "?");
                    }
                }
            }

            if (selfRefErrors > 0) {
                log.error("DSP VALIDATION FAILED: {} self-referencing input(s) — "
                        + "falling back to standard execution path", selfRefErrors);
                return null;
            }
            if (forwardRefWarnings > 0) {
                log.warn("DSP VALIDATION: {} forward reference(s) in non-CF ops — "
                        + "may cause NULL inputs at runtime", forwardRefWarnings);
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
        //
        // If any requested output is an external input (placeholder or variable), it has
        // no slot in the plan — the DSP executor cannot return external inputs as outputs.
        // Fall back to standard execution in this case to get correct values.
        // This occurs when GradCheckUtil requests gradient variable names that coincide with
        // the forward graph's placeholder/variable names in the grad function's SameDiff.
        Set<String> externalInputSet = new java.util.HashSet<>(externalInputKeys);
        for (String outputName : requestedOutputs) {
            if (!varToOutputSlot.containsKey(outputName) && externalInputSet.contains(outputName)) {
                log.debug("DSP compile: requested output '{}' is an external input (not an op output). " +
                        "DSP cannot return external inputs as outputs — falling back to standard path.", outputName);
                return null;
            }
        }

        Map<String, Integer> outputNameToSlotIndex = new java.util.LinkedHashMap<>();
        int missingOutputCount = 0;
        for (String outputName : requestedOutputs) {
            Integer slot = varToOutputSlot.get(outputName);
            if (slot != null) {
                outputNameToSlotIndex.put(outputName, slot);
                // Log first 5 output mappings and any k_rope/v_heads for diagnosis
                if (outputNameToSlotIndex.size() <= 5 || outputName.startsWith("k_rope") || outputName.startsWith("v_heads")) {
                    // Find which op produces this variable and trace its inputs
                    String producerInfo = "unknown";
                    String inputInfo = "";
                    for (int pi = 0; pi < opNodes.size(); pi++) {
                        for (String ov : opNodes.get(pi).getOutputVariables()) {
                            if (ov.equals(outputName)) {
                                producerInfo = opNodes.get(pi).getOperationName() + " (step " + pi + ")";
                                // Also log input variables and their slots
                                StringBuilder sb = new StringBuilder();
                                for (String iv : opNodes.get(pi).getInputVariables()) {
                                    Integer ivSlot = varToOutputSlot.get(iv);
                                    sb.append(iv).append("->slot").append(ivSlot != null ? ivSlot : "EXT").append(" ");
                                }
                                inputInfo = sb.toString();
                                break;
                            }
                        }
                    }
                    log.info("DSP compile: output '{}' -> slot {} (producer: {} inputs: [{}])", outputName, slot, producerInfo, inputInfo);
                }
            } else {
                missingOutputCount++;
                log.warn("DSP compile: requested output '{}' NOT FOUND in varToOutputSlot (will be slot -1)", outputName);
            }
        }
        if (missingOutputCount > 0) {
            log.warn("DSP compile: {} of {} requested outputs have no slot (will return zeros)", missingOutputCount, requestedOutputs.size());
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
        // Log all external input names (always at info level for diagnostics)
        for (int i = 0; i < externalInputKeys.size(); i++) {
            byte srcType = externalInputSourceTypes.get(i);
            String typeStr = srcType == DynamicShapeSlot.SOURCE_CONSTANT ? "CONSTANT" :
                    srcType == DynamicShapeSlot.SOURCE_VARIABLE ? "VARIABLE" :
                    srcType == DynamicShapeSlot.SOURCE_PLACEHOLDER ? "PLACEHOLDER" : "UNKNOWN";
            log.info("EXT_INPUT_DISCOVER: {} idx={} name=\"{}\"", typeStr, i, externalInputKeys.get(i));
        }
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
            // formatPlan renders one line per slot (MBs for LLM-scale plans) — slf4j
            // placeholders only defer FORMATTING, not argument evaluation, so guard it.
            if (log.isDebugEnabled()) {
                log.debug("Plan structure:\n{}", PlanIntrospection.formatPlan(plan));
            }
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

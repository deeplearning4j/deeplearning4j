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

package org.nd4j.autodiff.samediff.optimize.optimizations;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.ArrayHolder;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.autodiff.samediff.optimize.OptimizationHelper;
import org.nd4j.autodiff.samediff.optimize.Optimizer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce.Mmul;
import org.nd4j.linalg.api.ops.impl.reduce.TensorMmul;
import org.nd4j.linalg.api.ops.impl.shape.Concat;
import org.nd4j.linalg.api.ops.impl.shape.Transpose;
import org.nd4j.linalg.api.ops.impl.shape.Reshape;
import org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax;
import org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttentionV2;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MulOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.DivOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.AddOp;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarMultiplication;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarDivision;
import org.nd4j.linalg.api.ops.impl.indexaccum.custom.ArgMax;
import org.nd4j.linalg.api.ops.impl.shape.Shape;
import org.nd4j.linalg.api.ops.impl.shape.Permute;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Attention fusion optimizations for transformers/BERT models.
 * Detects manual attention patterns and replaces them with optimized attention ops.
 *
 * Common patterns detected:
 * 1. matmul(Q, K^T) -> scale -> softmax -> matmul(_, V) => dot_product_attention_v2
 * 2. Fuses attention output with subsequent linear projection
 */
@Slf4j
public class AttentionFusionOptimizations extends BaseOptimizerSet {

    /**
     * Detects the pattern: matmul(softmax_output, V) where softmax_output comes from
     * softmax(scale(matmul(Q, K^T)))
     *
     * This is the final matmul in the attention computation. We trace backwards to find
     * the full attention pattern and replace it with dot_product_attention_v2.
     */
    public static class FuseManualAttentionPattern implements Optimizer {

        private static final Set<Class<? extends DifferentialFunction>> APPLICABLE_OPS = new HashSet<>();
        static {
            APPLICABLE_OPS.add(Mmul.class);
            APPLICABLE_OPS.add(TensorMmul.class);
        }

        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return APPLICABLE_OPS;
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            // Look for the final matmul in attention: softmax_weights @ V
            // Note: type check is now done by GraphOptimizer via getApplicableOpTypes()
            if (!(op.getOp() instanceof Mmul) && !(op.getOp() instanceof TensorMmul)) {
                return false;
            }

            log.debug("Checking matmul op: {}", op.getName());

            List<String> mmulInputs = op.getInputsToOp();
            if (mmulInputs == null || mmulInputs.size() < 2) {
                log.debug("[ATTN] " + op.getName() + " has insufficient inputs");
                return false;
            }

            // Check if first input comes from softmax (attention weights)
            // The softmax may be directly connected or through a Reshape op
            String potentialSoftmaxOutput = mmulInputs.get(0);
            Variable firstInputVar = helper.getVariable(potentialSoftmaxOutput);
            if (firstInputVar == null) {
                // Try fallback to graph lookup
                firstInputVar = sd.getVariables().get(potentialSoftmaxOutput);
            }
            if (firstInputVar == null) {
                log.debug("[ATTN] " + op.getName() + " first input " + potentialSoftmaxOutput + " not found");
                return false;
            }

            String producerOpName = firstInputVar.getOutputOfOp();
            if (producerOpName == null) {
                log.debug("[ATTN] " + op.getName() + " first input " + potentialSoftmaxOutput + " has no producer op");
                return false;
            }

            SameDiffOp producerOp = sd.getOps().get(producerOpName);
            if (producerOp == null) {
                log.debug("[ATTN] " + op.getName() + " producer op " + producerOpName + " not found");
                return false;
            }

            log.debug("[ATTN] " + op.getName() + " first input " + potentialSoftmaxOutput + " comes from " + producerOpName + " (type: " + producerOp.getOp().getClass().getSimpleName() + ")");

            SameDiffOp softmaxOp = null;
            SameDiffOp reshapeAfterSoftmax = null;

            // Check if producer is directly softmax
            if (producerOp.getOp() instanceof SoftMax) {
                softmaxOp = producerOp;
                log.debug("[ATTN] " + op.getName() + " - found direct softmax producer");
            }
            // Check if producer is Reshape and Reshape's input is softmax
            else if (producerOp.getOp() instanceof Reshape) {
                List<String> reshapeInputs = producerOp.getInputsToOp();
                if (reshapeInputs != null && !reshapeInputs.isEmpty()) {
                    Variable reshapeInputVar = getVariableWithFallback(helper, sd, reshapeInputs.get(0));
                    if (reshapeInputVar != null) {
                        String reshapeInputProducerName = reshapeInputVar.getOutputOfOp();
                        if (reshapeInputProducerName != null) {
                            SameDiffOp reshapeInputProducer = sd.getOps().get(reshapeInputProducerName);
                            if (reshapeInputProducer != null && reshapeInputProducer.getOp() instanceof SoftMax) {
                                softmaxOp = reshapeInputProducer;
                                reshapeAfterSoftmax = producerOp;
                                log.debug("[ATTN] Found reshape " + producerOp.getName() + " between softmax and matmul");
                            }
                        }
                    }
                }
            }
            // Check if producer is Permute/Transpose and its input is softmax (or reshape->softmax)
            else if (producerOp.getOp() instanceof Permute || producerOp.getOp() instanceof Transpose) {
                List<String> permuteInputs = producerOp.getInputsToOp();
                if (permuteInputs != null && !permuteInputs.isEmpty()) {
                    Variable permuteInputVar = getVariableWithFallback(helper, sd, permuteInputs.get(0));
                    if (permuteInputVar != null) {
                        String permuteInputProducerName = permuteInputVar.getOutputOfOp();
                        if (permuteInputProducerName != null) {
                            SameDiffOp permuteInputProducer = sd.getOps().get(permuteInputProducerName);
                            if (permuteInputProducer != null && permuteInputProducer.getOp() instanceof SoftMax) {
                                softmaxOp = permuteInputProducer;
                                log.debug("[ATTN] Found permute " + producerOp.getName() + " between softmax and matmul");
                            }
                        }
                    }
                }
            }

            if (softmaxOp == null) {
                log.debug("[ATTN] First input " + potentialSoftmaxOutput + " is NOT from softmax (producer: " + producerOp.getOp().getClass().getSimpleName() + ")");
                return false;
            }

            log.debug("[ATTN] Found softmax " + softmaxOp.getName() + " before matmul " + op.getName());
            // Found softmax! Now trace back to find Q @ K^T pattern
            List<String> softmaxInputs = softmaxOp.getInputsToOp();
            if (softmaxInputs == null || softmaxInputs.isEmpty()) {
                log.debug("[ATTN] Softmax has no inputs");
                return false;
            }

            // The input to softmax could be:
            // 1. Direct matmul output (Q @ K^T)
            // 2. Scaled matmul output (Q @ K^T * scale or Q @ K^T / sqrt(d))
            // 3. Masked and/or scaled output

            String softmaxInput = softmaxInputs.get(0);
            log.debug("[ATTN] Tracing attention scores from softmax input: " + softmaxInput);
            AttentionComponents components = traceAttentionScores(sd, helper, softmaxInput);
            if (components == null) {
                log.debug("[ATTN] Could not trace attention scores from softmax input: " + softmaxInput);
                return false;
            }
            log.debug("[ATTN] Found Q=" + components.queryVar + ", K=" + components.keyVar + ", scale=" + components.scaleFactor);

            // Get V from the final matmul
            String vVar = mmulInputs.get(1);
            SDVariable vSDVar = sd.getVariable(vVar);
            if (vSDVar == null) {
                log.debug("[ATTN] V variable " + vVar + " not found");
                return false;
            }
            log.debug("[ATTN] V=" + vVar);

            // Check that intermediate ops are only used once (safe to fuse)
            if (!canSafelyFuse(sd, helper, softmaxOp, components)) {
                log.debug("[ATTN] canSafelyFuse check FAILED for op: " + op.getName());
                return false;
            }
            log.debug("[ATTN] canSafelyFuse check PASSED for op: " + op.getName());

            // Get the output of the attention (final matmul output)
            List<String> attentionOutputs = op.getOutputsOfOp();
            if (attentionOutputs == null || attentionOutputs.isEmpty()) {
                log.debug("[ATTN] No attention outputs");
                return false;
            }
            String attentionOutputVar = attentionOutputs.get(0);

            log.debug("[ATTN] *** FUSING *** Q=" + components.queryVar + ", K=" + components.keyVar + ", V=" + vVar + " (causalMask=" + components.useCausalMask + ")");

            try {
                SDVariable qSDVar = sd.getVariable(components.queryVar);
                SDVariable kSDVar = sd.getVariable(components.keyVar);

                if (qSDVar == null || kSDVar == null) {
                    return false;
                }

                // dot_product_attention_v2 only supports rank 2 or 3 inputs.
                // Multi-head attention patterns with rank 4 (batch, heads, seq, dim) cannot be fused.
                // Check static shapes first, then try to infer rank from producer ops.
                int qRank = inferVariableRank(sd, qSDVar);
                int kRank = inferVariableRank(sd, kSDVar);
                int vRank = inferVariableRank(sd, vSDVar);
                log.debug("Attention fusion rank check: Q={} rank={}, K={} rank={}, V={} rank={}",
                        components.queryVar, qRank, components.keyVar, kRank, vVar, vRank);
                // dot_product_attention_v2 only supports rank 2 or 3.
                // Reject unknown ranks (-1) and rank 4+ (multi-head with explicit head dim).
                if (qRank < 2 || kRank < 2 || vRank < 2 || qRank > 3 || kRank > 3 || vRank > 3) {
                    log.debug("Skipping attention fusion: ranks not supported (Q={}, K={}, V={}). " +
                            "dot_product_attention_v2 requires rank 2 or 3.", qRank, kRank, vRank);
                    return false;
                }

                // Create dot_product_attention_v2 op
                // Note: We use keys=K (not K^T) because the op handles transposition internally
                SDVariable emptyQueryMask = sd.constant("attn_empty_qmask_" + op.getName(), Nd4j.empty(DataType.FLOAT));

                // Use the detected mask if available, otherwise use empty mask
                SDVariable valueMask;
                if (components.hasAdditiveMask && components.maskVar != null) {
                    valueMask = sd.getVariable(components.maskVar);
                    if (valueMask == null) {
                        valueMask = sd.constant("attn_empty_vmask_" + op.getName(), Nd4j.empty(DataType.FLOAT));
                    }
                    log.debug("[ATTN-DEBUG] Using detected mask variable: " + components.maskVar);
                } else {
                    valueMask = sd.constant("attn_empty_vmask_" + op.getName(), Nd4j.empty(DataType.FLOAT));
                }

                SDVariable fusedOutput = new DotProductAttentionV2(sd,
                        qSDVar,           // queries
                        vSDVar,           // values
                        kSDVar,           // keys
                        emptyQueryMask,   // queryMask (empty)
                        valueMask,        // valueMask - use detected mask if available
                        components.scaleFactor,  // scale factor
                        0.0,              // no dropout for inference
                        components.useCausalMask,  // use causal mask if detected
                        false             // not training
                ).outputVariable();

                // Replace all uses of the attention output with the fused output
                OptimizationUtils.replaceOpInputsWith(sd, helper, attentionOutputVar, fusedOutput.name());

                // Remove old operations in reverse order
                OptimizationUtils.removeOp(sd, helper, op.getName());  // final matmul

                // Remove reshape between softmax and final matmul if it exists
                if (reshapeAfterSoftmax != null) {
                    OptimizationUtils.removeOp(sd, helper, reshapeAfterSoftmax.getName());
                }

                OptimizationUtils.removeOp(sd, helper, softmaxOp.getName());

                // Remove mask add op if it exists
                if (components.maskOpName != null) {
                    OptimizationUtils.removeOp(sd, helper, components.maskOpName);
                }

                // Remove scale op if it exists
                if (components.scaleOpName != null) {
                    OptimizationUtils.removeOp(sd, helper, components.scaleOpName);
                }

                // Remove reshape between matmul and scale if it exists
                if (components.reshapeBeforeScaleOpName != null) {
                    OptimizationUtils.removeOp(sd, helper, components.reshapeBeforeScaleOpName);
                }

                // Remove Q @ K^T matmul
                OptimizationUtils.removeOp(sd, helper, components.qkMatmulOpName);

                // Remove intermediate variables
                OptimizationUtils.removeVariable(sd, helper, potentialSoftmaxOutput);

                // Remove reshape output variable between matmul and scale
                if (components.reshapeBeforeScaleOutputVar != null) {
                    OptimizationUtils.removeVariable(sd, helper, components.reshapeBeforeScaleOutputVar);
                }

                // Remove softmax output variable (input to reshape) if there was a reshape
                if (reshapeAfterSoftmax != null) {
                    List<String> softmaxOutputs = softmaxOp.getOutputsOfOp();
                    if (softmaxOutputs != null && !softmaxOutputs.isEmpty()) {
                        OptimizationUtils.removeVariable(sd, helper, softmaxOutputs.get(0));
                    }
                }

                OptimizationUtils.removeVariable(sd, helper, softmaxInput);
                // Only remove mask output var, not the mask itself (which is still used by fused op)
                if (components.maskOutputVar != null) {
                    OptimizationUtils.removeVariable(sd, helper, components.maskOutputVar);
                }
                if (components.scaleOutputVar != null) {
                    OptimizationUtils.removeVariable(sd, helper, components.scaleOutputVar);
                }
                OptimizationUtils.removeVariable(sd, helper, components.qkMatmulOutputVar);
                OptimizationUtils.removeVariable(sd, helper, attentionOutputVar);
                // Note: We do NOT remove components.maskVar as it's used by the fused attention op

                return true;
            } catch (Exception e) {
                log.debug("[ATTN-WARN] Failed to fuse attention pattern: " + e.getMessage());
                return false;
            }
        }

        /**
         * Traces back from softmax input to find Q @ K^T pattern and optional scaling/masking.
         * Supports patterns:
         * 1. matmul(Q, K^T) -> softmax
         * 2. matmul(Q, K^T) -> scale -> softmax
         * 3. matmul(Q, K^T) -> add(mask) -> softmax
         * 4. matmul(Q, K^T) -> scale -> add(mask) -> softmax
         */
        private AttentionComponents traceAttentionScores(SameDiff sd, OptimizationHelper helper, String varName) {
            Variable v = helper.getVariable(varName);
            if (v == null) {
                // Fallback to graph lookup
                v = sd.getVariables().get(varName);
            }
            if (v == null) {
                log.debug("[ATTN-TRACE] Variable " + varName + " is null");
                return null;
            }

            String producerOpName = v.getOutputOfOp();
            if (producerOpName == null) {
                log.debug("[ATTN-TRACE] Variable " + varName + " has no producer op");
                return null;
            }

            SameDiffOp producerOp = sd.getOps().get(producerOpName);
            if (producerOp == null) {
                log.debug("[ATTN-TRACE] Producer op " + producerOpName + " not found");
                return null;
            }

            log.debug("[ATTN-TRACE] " + varName + " produced by " + producerOpName + " (type: " + producerOp.getOp().getClass().getSimpleName() + ")");

            // Case 0: Reshape - trace through it
            if (producerOp.getOp() instanceof Reshape) {
                List<String> reshapeInputs = producerOp.getInputsToOp();
                if (reshapeInputs != null && !reshapeInputs.isEmpty()) {
                    log.debug("[ATTN-TRACE] Tracing through reshape " + producerOpName + " to " + reshapeInputs.get(0));
                    return traceAttentionScores(sd, helper, reshapeInputs.get(0));
                }
                return null;
            }

            // Case 1: Direct matmul output (no scaling, no mask)
            if (producerOp.getOp() instanceof Mmul || producerOp.getOp() instanceof TensorMmul) {
                log.debug("[ATTN-TRACE] Found Q@K matmul: " + producerOpName);
                return extractQKFromMatmul(sd, helper, producerOp, varName, null, null, 1.0);
            }

            // Case 2: Scaled output - look for mul/div by scalar
            if (producerOp.getOp() instanceof MulOp || producerOp.getOp() instanceof ScalarMultiplication) {
                log.debug("[ATTN-TRACE] Found scale (mul): " + producerOpName);
                return traceScaledMatmul(sd, helper, producerOp, varName, true);
            }

            if (producerOp.getOp() instanceof DivOp || producerOp.getOp() instanceof ScalarDivision) {
                log.debug("[ATTN-TRACE] Found scale (div): " + producerOpName);
                return traceScaledMatmul(sd, helper, producerOp, varName, false);
            }

            // Case 3: Add operation - could be mask addition before softmax
            if (producerOp.getOp() instanceof AddOp) {
                log.debug("[ATTN-TRACE] Found add (possible mask): " + producerOpName);
                return traceAttentionWithMask(sd, helper, producerOp, varName);
            }

            log.debug("[ATTN-TRACE] Unhandled op type: " + producerOp.getOp().getClass().getSimpleName());
            return null;
        }

        /**
         * Traces through an add operation that applies a mask to attention scores.
         * Pattern: matmul(Q, K^T) [-> scale] -> add(mask) -> softmax
         */
        private AttentionComponents traceAttentionWithMask(SameDiff sd, OptimizationHelper helper, SameDiffOp addOp, String addOutputVar) {
            log.debug("[ATTN-MASK] === Entering traceAttentionWithMask for " + addOp.getName() + " ===");

            List<String> addInputs = addOp.getInputsToOp();
            if (addInputs == null || addInputs.size() != 2) {
                log.debug("[ATTN-MASK] Add op has wrong number of inputs: " + (addInputs != null ? addInputs.size() : 0));
                return null;
            }

            log.debug("[ATTN-MASK] Add inputs: " + addInputs.get(0) + ", " + addInputs.get(1));

            // Find which input comes from matmul/scale (attention scores) and which is the mask
            String scoresVar = null;
            String maskVar = null;
            SameDiffOp scoresProducerOp = null;
            boolean isCausalMask = false;

            for (int i = 0; i < 2; i++) {
                String input = addInputs.get(i);
                log.debug("[ATTN-MASK] Checking input " + i + ": " + input);

                Variable inputVar = getVariableWithFallback(helper, sd, input);
                if (inputVar == null) {
                    log.debug("[ATTN-MASK] Input " + i + " (" + input + ") variable is null");
                    continue;
                }

                String inputOpName = inputVar.getOutputOfOp();
                log.debug("[ATTN-MASK] Input " + i + " (" + input + ") produced by op: " + inputOpName);

                if (inputOpName == null) {
                    log.debug("[ATTN-MASK] Input " + i + " has no producer op (likely placeholder or constant)");
                    // This might be the mask - check if the other input has a producer
                    SDVariable sdVar = sd.getVariable(input);
                    if (sdVar != null) {
                        log.debug("[ATTN-MASK] Input " + i + " is SDVariable type: " + sdVar.getVariableType());
                    }
                    continue;
                }

                SameDiffOp inputOp = sd.getOps().get(inputOpName);
                if (inputOp == null) {
                    log.debug("[ATTN-MASK] Input " + i + " producer op " + inputOpName + " not found in graph");
                    continue;
                }

                String opType = inputOp.getOp().getClass().getSimpleName();
                log.debug("[ATTN-MASK] Input " + i + " producer op type: " + opType);

                // Check if this is matmul, scale op, or another pattern
                if (inputOp.getOp() instanceof Mmul || inputOp.getOp() instanceof TensorMmul ||
                    inputOp.getOp() instanceof MulOp || inputOp.getOp() instanceof ScalarMultiplication ||
                    inputOp.getOp() instanceof DivOp || inputOp.getOp() instanceof ScalarDivision) {
                    scoresVar = input;
                    scoresProducerOp = inputOp;
                    maskVar = addInputs.get(1 - i);
                    log.debug("[ATTN-MASK] Found scores from " + opType + ": " + scoresVar);
                    break;
                }
                // Check for Permute/Transpose - trace through it
                else if (inputOp.getOp() instanceof Permute || inputOp.getOp() instanceof Transpose) {
                    log.debug("[ATTN-MASK] Input " + i + " is Permute/Transpose, tracing through...");
                    List<String> permuteInputs = inputOp.getInputsToOp();
                    if (permuteInputs != null && !permuteInputs.isEmpty()) {
                        String permuteInput = permuteInputs.get(0);
                        log.debug("[ATTN-MASK] Permute input: " + permuteInput);
                        Variable permuteInputVar = getVariableWithFallback(helper, sd, permuteInput);
                        if (permuteInputVar != null) {
                            String permuteProducerName = permuteInputVar.getOutputOfOp();
                            log.debug("[ATTN-MASK] Permute input produced by: " + permuteProducerName);
                            if (permuteProducerName != null) {
                                SameDiffOp permuteProducerOp = sd.getOps().get(permuteProducerName);
                                if (permuteProducerOp != null) {
                                    String permuteProducerType = permuteProducerOp.getOp().getClass().getSimpleName();
                                    log.debug("[ATTN-MASK] Permute producer op type: " + permuteProducerType);
                                    if (permuteProducerOp.getOp() instanceof Mmul || permuteProducerOp.getOp() instanceof TensorMmul ||
                                        permuteProducerOp.getOp() instanceof MulOp || permuteProducerOp.getOp() instanceof ScalarMultiplication ||
                                        permuteProducerOp.getOp() instanceof DivOp || permuteProducerOp.getOp() instanceof ScalarDivision) {
                                        scoresVar = permuteInput;
                                        scoresProducerOp = permuteProducerOp;
                                        maskVar = addInputs.get(1 - i);
                                        log.debug("[ATTN-MASK] Found scores through permute: " + scoresVar);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                // Also check for Reshape - trace through it
                else if (inputOp.getOp() instanceof Reshape) {
                    log.debug("[ATTN-MASK] Input " + i + " is Reshape, tracing through...");
                    List<String> reshapeInputs = inputOp.getInputsToOp();
                    if (reshapeInputs != null && !reshapeInputs.isEmpty()) {
                        String reshapeInput = reshapeInputs.get(0);
                        log.debug("[ATTN-MASK] Reshape input: " + reshapeInput);
                        Variable reshapeInputVar = getVariableWithFallback(helper, sd, reshapeInput);
                        if (reshapeInputVar != null) {
                            String reshapeProducerName = reshapeInputVar.getOutputOfOp();
                            log.debug("[ATTN-MASK] Reshape input produced by: " + reshapeProducerName);
                            if (reshapeProducerName != null) {
                                SameDiffOp reshapeProducerOp = sd.getOps().get(reshapeProducerName);
                                if (reshapeProducerOp != null) {
                                    String reshapeProducerType = reshapeProducerOp.getOp().getClass().getSimpleName();
                                    log.debug("[ATTN-MASK] Reshape producer op type: " + reshapeProducerType);
                                    if (reshapeProducerOp.getOp() instanceof Mmul || reshapeProducerOp.getOp() instanceof TensorMmul ||
                                        reshapeProducerOp.getOp() instanceof MulOp || reshapeProducerOp.getOp() instanceof ScalarMultiplication ||
                                        reshapeProducerOp.getOp() instanceof DivOp || reshapeProducerOp.getOp() instanceof ScalarDivision) {
                                        // The reshape input is the scores - use the original input name for tracing
                                        scoresVar = reshapeInput;
                                        scoresProducerOp = reshapeProducerOp;
                                        maskVar = addInputs.get(1 - i);
                                        log.debug("[ATTN-MASK] Found scores through reshape: " + scoresVar);
                                        break;
                                    }
                                    // Also check if reshape producer is Permute/Transpose
                                    else if (reshapeProducerOp.getOp() instanceof Permute || reshapeProducerOp.getOp() instanceof Transpose) {
                                        log.debug("[ATTN-MASK] Reshape producer is Permute, tracing further...");
                                        List<String> permuteInputs = reshapeProducerOp.getInputsToOp();
                                        if (permuteInputs != null && !permuteInputs.isEmpty()) {
                                            Variable permuteInputVar = getVariableWithFallback(helper, sd, permuteInputs.get(0));
                                            if (permuteInputVar != null) {
                                                String deepProducerName = permuteInputVar.getOutputOfOp();
                                                if (deepProducerName != null) {
                                                    SameDiffOp deepProducerOp = sd.getOps().get(deepProducerName);
                                                    if (deepProducerOp != null) {
                                                        String deepProducerType = deepProducerOp.getOp().getClass().getSimpleName();
                                                        log.debug("[ATTN-MASK] Deep producer type: " + deepProducerType);
                                                        if (deepProducerOp.getOp() instanceof Mmul || deepProducerOp.getOp() instanceof TensorMmul ||
                                                            deepProducerOp.getOp() instanceof MulOp || deepProducerOp.getOp() instanceof ScalarMultiplication ||
                                                            deepProducerOp.getOp() instanceof DivOp || deepProducerOp.getOp() instanceof ScalarDivision) {
                                                            scoresVar = permuteInputs.get(0);
                                                            scoresProducerOp = deepProducerOp;
                                                            maskVar = addInputs.get(1 - i);
                                                            log.debug("[ATTN-MASK] Found scores through reshape+permute: " + scoresVar);
                                                            break;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Check if this could be the mask (constant array or placeholder)
                SDVariable sdVar = sd.getVariable(input);
                if (sdVar != null && sdVar.getArr() != null) {
                    // This looks like a constant mask
                    if (isCausalMaskArray(sdVar.getArr())) {
                        maskVar = input;
                        scoresVar = addInputs.get(1 - i);
                        isCausalMask = true;
                        log.debug("[ATTN-MASK] Found causal mask: " + maskVar);
                    }
                }
            }

            if (scoresVar == null) {
                log.debug("[ATTN-MASK] Could not find scores variable! Tried both inputs.");
                log.debug("[ATTN-MASK] === Exiting traceAttentionWithMask with NULL ===");
                return null;
            }

            log.debug("[ATTN-MASK] Scores var: " + scoresVar + ", Mask var: " + maskVar);

            // Now trace back from scoresVar to find the actual Q @ K^T pattern
            log.debug("[ATTN-MASK] Calling traceAttentionScoresWithoutMask for: " + scoresVar);
            AttentionComponents components = traceAttentionScoresWithoutMask(sd, helper, scoresVar);
            if (components == null) {
                log.debug("[ATTN-MASK] traceAttentionScoresWithoutMask returned null for " + scoresVar);
                log.debug("[ATTN-MASK] === Exiting traceAttentionWithMask with NULL ===");
                return null;
            }

            // Add mask information to components
            components.maskOpName = addOp.getName();
            components.maskOutputVar = addOutputVar;
            components.maskVar = maskVar;
            components.useCausalMask = isCausalMask;
            components.hasAdditiveMask = true;

            log.debug("[ATTN-MASK] === Exiting traceAttentionWithMask with components ===");
            return components;
        }

        /**
         * Traces attention scores without considering mask add operations.
         * Used after we've already identified and removed the mask add layer.
         */
        private AttentionComponents traceAttentionScoresWithoutMask(SameDiff sd, OptimizationHelper helper, String varName) {
            log.debug("[ATTN-TRACE-NOMASK] Tracing from: " + varName);

            Variable v = getVariableWithFallback(helper, sd, varName);
            if (v == null) {
                log.debug("[ATTN-TRACE-NOMASK] Variable " + varName + " is null");
                return null;
            }

            String producerOpName = v.getOutputOfOp();
            if (producerOpName == null) {
                log.debug("[ATTN-TRACE-NOMASK] Variable " + varName + " has no producer op");
                return null;
            }

            SameDiffOp producerOp = sd.getOps().get(producerOpName);
            if (producerOp == null) {
                log.debug("[ATTN-TRACE-NOMASK] Producer op " + producerOpName + " not found");
                return null;
            }

            String opType = producerOp.getOp().getClass().getSimpleName();
            log.debug("[ATTN-TRACE-NOMASK] Producer op type: " + opType);

            // Direct matmul output
            if (producerOp.getOp() instanceof Mmul || producerOp.getOp() instanceof TensorMmul) {
                log.debug("[ATTN-TRACE-NOMASK] Found matmul: " + producerOpName);
                return extractQKFromMatmul(sd, helper, producerOp, varName, null, null, 1.0);
            }

            // Scaled output
            if (producerOp.getOp() instanceof MulOp || producerOp.getOp() instanceof ScalarMultiplication) {
                log.debug("[ATTN-TRACE-NOMASK] Found scale (mul): " + producerOpName);
                return traceScaledMatmul(sd, helper, producerOp, varName, true);
            }

            if (producerOp.getOp() instanceof DivOp || producerOp.getOp() instanceof ScalarDivision) {
                log.debug("[ATTN-TRACE-NOMASK] Found scale (div): " + producerOpName);
                return traceScaledMatmul(sd, helper, producerOp, varName, false);
            }

            // Reshape - trace through it
            if (producerOp.getOp() instanceof Reshape) {
                log.debug("[ATTN-TRACE-NOMASK] Tracing through reshape: " + producerOpName);
                List<String> reshapeInputs = producerOp.getInputsToOp();
                if (reshapeInputs != null && !reshapeInputs.isEmpty()) {
                    return traceAttentionScoresWithoutMask(sd, helper, reshapeInputs.get(0));
                }
            }

            // Permute/Transpose - trace through it
            if (producerOp.getOp() instanceof Permute || producerOp.getOp() instanceof Transpose) {
                log.debug("[ATTN-TRACE-NOMASK] Tracing through permute/transpose: " + producerOpName);
                List<String> permuteInputs = producerOp.getInputsToOp();
                if (permuteInputs != null && !permuteInputs.isEmpty()) {
                    return traceAttentionScoresWithoutMask(sd, helper, permuteInputs.get(0));
                }
            }

            log.debug("[ATTN-TRACE-NOMASK] Unhandled op type: " + opType);
            return null;
        }

        /**
         * Checks if an array looks like a causal (lower triangular) mask.
         */
        private boolean isCausalMaskArray(INDArray arr) {
            if (arr == null || arr.rank() < 2) return false;

            long[] shape = arr.shape();
            long rows = shape[shape.length - 2];
            long cols = shape[shape.length - 1];

            // Causal masks are typically square
            if (rows != cols) return false;

            try {
                // Check upper right corner (should be very negative for -inf)
                double upperRight = arr.getDouble(0, cols - 1);
                // Check lower left corner (should be 0 or close to 0)
                double lowerLeft = arr.getDouble(rows - 1, 0);

                return upperRight < -1e4 && Math.abs(lowerLeft) < 1e-6;
            } catch (Exception e) {
                return false;
            }
        }

        /**
         * Traces through a scaling operation to find the underlying matmul.
         */
        private AttentionComponents traceScaledMatmul(SameDiff sd, OptimizationHelper helper, SameDiffOp scaleOp,
                                                       String scaleOutputVar, boolean isMul) {
            log.debug("[ATTN-SCALE] Tracing scaled matmul from: " + scaleOp.getName() + " (isMul=" + isMul + ")");

            List<String> scaleInputs = scaleOp.getInputsToOp();
            if (scaleInputs == null || scaleInputs.size() < 2) {
                log.debug("[ATTN-SCALE] Scale op has insufficient inputs: " + (scaleInputs != null ? scaleInputs.size() : 0));
                return null;
            }

            log.debug("[ATTN-SCALE] Scale inputs: " + scaleInputs);

            // Find which input is the matmul output and which is the scale factor
            String matmulOutputVar = null;
            double scaleFactor = 1.0;
            // Track reshape between matmul and scale
            String reshapeOpName = null;
            String reshapeOutputVar = null;

            for (String input : scaleInputs) {
                log.debug("[ATTN-SCALE] Checking scale input: " + input);

                Variable inputVar = getVariableWithFallback(helper, sd, input);
                if (inputVar == null) {
                    log.debug("[ATTN-SCALE] Input " + input + " variable is null");
                    continue;
                }

                String inputOpName = inputVar.getOutputOfOp();
                log.debug("[ATTN-SCALE] Input " + input + " produced by: " + inputOpName);

                if (inputOpName != null) {
                    SameDiffOp inputOp = sd.getOps().get(inputOpName);
                    if (inputOp != null) {
                        String opType = inputOp.getOp().getClass().getSimpleName();
                        log.debug("[ATTN-SCALE] Input producer op type: " + opType);

                        if (inputOp.getOp() instanceof Mmul || inputOp.getOp() instanceof TensorMmul) {
                            matmulOutputVar = input;
                            log.debug("[ATTN-SCALE] Found matmul output: " + input);
                        }
                        // Also trace through reshape/permute to find matmul
                        else if (inputOp.getOp() instanceof Reshape || inputOp.getOp() instanceof Permute || inputOp.getOp() instanceof Transpose) {
                            log.debug("[ATTN-SCALE] Tracing through " + opType + " to find matmul...");
                            List<String> innerInputs = inputOp.getInputsToOp();
                            if (innerInputs != null && !innerInputs.isEmpty()) {
                                Variable innerVar = getVariableWithFallback(helper, sd, innerInputs.get(0));
                                if (innerVar != null) {
                                    String innerProducerName = innerVar.getOutputOfOp();
                                    if (innerProducerName != null) {
                                        SameDiffOp innerProducerOp = sd.getOps().get(innerProducerName);
                                        if (innerProducerOp != null && (innerProducerOp.getOp() instanceof Mmul || innerProducerOp.getOp() instanceof TensorMmul)) {
                                            matmulOutputVar = innerInputs.get(0);
                                            // Track the reshape op and its output for removal later
                                            reshapeOpName = inputOp.getName();
                                            reshapeOutputVar = input;
                                            log.debug("[ATTN-SCALE] Found matmul through " + opType + ": " + matmulOutputVar);
                                            log.debug("[ATTN-SCALE] Tracking reshape for removal: " + reshapeOpName + " -> " + reshapeOutputVar);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Check if this is a constant scalar (scale factor)
                SDVariable sdVar = sd.getVariable(input);
                if (sdVar != null && sdVar.getArr() != null) {
                    INDArray arr = sdVar.getArr();
                    if (arr.isScalar()) {
                        double val = arr.getDouble(0);
                        scaleFactor = isMul ? val : (1.0 / val);
                        log.debug("[ATTN-SCALE] Found scale factor: " + scaleFactor + " from " + input);
                    }
                }
            }

            if (matmulOutputVar == null) {
                log.debug("[ATTN-SCALE] Could not find matmul output!");
                return null;
            }

            Variable mmOutVar = getVariableWithFallback(helper, sd, matmulOutputVar);
            if (mmOutVar == null) {
                log.debug("[ATTN-SCALE] Matmul output variable is null");
                return null;
            }

            String mmOpName = mmOutVar.getOutputOfOp();
            if (mmOpName == null) {
                log.debug("[ATTN-SCALE] Matmul output has no producer op");
                return null;
            }

            SameDiffOp mmOp = sd.getOps().get(mmOpName);
            if (mmOp == null) {
                log.debug("[ATTN-SCALE] Matmul op not found: " + mmOpName);
                return null;
            }

            AttentionComponents components = extractQKFromMatmul(sd, helper, mmOp, matmulOutputVar, scaleOp.getName(), scaleOutputVar, scaleFactor);
            if (components != null) {
                // Add reshape tracking info
                components.reshapeBeforeScaleOpName = reshapeOpName;
                components.reshapeBeforeScaleOutputVar = reshapeOutputVar;
            }
            return components;
        }

        /**
         * Extracts Q and K variables from a matmul operation that computes Q @ K^T.
         */
        private AttentionComponents extractQKFromMatmul(SameDiff sd, OptimizationHelper helper, SameDiffOp matmulOp,
                                                         String matmulOutputVar,
                                                         String scaleOpName, String scaleOutputVar,
                                                         double scaleFactor) {
            log.debug("[ATTN-EXTRACT] Extracting Q,K from matmul: " + matmulOp.getName());

            List<String> mmInputs = matmulOp.getInputsToOp();
            if (mmInputs == null || mmInputs.size() < 2) {
                log.debug("[ATTN-EXTRACT] Matmul has insufficient inputs: " + (mmInputs != null ? mmInputs.size() : 0));
                return null;
            }

            log.debug("[ATTN-EXTRACT] Matmul inputs: " + mmInputs);

            String qVar = mmInputs.get(0);
            String kVar = mmInputs.get(1);

            log.debug("[ATTN-EXTRACT] Initial Q=" + qVar + ", K=" + kVar);

            // Check if K is transposed (either via Mmul transpose flag or explicit Transpose op)
            boolean kTransposed = false;

            if (matmulOp.getOp() instanceof Mmul) {
                Mmul mmul = (Mmul) matmulOp.getOp();
                // Check transpose flags via iArguments (index 1 is transposeB)
                if (mmul.numIArguments() > 1 && mmul.getIArgument(1) > 0) {
                    kTransposed = true;
                    log.debug("[ATTN-EXTRACT] K transposed via Mmul iArgs");
                }
            }

            // Check for explicit transpose on K
            Variable kVariable = getVariableWithFallback(helper, sd, kVar);
            if (kVariable != null) {
                String kProducerName = kVariable.getOutputOfOp();
                log.debug("[ATTN-EXTRACT] K produced by: " + kProducerName);
                if (kProducerName != null) {
                    SameDiffOp kProducerOp = sd.getOps().get(kProducerName);
                    if (kProducerOp != null) {
                        String kProducerType = kProducerOp.getOp().getClass().getSimpleName();
                        log.debug("[ATTN-EXTRACT] K producer op type: " + kProducerType);

                        if (kProducerOp.getOp() instanceof Transpose) {
                            // K comes from a transpose - get the original K
                            List<String> transposeInputs = kProducerOp.getInputsToOp();
                            if (transposeInputs != null && !transposeInputs.isEmpty()) {
                                kVar = transposeInputs.get(0);
                                kTransposed = true;
                                log.debug("[ATTN-EXTRACT] K traced through Transpose to: " + kVar);
                            }
                        }
                        // Also check for Permute
                        else if (kProducerOp.getOp() instanceof Permute) {
                            // K comes from permute - get the original K
                            List<String> permuteInputs = kProducerOp.getInputsToOp();
                            if (permuteInputs != null && !permuteInputs.isEmpty()) {
                                kVar = permuteInputs.get(0);
                                kTransposed = true; // Permute often serves as transpose for attention
                                log.debug("[ATTN-EXTRACT] K traced through Permute to: " + kVar);
                            }
                        }
                    }
                }
            }

            // For attention, K should be transposed (Q @ K^T)
            if (!kTransposed) {
                // This might not be an attention pattern, but continue anyway
                log.debug("[ATTN-EXTRACT] NOTE: K is not transposed - may not be attention pattern, but continuing");
            }

            AttentionComponents components = new AttentionComponents();
            components.queryVar = qVar;
            components.keyVar = kVar;
            components.qkMatmulOpName = matmulOp.getName();
            components.qkMatmulOutputVar = matmulOutputVar;
            components.scaleOpName = scaleOpName;
            components.scaleOutputVar = scaleOutputVar;
            components.scaleFactor = scaleFactor;

            log.debug("[ATTN-EXTRACT] SUCCESS: Q=" + qVar + ", K=" + kVar + ", scale=" + scaleFactor);
            return components;
        }

        /**
         * Checks if intermediate operations can be safely removed (only used in this attention pattern).
         */
        private boolean canSafelyFuse(SameDiff sd, OptimizationHelper helper, SameDiffOp softmaxOp, AttentionComponents components) {
            // Check softmax output is only used by one op (the final matmul)
            List<String> softmaxOutputs = softmaxOp.getOutputsOfOp();
            if (softmaxOutputs == null || softmaxOutputs.isEmpty()) {
                log.debug("[ATTN-FUSE] softmax has no outputs");
                return false;
            }

            Variable softmaxOutVar = getVariableWithFallback(helper, sd, softmaxOutputs.get(0));
            if (softmaxOutVar == null) {
                log.debug("[ATTN-FUSE] softmax output var " + softmaxOutputs.get(0) + " is null");
                return false;
            }

            List<String> softmaxUsers = softmaxOutVar.getInputsForOp();
            // Allow 1 or 2 users - sometimes there's a reshape between softmax and matmul that counts as a user
            if (softmaxUsers == null || softmaxUsers.isEmpty() || softmaxUsers.size() > 2) {
                log.debug("[ATTN-FUSE] softmax output " + softmaxOutputs.get(0) + " has " +
                    (softmaxUsers != null ? softmaxUsers.size() : 0) + " users (expected 1-2): " + softmaxUsers);
                return false;
            }
            log.debug("[ATTN-FUSE] softmax output " + softmaxOutputs.get(0) + " has " + softmaxUsers.size() + " users: " + softmaxUsers);

            // Check Q @ K^T output is only used once (or twice if there's a scale op)
            Variable qkOutVar = getVariableWithFallback(helper, sd, components.qkMatmulOutputVar);
            if (qkOutVar == null) {
                log.debug("[ATTN-FUSE] Q@K^T output var " + components.qkMatmulOutputVar + " is null");
                return false;
            }

            List<String> qkUsers = qkOutVar.getInputsForOp();
            if (qkUsers == null || qkUsers.isEmpty() || qkUsers.size() > 2) {
                log.debug("[ATTN-FUSE] Q@K^T output " + components.qkMatmulOutputVar + " has " +
                    (qkUsers != null ? qkUsers.size() : 0) + " users (expected 1-2): " + qkUsers);
                return false;
            }
            log.debug("[ATTN-FUSE] Q@K^T output " + components.qkMatmulOutputVar + " has " + qkUsers.size() + " users: " + qkUsers);

            log.debug("[ATTN-FUSE] All checks passed!");
            return true;
        }
    }

    /**
     * Infer the rank of a variable for fusion eligibility checks.
     * Tries multiple strategies:
     * 1. Static shape from SDVariable.getShape()
     * 2. Shape from constant/variable arrays
     * 3. Infer from producer op (Reshape shape input, Permute preserves rank)
     * @return inferred rank, or -1 if unknown
     */
    private static int inferVariableRank(SameDiff sd, SDVariable var) {
        return inferVariableRankImpl(sd, var, 0);
    }

    private static int inferVariableRankImpl(SameDiff sd, SDVariable var, int depth) {
        if (var == null || depth > 15) return -1;
        String indent = depth < 10 ? "                    ".substring(0, depth * 2) : "  ";

        // Strategy 1: static shape
        long[] shape = var.getShape();
        if (shape != null && shape.length > 0) {
            log.debug("[ATTN-RANK-DBG] " + indent + var.name() + " -> static shape len=" + shape.length);
            return shape.length;
        }

        // Strategy 2: check actual arrays for constants/variables
        if (var.getVariableType() == VariableType.CONSTANT) {
            INDArray arr = sd.getConstantArrays().getArray(var.name());
            if (arr != null) {
                log.debug("[ATTN-RANK-DBG] " + indent + var.name() + " -> constant rank=" + arr.rank());
                return arr.rank();
            }
        } else if (var.getVariableType() == VariableType.VARIABLE) {
            INDArray arr = sd.getVariablesArrays().getArray(var.name());
            if (arr != null) {
                log.debug("[ATTN-RANK-DBG] " + indent + var.name() + " -> variable rank=" + arr.rank());
                return arr.rank();
            }
        }

        // Strategy 3: infer from producer op
        Variable v = sd.getVariables().get(var.name());
        if (v == null || v.getOutputOfOp() == null) {
            log.debug("[ATTN-RANK-DBG] " + indent + var.name() + " -> type=" + var.getVariableType() + " no producer, FAIL");
            return -1;
        }

        SameDiffOp producerOp = sd.getOps().get(v.getOutputOfOp());
        if (producerOp == null || producerOp.getOp() == null) {
            log.debug("[ATTN-RANK-DBG] " + indent + var.name() + " -> producer op null, FAIL");
            return -1;
        }

        DifferentialFunction opFunc = producerOp.getOp();
        List<String> opInputs = producerOp.getInputsToOp();
        log.debug("[ATTN-RANK-DBG] " + indent + var.name() + " -> produced by " + opFunc.getClass().getSimpleName() + " inputs=" + opInputs);

        // Permute/Transpose preserve rank — determine from permutation or recurse
        if (opFunc instanceof Permute || opFunc instanceof Transpose) {
            // Check iArgs: for Permute created with long[] dims, iArgs = permutation array
            if (opFunc instanceof DynamicCustomOp) {
                long[] iArgs = ((DynamicCustomOp) opFunc).iArgs();
                if (iArgs != null && iArgs.length > 0) {
                    log.debug("[ATTN-RANK-DBG] " + indent + "  Permute/Transpose iArgs length=" + iArgs.length + " -> rank=" + iArgs.length);
                    return iArgs.length;
                }
            }
            // Check second input: for Permute with SDVariable permutation dims
            if (opInputs != null && opInputs.size() >= 2) {
                SDVariable permVar = sd.getVariable(opInputs.get(1));
                if (permVar != null && permVar.getVariableType() == VariableType.CONSTANT) {
                    INDArray permArr = sd.getConstantArrays().getArray(permVar.name());
                    if (permArr != null) {
                        log.debug("[ATTN-RANK-DBG] " + indent + "  Permute/Transpose perm constant length=" + permArr.length() + " -> rank=" + permArr.length());
                        return (int) permArr.length();
                    }
                }
            }
            // Fallback: recurse on data input
            if (opInputs != null && !opInputs.isEmpty()) {
                SDVariable input = sd.getVariable(opInputs.get(0));
                return inferVariableRankImpl(sd, input, depth + 1);
            }
        }

        // Reshape: output rank = length of shape input, or from iArguments
        if (opFunc instanceof Reshape) {
            log.debug("[ATTN-RANK-DBG] " + indent + "  Reshape: opInputs.size()=" + (opInputs != null ? opInputs.size() : "null"));
            // Check second input variable (dynamic shape from ONNX)
            if (opInputs != null && opInputs.size() >= 2) {
                SDVariable shapeVar = sd.getVariable(opInputs.get(1));
                log.debug("[ATTN-RANK-DBG] " + indent + "  Reshape shape var: " + (shapeVar != null ? shapeVar.name() + " type=" + shapeVar.getVariableType() : "NULL"));
                if (shapeVar != null) {
                    if (shapeVar.getVariableType() == VariableType.CONSTANT) {
                        INDArray shapeArr = sd.getConstantArrays().getArray(shapeVar.name());
                        log.debug("[ATTN-RANK-DBG] " + indent + "  Reshape shape constant array: " + (shapeArr != null ? "length=" + shapeArr.length() : "NULL"));
                        if (shapeArr != null) {
                            return (int) shapeArr.length();
                        }
                    }
                    long[] shapeShape = shapeVar.getShape();
                    log.debug("[ATTN-RANK-DBG] " + indent + "  Reshape shape var getShape(): " + java.util.Arrays.toString(shapeShape));
                    if (shapeShape != null && shapeShape.length == 1 && shapeShape[0] > 0) {
                        return (int) shapeShape[0];
                    }
                    // Trace: if shape variable is produced by Concat, count elements
                    Variable shapeVarInfo = sd.getVariables().get(shapeVar.name());
                    if (shapeVarInfo != null && shapeVarInfo.getOutputOfOp() != null) {
                        SameDiffOp shapeProducer = sd.getOps().get(shapeVarInfo.getOutputOfOp());
                        if (shapeProducer != null && shapeProducer.getOp() instanceof Concat) {
                            List<String> concatInputs = shapeProducer.getInputsToOp();
                            if (concatInputs != null && !concatInputs.isEmpty()) {
                                // Each Concat input is typically a scalar/1-element for reshape shapes
                                // Count total elements across all inputs
                                int totalElements = 0;
                                boolean allKnown = true;
                                for (String ci : concatInputs) {
                                    SDVariable ciVar = sd.getVariable(ci);
                                    if (ciVar == null) { allKnown = false; break; }
                                    if (ciVar.getVariableType() == VariableType.CONSTANT) {
                                        INDArray ciArr = sd.getConstantArrays().getArray(ci);
                                        if (ciArr != null) { totalElements += (int) ciArr.length(); continue; }
                                    }
                                    // Assume each non-constant input contributes 1 element (scalar)
                                    totalElements += 1;
                                }
                                if (allKnown && totalElements > 0) {
                                    log.debug("[ATTN-RANK-DBG] " + indent + "  Reshape shape from Concat: " + totalElements + " elements -> rank=" + totalElements);
                                    return totalElements;
                                }
                            }
                        }
                    }
                }
            }
            // Fallback: static shape stored in iArguments [orderFlag, dim0, dim1, ...]
            if (opFunc instanceof DynamicCustomOp) {
                long[] iArgs = ((DynamicCustomOp) opFunc).iArgs();
                log.debug("[ATTN-RANK-DBG] " + indent + "  Reshape iArgs: " + java.util.Arrays.toString(iArgs));
                if (iArgs != null && iArgs.length > 1) {
                    return iArgs.length - 1;
                }
            }
            return -1;
        }

        // Mmul/TensorMmul: batched matmul preserves rank from first input
        if (opFunc instanceof Mmul || opFunc instanceof TensorMmul) {
            if (opInputs != null && !opInputs.isEmpty()) {
                SDVariable input = sd.getVariable(opInputs.get(0));
                return inferVariableRankImpl(sd, input, depth + 1);
            }
        }

        // Generic fallback: recurse on first input for other ops
        // Covers element-wise ops (Add, Mul, Div, Cast, etc.) which preserve rank
        if (opInputs != null && !opInputs.isEmpty()) {
            SDVariable firstInput = sd.getVariable(opInputs.get(0));
            if (firstInput != null) {
                return inferVariableRankImpl(sd, firstInput, depth + 1);
            }
        }

        log.debug("[ATTN-RANK-DBG] " + indent + var.name() + " -> FAIL (no inputs)");
        return -1;
    }

    /**
     * Helper method to get a variable with fallback to graph lookup.
     * This ensures we can find variables that weren't in the initial cache.
     */
    private static Variable getVariableWithFallback(OptimizationHelper helper, SameDiff sd, String name) {
        Variable v = helper != null ? helper.getVariable(name) : null;
        if (v == null && sd != null) {
            v = sd.getVariables().get(name);
        }
        return v;
    }

    /**
     * Helper class to hold attention pattern components.
     */
    private static class AttentionComponents {
        String queryVar;
        String keyVar;
        String qkMatmulOpName;
        String qkMatmulOutputVar;
        String scaleOpName;
        String scaleOutputVar;
        double scaleFactor = 1.0;
        // Reshape between matmul and scale (if exists)
        String reshapeBeforeScaleOpName;
        String reshapeBeforeScaleOutputVar;
        // Mask-related fields
        String maskOpName;
        String maskOutputVar;
        String maskVar;
        boolean useCausalMask = false;
        boolean hasAdditiveMask = false;
    }

    /**
     * Fuses attention output with a subsequent linear projection.
     * Pattern: attention_output -> matmul(_, W) -> add(bias) => attention with fused output projection
     *
     * Note: This is a placeholder for future optimization. The current implementation
     * focuses on detecting and fusing the core attention pattern.
     */
    public static class FuseAttentionWithProjection implements Optimizer {

        private static final Set<Class<? extends DifferentialFunction>> APPLICABLE_OPS = new HashSet<>();
        static {
            APPLICABLE_OPS.add(DotProductAttentionV2.class);
        }

        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return APPLICABLE_OPS;
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            // Check if this is a DotProductAttentionV2 op
            if (!(op.getOp() instanceof DotProductAttentionV2)) {
                return false;
            }

            // Get attention output
            List<String> attentionOutputs = op.getOutputsOfOp();
            if (attentionOutputs == null || attentionOutputs.isEmpty()) {
                return false;
            }

            String attentionOutput = attentionOutputs.get(0);
            Variable attOutVar = helper.getVariable(attentionOutput);
            if (attOutVar == null) return false;

            // Check if output goes to a matmul (linear projection)
            List<String> users = attOutVar.getInputsForOp();
            if (users == null || users.size() != 1) {
                // Multiple users or no users - can't fuse
                return false;
            }

            // For now, just log that we found a potential fusion opportunity
            // Full implementation would fuse the projection into the attention op
            // or create a combined attention+projection op
            log.debug("[ATTN-DEBUG] Found potential attention+projection fusion opportunity at " + op.getName());

            // Return false for now - this is a placeholder for future optimization
            return false;
        }
    }

    /**
     * Detects attention patterns with causal (autoregressive) masking and fuses them.
     * Pattern: matmul(Q, K^T) [-> scale] -> add(causal_mask) -> softmax -> matmul(_, V)
     *
     * The causal mask is typically a lower triangular matrix with -inf in upper positions.
     * This optimizer detects the mask add, traces forward to find the complete pattern,
     * and creates DotProductAttentionV2 with useCausalMask=true.
     */
    public static class FuseAttentionWithCausalMask implements Optimizer {

        private static final Set<Class<? extends DifferentialFunction>> APPLICABLE_OPS = new HashSet<>();
        static {
            APPLICABLE_OPS.add(AddOp.class);
        }

        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return APPLICABLE_OPS;
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            // Look for add operations that could be causal mask application
            if (!(op.getOp() instanceof AddOp)) {
                return false;
            }

            List<String> addInputs = op.getInputsToOp();
            if (addInputs == null || addInputs.size() != 2) {
                return false;
            }

            // Check if one input comes from a matmul or scale operation (Q @ K^T pattern)
            String scoresVar = null;
            String maskVar = null;

            for (int i = 0; i < 2; i++) {
                String input = addInputs.get(i);
                Variable v = helper.getVariable(input);
                if (v == null) continue;

                String producerOpName = v.getOutputOfOp();
                if (producerOpName != null) {
                    SameDiffOp producerOp = sd.getOps().get(producerOpName);
                    if (producerOp != null &&
                        (producerOp.getOp() instanceof Mmul || producerOp.getOp() instanceof TensorMmul ||
                         producerOp.getOp() instanceof MulOp || producerOp.getOp() instanceof ScalarMultiplication ||
                         producerOp.getOp() instanceof DivOp || producerOp.getOp() instanceof ScalarDivision)) {
                        scoresVar = input;
                        maskVar = addInputs.get(1 - i);
                        break;
                    }
                }
            }

            if (scoresVar == null || maskVar == null) {
                return false;
            }

            // Check if the mask looks like a causal mask
            SDVariable maskSDVar = sd.getVariable(maskVar);
            if (maskSDVar == null) return false;

            INDArray maskArr = maskSDVar.getArr();
            if (maskArr == null) {
                return false;
            }

            if (!isCausalMask(maskArr)) {
                return false;
            }

            // Check if add output goes to softmax
            List<String> addOutputs = op.getOutputsOfOp();
            if (addOutputs == null || addOutputs.isEmpty()) {
                return false;
            }

            String addOutput = addOutputs.get(0);
            Variable addOutVar = helper.getVariable(addOutput);
            if (addOutVar == null) return false;

            List<String> addUsers = addOutVar.getInputsForOp();
            if (addUsers == null || addUsers.size() != 1) {
                return false;
            }

            SameDiffOp softmaxOp = sd.getOps().get(addUsers.get(0));
            if (softmaxOp == null || !(softmaxOp.getOp() instanceof SoftMax)) {
                return false;
            }

            // Check if softmax output goes to final matmul with V
            List<String> softmaxOutputs = softmaxOp.getOutputsOfOp();
            if (softmaxOutputs == null || softmaxOutputs.isEmpty()) {
                return false;
            }

            String softmaxOutput = softmaxOutputs.get(0);
            Variable softmaxOutVar = helper.getVariable(softmaxOutput);
            if (softmaxOutVar == null) return false;

            List<String> softmaxUsers = softmaxOutVar.getInputsForOp();
            if (softmaxUsers == null || softmaxUsers.size() != 1) {
                return false;
            }

            SameDiffOp finalMatmulOp = sd.getOps().get(softmaxUsers.get(0));
            if (finalMatmulOp == null ||
                !(finalMatmulOp.getOp() instanceof Mmul || finalMatmulOp.getOp() instanceof TensorMmul)) {
                return false;
            }

            // Found complete pattern! Now trace back to get Q and K
            AttentionComponents components = traceQKFromScores(sd, helper, scoresVar);
            if (components == null) {
                return false;
            }

            // Get V from final matmul
            List<String> finalMmInputs = finalMatmulOp.getInputsToOp();
            if (finalMmInputs == null || finalMmInputs.size() < 2) {
                return false;
            }
            String vVar = finalMmInputs.get(1);
            SDVariable vSDVar = sd.getVariable(vVar);
            if (vSDVar == null) return false;

            // Get attention output
            List<String> attentionOutputs = finalMatmulOp.getOutputsOfOp();
            if (attentionOutputs == null || attentionOutputs.isEmpty()) {
                return false;
            }
            String attentionOutputVar = attentionOutputs.get(0);

            log.debug("[ATTN-DEBUG] Fusing causal masked attention: Q=" +
                    components.queryVar + ", K=" + components.keyVar + ", V=" + vVar + " into dot_product_attention_v2");

            try {
                SDVariable qSDVar = sd.getVariable(components.queryVar);
                SDVariable kSDVar = sd.getVariable(components.keyVar);

                if (qSDVar == null || kSDVar == null) {
                    return false;
                }

                // Create dot_product_attention_v2 with causal mask enabled
                // Create explicit empty mask constants to avoid array access issues with intermediate variables
                SDVariable emptyQueryMask = sd.constant("attn_causal_empty_qmask_" + op.getName(), Nd4j.empty(DataType.FLOAT));
                SDVariable emptyValueMask = sd.constant("attn_causal_empty_vmask_" + op.getName(), Nd4j.empty(DataType.FLOAT));

                SDVariable fusedOutput = new DotProductAttentionV2(sd,
                        qSDVar,           // queries
                        vSDVar,           // values
                        kSDVar,           // keys
                        emptyQueryMask,   // queryMask (empty)
                        emptyValueMask,   // valueMask (empty)
                        components.scaleFactor,
                        0.0,              // no dropout for inference
                        true,             // use causal mask
                        false             // not training
                ).outputVariable();

                // Replace all uses of the attention output with the fused output
                OptimizationUtils.replaceOpInputsWith(sd, helper, attentionOutputVar, fusedOutput.name());

                // Remove old operations
                OptimizationUtils.removeOp(sd, helper, finalMatmulOp.getName());
                OptimizationUtils.removeOp(sd, helper, softmaxOp.getName());
                OptimizationUtils.removeOp(sd, helper, op.getName()); // The add op
                if (components.scaleOpName != null) {
                    OptimizationUtils.removeOp(sd, helper, components.scaleOpName);
                }
                OptimizationUtils.removeOp(sd, helper, components.qkMatmulOpName);

                // Remove intermediate variables
                OptimizationUtils.removeVariable(sd, helper, softmaxOutput);
                OptimizationUtils.removeVariable(sd, helper, addOutput);
                if (components.scaleOutputVar != null) {
                    OptimizationUtils.removeVariable(sd, helper, components.scaleOutputVar);
                }
                OptimizationUtils.removeVariable(sd, helper, components.qkMatmulOutputVar);
                OptimizationUtils.removeVariable(sd, helper, attentionOutputVar);

                return true;
            } catch (Exception e) {
                log.debug("[ATTN-WARN] Failed to fuse causal masked attention: " + e.getMessage());
                return false;
            }
        }

        /**
         * Traces back from attention scores to find Q and K variables.
         */
        private AttentionComponents traceQKFromScores(SameDiff sd, OptimizationHelper helper, String scoresVar) {
            Variable v = helper.getVariable(scoresVar);
            if (v == null) return null;

            String producerOpName = v.getOutputOfOp();
            if (producerOpName == null) return null;

            SameDiffOp producerOp = sd.getOps().get(producerOpName);
            if (producerOp == null) return null;

            // Direct matmul
            if (producerOp.getOp() instanceof Mmul || producerOp.getOp() instanceof TensorMmul) {
                return extractQKFromMatmulCausal(sd, helper, producerOp, scoresVar, null, null, 1.0);
            }

            // Scaled - trace through scale op
            if (producerOp.getOp() instanceof MulOp || producerOp.getOp() instanceof ScalarMultiplication ||
                producerOp.getOp() instanceof DivOp || producerOp.getOp() instanceof ScalarDivision) {
                return traceScaledQK(sd, helper, producerOp, scoresVar);
            }

            return null;
        }

        /**
         * Traces through a scale operation to find Q @ K^T.
         */
        private AttentionComponents traceScaledQK(SameDiff sd, OptimizationHelper helper, SameDiffOp scaleOp, String scaleOutputVar) {
            List<String> scaleInputs = scaleOp.getInputsToOp();
            if (scaleInputs == null || scaleInputs.size() < 2) {
                return null;
            }

            boolean isMul = scaleOp.getOp() instanceof MulOp || scaleOp.getOp() instanceof ScalarMultiplication;
            String matmulOutputVar = null;
            double scaleFactor = 1.0;

            for (String input : scaleInputs) {
                Variable inputVar = helper.getVariable(input);
                if (inputVar == null) continue;

                String inputOpName = inputVar.getOutputOfOp();
                if (inputOpName != null) {
                    SameDiffOp inputOp = sd.getOps().get(inputOpName);
                    if (inputOp != null && (inputOp.getOp() instanceof Mmul || inputOp.getOp() instanceof TensorMmul)) {
                        matmulOutputVar = input;
                    }
                }

                SDVariable sdVar = sd.getVariable(input);
                if (sdVar != null && sdVar.getArr() != null) {
                    INDArray arr = sdVar.getArr();
                    if (arr.isScalar()) {
                        double val = arr.getDouble(0);
                        scaleFactor = isMul ? val : (1.0 / val);
                    }
                }
            }

            if (matmulOutputVar == null) {
                return null;
            }

            Variable mmOutVar = helper.getVariable(matmulOutputVar);
            if (mmOutVar == null) return null;

            String mmOpName = mmOutVar.getOutputOfOp();
            if (mmOpName == null) return null;

            SameDiffOp mmOp = sd.getOps().get(mmOpName);
            if (mmOp == null) return null;

            return extractQKFromMatmulCausal(sd, helper, mmOp, matmulOutputVar, scaleOp.getName(), scaleOutputVar, scaleFactor);
        }

        /**
         * Extracts Q and K from a matmul operation.
         */
        private AttentionComponents extractQKFromMatmulCausal(SameDiff sd, OptimizationHelper helper, SameDiffOp matmulOp,
                                                         String matmulOutputVar,
                                                         String scaleOpName, String scaleOutputVar,
                                                         double scaleFactor) {
            List<String> mmInputs = matmulOp.getInputsToOp();
            if (mmInputs == null || mmInputs.size() < 2) {
                return null;
            }

            String qVar = mmInputs.get(0);
            String kVar = mmInputs.get(1);

            // Check for explicit transpose on K
            Variable kVariable = helper.getVariable(kVar);
            if (kVariable != null) {
                String kProducerName = kVariable.getOutputOfOp();
                if (kProducerName != null) {
                    SameDiffOp kProducerOp = sd.getOps().get(kProducerName);
                    if (kProducerOp != null && kProducerOp.getOp() instanceof Transpose) {
                        List<String> transposeInputs = kProducerOp.getInputsToOp();
                        if (transposeInputs != null && !transposeInputs.isEmpty()) {
                            kVar = transposeInputs.get(0);
                        }
                    }
                }
            }

            AttentionComponents components = new AttentionComponents();
            components.queryVar = qVar;
            components.keyVar = kVar;
            components.qkMatmulOpName = matmulOp.getName();
            components.qkMatmulOutputVar = matmulOutputVar;
            components.scaleOpName = scaleOpName;
            components.scaleOutputVar = scaleOutputVar;
            components.scaleFactor = scaleFactor;
            components.useCausalMask = true;

            return components;
        }

        /**
         * Checks if the given array looks like a causal (lower triangular) mask.
         */
        private boolean isCausalMask(INDArray arr) {
            if (arr.rank() < 2) return false;

            long[] shape = arr.shape();
            long rows = shape[shape.length - 2];
            long cols = shape[shape.length - 1];

            if (rows != cols) return false;

            try {
                double upperRight = arr.getDouble(0, cols - 1);
                double lowerLeft = arr.getDouble(rows - 1, 0);
                return upperRight < -1e4 && Math.abs(lowerLeft) < 1e-6;
            } catch (Exception e) {
                return false;
            }
        }
    }

    /**
     * Detects attention patterns with general additive masks (e.g., padding masks) and fuses them.
     * Pattern: matmul(Q, K^T) [-> scale] -> add(mask) -> softmax -> matmul(_, V)
     *
     * Unlike FuseAttentionWithCausalMask, this handles arbitrary additive masks that are
     * passed as value masks to the attention operation.
     */
    public static class FuseAttentionWithMask implements Optimizer {

        private static final Set<Class<? extends DifferentialFunction>> APPLICABLE_OPS = new HashSet<>();
        static {
            APPLICABLE_OPS.add(AddOp.class);
        }

        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return APPLICABLE_OPS;
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            // Look for add operations that could be mask application
            if (!(op.getOp() instanceof AddOp)) {
                return false;
            }

            List<String> addInputs = op.getInputsToOp();
            if (addInputs == null || addInputs.size() != 2) {
                return false;
            }

            // Check if one input comes from a matmul or scale operation (Q @ K^T pattern)
            String scoresVar = null;
            String maskVar = null;

            for (int i = 0; i < 2; i++) {
                String input = addInputs.get(i);
                Variable v = helper.getVariable(input);
                if (v == null) continue;

                String producerOpName = v.getOutputOfOp();
                if (producerOpName != null) {
                    SameDiffOp producerOp = sd.getOps().get(producerOpName);
                    if (producerOp != null &&
                        (producerOp.getOp() instanceof Mmul || producerOp.getOp() instanceof TensorMmul ||
                         producerOp.getOp() instanceof MulOp || producerOp.getOp() instanceof ScalarMultiplication ||
                         producerOp.getOp() instanceof DivOp || producerOp.getOp() instanceof ScalarDivision)) {
                        scoresVar = input;
                        maskVar = addInputs.get(1 - i);
                        break;
                    }
                }
            }

            if (scoresVar == null || maskVar == null) {
                return false;
            }

            // Get the mask variable - skip if it looks like a causal mask (handled by other optimizer)
            SDVariable maskSDVar = sd.getVariable(maskVar);
            if (maskSDVar == null) return false;

            INDArray maskArr = maskSDVar.getArr();
            if (maskArr != null && isCausalMaskCheck(maskArr)) {
                // Let FuseAttentionWithCausalMask handle this
                return false;
            }

            // Check if add output goes to softmax
            List<String> addOutputs = op.getOutputsOfOp();
            if (addOutputs == null || addOutputs.isEmpty()) {
                return false;
            }

            String addOutput = addOutputs.get(0);
            Variable addOutVar = helper.getVariable(addOutput);
            if (addOutVar == null) return false;

            List<String> addUsers = addOutVar.getInputsForOp();
            if (addUsers == null || addUsers.size() != 1) {
                return false;
            }

            SameDiffOp softmaxOp = sd.getOps().get(addUsers.get(0));
            if (softmaxOp == null || !(softmaxOp.getOp() instanceof SoftMax)) {
                return false;
            }

            // Check if softmax output goes to final matmul with V
            List<String> softmaxOutputs = softmaxOp.getOutputsOfOp();
            if (softmaxOutputs == null || softmaxOutputs.isEmpty()) {
                return false;
            }

            String softmaxOutput = softmaxOutputs.get(0);
            Variable softmaxOutVar = helper.getVariable(softmaxOutput);
            if (softmaxOutVar == null) return false;

            List<String> softmaxUsers = softmaxOutVar.getInputsForOp();
            if (softmaxUsers == null || softmaxUsers.size() != 1) {
                return false;
            }

            SameDiffOp finalMatmulOp = sd.getOps().get(softmaxUsers.get(0));
            if (finalMatmulOp == null ||
                !(finalMatmulOp.getOp() instanceof Mmul || finalMatmulOp.getOp() instanceof TensorMmul)) {
                return false;
            }

            // Found complete pattern! Now trace back to get Q and K
            AttentionComponents components = traceQKFromScoresInternal(sd, helper, scoresVar);
            if (components == null) {
                return false;
            }

            // Get V from final matmul
            List<String> finalMmInputs = finalMatmulOp.getInputsToOp();
            if (finalMmInputs == null || finalMmInputs.size() < 2) {
                return false;
            }
            String vVar = finalMmInputs.get(1);
            SDVariable vSDVar = sd.getVariable(vVar);
            if (vSDVar == null) return false;

            // Get attention output
            List<String> attentionOutputs = finalMatmulOp.getOutputsOfOp();
            if (attentionOutputs == null || attentionOutputs.isEmpty()) {
                return false;
            }
            String attentionOutputVar = attentionOutputs.get(0);

            log.debug("[ATTN-DEBUG] Fusing masked attention: Q=" +
                    components.queryVar + ", K=" + components.keyVar + ", V=" + vVar + ", mask=" + maskVar + " into dot_product_attention_v2");

            try {
                SDVariable qSDVar = sd.getVariable(components.queryVar);
                SDVariable kSDVar = sd.getVariable(components.keyVar);

                if (qSDVar == null || kSDVar == null) {
                    return false;
                }

                // Create dot_product_attention_v2 with value mask
                // Note: The additive mask applied before softmax corresponds to a value mask
                // Create explicit empty query mask constant to avoid array access issues with intermediate variables
                SDVariable emptyQueryMask = sd.constant("attn_masked_empty_qmask_" + op.getName(), Nd4j.empty(DataType.FLOAT));

                SDVariable fusedOutput = new DotProductAttentionV2(sd,
                        qSDVar,           // queries
                        vSDVar,           // values
                        kSDVar,           // keys
                        emptyQueryMask,   // queryMask (empty)
                        maskSDVar,        // valueMask - the additive mask
                        components.scaleFactor,
                        0.0,              // no dropout for inference
                        false,            // not causal mask
                        false             // not training
                ).outputVariable();

                // Replace all uses of the attention output with the fused output
                OptimizationUtils.replaceOpInputsWith(sd, helper, attentionOutputVar, fusedOutput.name());

                // Remove old operations
                OptimizationUtils.removeOp(sd, helper, finalMatmulOp.getName());
                OptimizationUtils.removeOp(sd, helper, softmaxOp.getName());
                OptimizationUtils.removeOp(sd, helper, op.getName()); // The add op
                if (components.scaleOpName != null) {
                    OptimizationUtils.removeOp(sd, helper, components.scaleOpName);
                }
                OptimizationUtils.removeOp(sd, helper, components.qkMatmulOpName);

                // Remove intermediate variables
                OptimizationUtils.removeVariable(sd, helper, softmaxOutput);
                OptimizationUtils.removeVariable(sd, helper, addOutput);
                if (components.scaleOutputVar != null) {
                    OptimizationUtils.removeVariable(sd, helper, components.scaleOutputVar);
                }
                OptimizationUtils.removeVariable(sd, helper, components.qkMatmulOutputVar);
                OptimizationUtils.removeVariable(sd, helper, attentionOutputVar);

                return true;
            } catch (Exception e) {
                log.debug("[ATTN-WARN] Failed to fuse masked attention: " + e.getMessage());
                return false;
            }
        }

        private AttentionComponents traceQKFromScoresInternal(SameDiff sd, OptimizationHelper helper, String scoresVar) {
            Variable v = helper.getVariable(scoresVar);
            if (v == null) return null;

            String producerOpName = v.getOutputOfOp();
            if (producerOpName == null) return null;

            SameDiffOp producerOp = sd.getOps().get(producerOpName);
            if (producerOp == null) return null;

            if (producerOp.getOp() instanceof Mmul || producerOp.getOp() instanceof TensorMmul) {
                return extractQKFromMatmulInternal(sd, helper, producerOp, scoresVar, null, null, 1.0);
            }

            if (producerOp.getOp() instanceof MulOp || producerOp.getOp() instanceof ScalarMultiplication ||
                producerOp.getOp() instanceof DivOp || producerOp.getOp() instanceof ScalarDivision) {
                return traceScaledQKInternal(sd, helper, producerOp, scoresVar);
            }

            return null;
        }

        private AttentionComponents traceScaledQKInternal(SameDiff sd, OptimizationHelper helper, SameDiffOp scaleOp, String scaleOutputVar) {
            List<String> scaleInputs = scaleOp.getInputsToOp();
            if (scaleInputs == null || scaleInputs.size() < 2) {
                return null;
            }

            boolean isMul = scaleOp.getOp() instanceof MulOp || scaleOp.getOp() instanceof ScalarMultiplication;
            String matmulOutputVar = null;
            double scaleFactor = 1.0;

            for (String input : scaleInputs) {
                Variable inputVar = helper.getVariable(input);
                if (inputVar == null) continue;

                String inputOpName = inputVar.getOutputOfOp();
                if (inputOpName != null) {
                    SameDiffOp inputOp = sd.getOps().get(inputOpName);
                    if (inputOp != null && (inputOp.getOp() instanceof Mmul || inputOp.getOp() instanceof TensorMmul)) {
                        matmulOutputVar = input;
                    }
                }

                SDVariable sdVar = sd.getVariable(input);
                if (sdVar != null && sdVar.getArr() != null) {
                    INDArray arr = sdVar.getArr();
                    if (arr.isScalar()) {
                        double val = arr.getDouble(0);
                        scaleFactor = isMul ? val : (1.0 / val);
                    }
                }
            }

            if (matmulOutputVar == null) {
                return null;
            }

            Variable mmOutVar = helper.getVariable(matmulOutputVar);
            if (mmOutVar == null) return null;

            String mmOpName = mmOutVar.getOutputOfOp();
            if (mmOpName == null) return null;

            SameDiffOp mmOp = sd.getOps().get(mmOpName);
            if (mmOp == null) return null;

            return extractQKFromMatmulInternal(sd, helper, mmOp, matmulOutputVar, scaleOp.getName(), scaleOutputVar, scaleFactor);
        }

        private AttentionComponents extractQKFromMatmulInternal(SameDiff sd, OptimizationHelper helper, SameDiffOp matmulOp,
                                                                  String matmulOutputVar,
                                                                  String scaleOpName, String scaleOutputVar,
                                                                  double scaleFactor) {
            List<String> mmInputs = matmulOp.getInputsToOp();
            if (mmInputs == null || mmInputs.size() < 2) {
                return null;
            }

            String qVar = mmInputs.get(0);
            String kVar = mmInputs.get(1);

            Variable kVariable = helper.getVariable(kVar);
            if (kVariable != null) {
                String kProducerName = kVariable.getOutputOfOp();
                if (kProducerName != null) {
                    SameDiffOp kProducerOp = sd.getOps().get(kProducerName);
                    if (kProducerOp != null && kProducerOp.getOp() instanceof Transpose) {
                        List<String> transposeInputs = kProducerOp.getInputsToOp();
                        if (transposeInputs != null && !transposeInputs.isEmpty()) {
                            kVar = transposeInputs.get(0);
                        }
                    }
                }
            }

            AttentionComponents components = new AttentionComponents();
            components.queryVar = qVar;
            components.keyVar = kVar;
            components.qkMatmulOpName = matmulOp.getName();
            components.qkMatmulOutputVar = matmulOutputVar;
            components.scaleOpName = scaleOpName;
            components.scaleOutputVar = scaleOutputVar;
            components.scaleFactor = scaleFactor;

            return components;
        }

        private boolean isCausalMaskCheck(INDArray arr) {
            if (arr.rank() < 2) return false;

            long[] shape = arr.shape();
            long rows = shape[shape.length - 2];
            long cols = shape[shape.length - 1];

            if (rows != cols) return false;

            try {
                double upperRight = arr.getDouble(0, cols - 1);
                double lowerLeft = arr.getDouble(rows - 1, 0);
                return upperRight < -1e4 && Math.abs(lowerLeft) < 1e-6;
            } catch (Exception e) {
                return false;
            }
        }
    }

    /**
     * Collects multiple single-head attention ops that operate on different heads
     * and potentially fuses them into a multi-head attention pattern.
     *
     * This is useful when models are exported with explicit head splitting.
     */
    public static class CollectMultiHeadAttention implements Optimizer {

        private static final Set<Class<? extends DifferentialFunction>> APPLICABLE_OPS = new HashSet<>();
        static {
            APPLICABLE_OPS.add(Reshape.class);
        }

        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return APPLICABLE_OPS;
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            // Look for reshape operations that split heads
            // Pattern: reshape([batch, seq, hidden]) -> reshape([batch, seq, num_heads, head_dim])
            //          -> transpose -> separate attention per head -> concat

            if (!(op.getOp() instanceof Reshape)) {
                return false;
            }

            // This is a placeholder for multi-head attention detection
            // Full implementation would:
            // 1. Detect head splitting pattern (reshape + transpose)
            // 2. Find parallel attention computations on each head
            // 3. Detect head concatenation at the end
            // 4. Replace with a single multi-head attention op

            return false;
        }
    }
}

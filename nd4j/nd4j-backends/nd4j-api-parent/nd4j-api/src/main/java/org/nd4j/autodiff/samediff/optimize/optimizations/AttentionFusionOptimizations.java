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
import org.nd4j.linalg.api.ops.impl.transforms.dtype.Cast;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
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
            // Check if producer is Cast (common in mixed-precision models) and trace through
            else if (producerOp.getOp() instanceof Cast) {
                List<String> castInputs = producerOp.getInputsToOp();
                if (castInputs != null && !castInputs.isEmpty()) {
                    Variable castInputVar = getVariableWithFallback(helper, sd, castInputs.get(0));
                    if (castInputVar != null) {
                        String castProducerName = castInputVar.getOutputOfOp();
                        if (castProducerName != null) {
                            SameDiffOp castProducer = sd.getOps().get(castProducerName);
                            if (castProducer != null && castProducer.getOp() instanceof SoftMax) {
                                softmaxOp = castProducer;
                                log.debug("[ATTN] Found cast " + producerOp.getName() + " between softmax and matmul");
                            }
                        }
                    }
                }
            }

            if (softmaxOp == null) {
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

                // dot_product_attention_v2 supports rank 2, 3, or 4 inputs.
                // Rank 4 uses BSHD (batch, seq, numHeads, headDim) format via FlashAttentionHelper::forward4D.
                int qRank = inferVariableRank(sd, qSDVar);
                int kRank = inferVariableRank(sd, kSDVar);
                int vRank = inferVariableRank(sd, vSDVar);

                // For unknown ranks (-1): infer from known ranks. In attention, Q/K/V must
                // be compatible for matmul, so if 2 of 3 have known rank, the third matches.
                if (qRank == -1 && kRank > 0) qRank = kRank;
                if (qRank == -1 && vRank > 0) qRank = vRank;
                if (kRank == -1 && qRank > 0) kRank = qRank;
                if (kRank == -1 && vRank > 0) kRank = vRank;
                if (vRank == -1 && qRank > 0) vRank = qRank;
                if (vRank == -1 && kRank > 0) vRank = kRank;

                log.debug("[ATTN-DIAG] Rank check (after inference): Q={}, K={}, V={}", qRank, kRank, vRank);

                // Reject unknown ranks (-1) and rank 5+.
                if (qRank < 2 || kRank < 2 || vRank < 2 || qRank > 4 || kRank > 4 || vRank > 4) {
                    log.debug("[ATTN-DIAG] Skipping: ranks not supported (Q={}, K={}, V={})", qRank, kRank, vRank);
                    return false;
                }

                // For rank 4: The pattern typically has permute([0,2,1,3]) converting BSHD→BHSD
                // before the attention matmuls. The C++ op expects BSHD, so we absorb the upstream
                // permutes and pass the pre-permute BSHD variables directly.
                // Track permute ops to remove.
                List<String> permuteOpsToRemove = new ArrayList<>();
                List<String> permuteVarsToRemove = new ArrayList<>();
                // Track downstream permute on attention output (BHSD→BSHD)
                String downstreamPermuteOpName = null;
                String downstreamPermuteOutputVar = null;

                if (qRank == 4 || kRank == 4 || vRank == 4) {
                    // Absorb upstream permute([0,2,1,3]) on Q
                    String[] qAbsorbed = absorbUpstreamPermute0213(sd, helper, components.queryVar);
                    if (qAbsorbed != null) {
                        log.debug("[ATTN-R4] Absorbing Q permute: {} -> {}", components.queryVar, qAbsorbed[0]);
                        permuteOpsToRemove.add(qAbsorbed[1]);
                        permuteVarsToRemove.add(components.queryVar);
                        components.queryVar = qAbsorbed[0];
                        qSDVar = sd.getVariable(components.queryVar);
                    }

                    // Absorb upstream permute on K.
                    // Two patterns are handled:
                    //   permute(0,2,1,3): BSHD→BHSD (standard multi-head layout swap)
                    //   permute(0,2,3,1): BSHD→BHDS (K transposed for Q@K^T matmul)
                    // In both cases absorbing recovers the original BSHD variable, which is
                    // what dot_product_attention_v2 expects (it handles K transposition internally).
                    String[] kAbsorbed = absorbUpstreamPermute0213(sd, helper, components.keyVar);
                    if (kAbsorbed == null) {
                        // Try the K-transposed pattern: permute(BSHD, 0,2,3,1) → BHDS
                        kAbsorbed = absorbPermute0231(sd, helper, components.keyVar);
                        if (kAbsorbed != null) {
                            log.debug("[ATTN-R4] Absorbing K transpose-permute (0,2,3,1): {} -> {}", components.keyVar, kAbsorbed[0]);
                        }
                    }
                    if (kAbsorbed != null) {
                        log.debug("[ATTN-R4] Absorbing K permute: {} -> {}", components.keyVar, kAbsorbed[0]);
                        permuteOpsToRemove.add(kAbsorbed[1]);
                        permuteVarsToRemove.add(components.keyVar);
                        components.keyVar = kAbsorbed[0];
                        kSDVar = sd.getVariable(components.keyVar);
                    }

                    // Absorb upstream permute([0,2,1,3]) on V
                    String[] vAbsorbed = absorbUpstreamPermute0213(sd, helper, vVar);
                    if (vAbsorbed != null) {
                        log.debug("[ATTN-R4] Absorbing V permute: {} -> {}", vVar, vAbsorbed[0]);
                        permuteOpsToRemove.add(vAbsorbed[1]);
                        permuteVarsToRemove.add(vVar);
                        vVar = vAbsorbed[0];
                        vSDVar = sd.getVariable(vVar);
                    }

                    // Detect downstream permute([0,2,1,3]) on attention output (BHSD→BSHD).
                    // The fused op outputs BSHD directly, so we can absorb this permute.
                    Variable attOutVarInfo = getVariableWithFallback(helper, sd, attentionOutputVar);
                    if (attOutVarInfo != null) {
                        List<String> attUsers = attOutVarInfo.getInputsForOp();
                        if (attUsers != null && attUsers.size() == 1) {
                            SameDiffOp userOp = sd.getOps().get(attUsers.get(0));
                            if (userOp != null && isPermute0213(userOp)) {
                                List<String> userOutputs = userOp.getOutputsOfOp();
                                if (userOutputs != null && !userOutputs.isEmpty()) {
                                    downstreamPermuteOpName = userOp.getName();
                                    downstreamPermuteOutputVar = userOutputs.get(0);
                                    log.debug("[ATTN-R4] Absorbing downstream permute: {} -> output {}",
                                            downstreamPermuteOpName, downstreamPermuteOutputVar);
                                }
                            }
                        }
                    }

                    log.debug("[ATTN-R4] After permute absorption: Q={}, K={}, V={}", components.queryVar, components.keyVar, vVar);
                }

                // Create dot_product_attention_v2 op
                // Note: We use keys=K (not K^T) because the op handles transposition internally
                SDVariable emptyQueryMask = sd.constant("attn_empty_qmask_" + op.getName(), Nd4j.empty(DataType.FLOAT));

                // Use the detected mask if available, otherwise use empty mask.
                // When useCausalMask is true, use empty valueMask — FlashAttention
                // handles causal masking internally via config.isCausal, which is much
                // faster than passing an explicit mask (which would disable FlashAttention).
                SDVariable valueMask;
                if (components.hasAdditiveMask && components.maskVar != null && !components.useCausalMask) {
                    // Non-causal additive mask: rank 4 not supported by AttentionHelper fallback
                    if (qRank == 4 || kRank == 4 || vRank == 4) {
                        log.debug("[ATTN-DIAG] Skipping rank-4 fusion: non-causal additive mask not supported in flash path");
                        return false;
                    }
                    valueMask = sd.getVariable(components.maskVar);
                    if (valueMask == null) {
                        valueMask = sd.constant("attn_empty_vmask_" + op.getName(), Nd4j.empty(DataType.FLOAT));
                    }
                    log.debug("[ATTN-DEBUG] Using detected mask variable: " + components.maskVar);
                } else {
                    // No mask, or causal mask (FlashAttention handles causal via isCausal flag)
                    valueMask = sd.constant("attn_empty_vmask_" + op.getName(), Nd4j.empty(DataType.FLOAT));
                }

                SDVariable fusedOutput = new DotProductAttentionV2(sd,
                        qSDVar,           // queries
                        vSDVar,           // values
                        kSDVar,           // keys
                        emptyQueryMask,   // queryMask (empty)
                        valueMask,        // valueMask - use detected mask if available
                        null, null, null, null,  // no KV cache, no attention bias
                        components.scaleFactor,  // scale factor
                        0.0,              // no dropout for inference
                        components.useCausalMask,  // use causal mask if detected
                        false             // not training
                ).outputVariable();

                // For rank 4 with downstream permute absorption: replace the permute output's users
                // (the fused op already outputs BSHD, which is what comes after the permute)
                if (downstreamPermuteOutputVar != null) {
                    OptimizationUtils.replaceOpInputsWith(sd, helper, downstreamPermuteOutputVar, fusedOutput.name());
                } else {
                    // Replace all uses of the attention output with the fused output
                    OptimizationUtils.replaceOpInputsWith(sd, helper, attentionOutputVar, fusedOutput.name());
                }

                // Remove old operations in reverse order
                OptimizationUtils.removeOp(sd, helper, op.getName());  // final matmul

                // Remove downstream permute if absorbed
                if (downstreamPermuteOpName != null) {
                    OptimizationUtils.removeOp(sd, helper, downstreamPermuteOpName);
                }

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

                // Remove absorbed upstream permute ops
                for (String permuteOpName : permuteOpsToRemove) {
                    OptimizationUtils.removeOp(sd, helper, permuteOpName);
                }

                // Remove intermediate variables
                OptimizationUtils.removeVariable(sd, helper, potentialSoftmaxOutput);

                // Remove downstream permute output variable
                if (downstreamPermuteOutputVar != null) {
                    OptimizationUtils.removeVariable(sd, helper, downstreamPermuteOutputVar);
                }

                // Remove absorbed upstream permute output variables
                for (String permuteVar : permuteVarsToRemove) {
                    OptimizationUtils.removeVariable(sd, helper, permuteVar);
                }

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
         * Checks if an op is a permute([0,2,1,3]) — swapping dims 1 and 2.
         * This is the standard BSHD↔BHSD conversion in multi-head attention.
         */
        private boolean isPermute0213(SameDiffOp op) {
            return checkPermute0213(op);
        }

        private String[] absorbUpstreamPermute0213(SameDiff sd, OptimizationHelper helper, String varName) {
            return absorbPermute0213(sd, helper, varName);
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

            // Post-loop causal mask check: the loop may have found scores first (break)
            // without checking if the other input is a causal mask array.
            if (maskVar != null && !isCausalMask) {
                SDVariable maskSdVar = sd.getVariable(maskVar);
                if (maskSdVar != null && maskSdVar.getArr() != null) {
                    if (isCausalMaskArray(maskSdVar.getArr())) {
                        isCausalMask = true;
                        log.debug("[ATTN-MASK] Detected causal mask post-loop: " + maskVar);
                    }
                }
                if (!isCausalMask && isDynamicCausalMaskGraph(sd, helper, maskVar)) {
                    isCausalMask = true;
                    log.debug("[ATTN-MASK] Detected dynamic causal mask graph: " + maskVar);
                }
            }

            log.debug("[ATTN-MASK] Scores var: " + scoresVar + ", Mask var: " + maskVar + ", isCausal: " + isCausalMask);

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
                // Build full-rank index arrays for upper-right and lower-left corners.
                // For rank > 2 (e.g., [1,1,S,S]), leading dims are 0.
                long[] upperRightIdx = new long[arr.rank()];
                long[] lowerLeftIdx = new long[arr.rank()];
                Arrays.fill(upperRightIdx, 0);
                Arrays.fill(lowerLeftIdx, 0);
                // Upper right: row=0, col=cols-1
                upperRightIdx[arr.rank() - 1] = cols - 1;
                // Lower left: row=rows-1, col=0
                lowerLeftIdx[arr.rank() - 2] = rows - 1;

                double upperRight = arr.getDouble(upperRightIdx);
                double lowerLeft = arr.getDouble(lowerLeftIdx);

                return upperRight < -1e4 && Math.abs(lowerLeft) < 1e-6;
            } catch (Exception e) {
                return false;
            }
        }

        /**
         * Heuristic for dynamic causal masks that are not materialized as constant arrays.
         * Looks for range+comparison based causal construction with large negative mask values.
         */
        private boolean isDynamicCausalMaskGraph(SameDiff sd, OptimizationHelper helper, String maskVar) {
            if (maskVar == null) return false;

            String lowerName = maskVar.toLowerCase();
            if (lowerName.contains("causal") && lowerName.contains("mask")) {
                return true;
            }

            // SmolDocling-style dynamic causal mask subgraph naming
            if (lowerName.contains("attn_mask_reformat")) {
                return true;
            }

            Set<String> visitedVars = new HashSet<>();
            ArrayList<String> queue = new ArrayList<>();
            queue.add(maskVar);

            boolean hasCompare = false;
            boolean hasRange = false;
            boolean hasLargeNegativeConstant = false;
            int idx = 0;

            while (idx < queue.size() && idx < 256) {
                String currentVar = queue.get(idx++);
                if (!visitedVars.add(currentVar)) {
                    continue;
                }

                Variable v = getVariableWithFallback(helper, sd, currentVar);
                if (v == null) continue;
                String producerName = v.getOutputOfOp();
                if (producerName == null) continue;

                SameDiffOp producerOp = sd.getOps().get(producerName);
                if (producerOp == null || producerOp.getOp() == null) continue;

                String opName = producerOp.getOp().opName();
                String opNameLower = opName != null ? opName.toLowerCase() : "";

                if ("trilu".equals(opNameLower) || "tril".equals(opNameLower) || "triu".equals(opNameLower)) {
                    return true;
                }
                if ("less".equals(opNameLower) || "less_equal".equals(opNameLower) ||
                        "greater".equals(opNameLower) || "greater_equal".equals(opNameLower)) {
                    hasCompare = true;
                }
                if ("range".equals(opNameLower)) {
                    hasRange = true;
                }

                List<String> inputs = producerOp.getInputsToOp();
                if (inputs == null) continue;
                for (String in : inputs) {
                    SDVariable inVar = sd.getVariable(in);
                    if (inVar != null && inVar.getArr() != null && inVar.getArr().isScalar()) {
                        double d = inVar.getArr().getDouble(0);
                        if (d < -1e4) {
                            hasLargeNegativeConstant = true;
                        }
                    }
                    if (!visitedVars.contains(in)) {
                        queue.add(in);
                    }
                }
            }

            return hasCompare && hasRange && hasLargeNegativeConstant;
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
     * Checks if an op is a permute([0,2,1,3]) — swapping dims 1 and 2.
     * Shared by multiple attention fusion optimizers.
     */
    static boolean checkPermute0213(SameDiffOp op) {
        if (op == null || !(op.getOp() instanceof Permute)) {
            return false;
        }
        DynamicCustomOp permOp = (DynamicCustomOp) op.getOp();
        long[] iArgs = permOp.iArgs();
        if (iArgs != null && iArgs.length == 4) {
            return iArgs[0] == 0 && iArgs[1] == 2 && iArgs[2] == 1 && iArgs[3] == 3;
        }
        // Also check second input (constant permutation array)
        List<String> inputs = op.getInputsToOp();
        if (inputs != null && inputs.size() >= 2) {
            SameDiff sd2 = permOp.getSameDiff();
            if (sd2 != null) {
                SDVariable permVar = sd2.getVariable(inputs.get(1));
                if (permVar != null && permVar.getVariableType() == VariableType.CONSTANT) {
                    INDArray permArr = sd2.getConstantArrays().getArray(permVar.name());
                    if (permArr != null && permArr.length() == 4) {
                        return permArr.getLong(0) == 0 && permArr.getLong(1) == 2 &&
                               permArr.getLong(2) == 1 && permArr.getLong(3) == 3;
                    }
                }
            }
        }
        return false;
    }

    /**
     * If the given variable is the output of a permute([0,2,1,3]), returns the
     * pre-permute variable name and the permute op name as [varName, opName].
     * Otherwise returns null. Shared by multiple attention fusion optimizers.
     */
    static String[] absorbPermute0213(SameDiff sd, OptimizationHelper helper, String varName) {
        Variable v = getVariableWithFallback(helper, sd, varName);
        if (v == null) return null;
        String producerOpName = v.getOutputOfOp();
        if (producerOpName == null) return null;
        SameDiffOp producerOp = sd.getOps().get(producerOpName);
        if (producerOp == null) return null;
        if (checkPermute0213(producerOp)) {
            List<String> inputs = producerOp.getInputsToOp();
            if (inputs != null && !inputs.isEmpty()) {
                return new String[] { inputs.get(0), producerOpName };
            }
        }
        return null;
    }

    /**
     * Checks if an op is a permute([0,2,3,1]) — the K-transposed pattern used in multi-head
     * attention where K is reshaped to BSHD and then transposed to BHDS for the Q@K^T matmul.
     * permute(BSHD, 0,2,3,1) → BHDS  (K^T for attention scores matmul)
     * Absorbing this permute recovers the original BSHD variable, which is what
     * dot_product_attention_v2 expects as its key input.
     */
    static boolean checkPermute0231(SameDiffOp op) {
        if (op == null || !(op.getOp() instanceof Permute)) {
            return false;
        }
        DynamicCustomOp permOp = (DynamicCustomOp) op.getOp();
        long[] iArgs = permOp.iArgs();
        if (iArgs != null && iArgs.length == 4) {
            return iArgs[0] == 0 && iArgs[1] == 2 && iArgs[2] == 3 && iArgs[3] == 1;
        }
        // Also check second input (constant permutation array)
        List<String> inputs = op.getInputsToOp();
        if (inputs != null && inputs.size() >= 2) {
            SameDiff sd2 = permOp.getSameDiff();
            if (sd2 != null) {
                SDVariable permVar = sd2.getVariable(inputs.get(1));
                if (permVar != null && permVar.getVariableType() == VariableType.CONSTANT) {
                    INDArray permArr = sd2.getConstantArrays().getArray(permVar.name());
                    if (permArr != null && permArr.length() == 4) {
                        return permArr.getLong(0) == 0 && permArr.getLong(1) == 2 &&
                               permArr.getLong(2) == 3 && permArr.getLong(3) == 1;
                    }
                }
            }
        }
        return false;
    }

    /**
     * If the given variable is the output of a permute([0,2,3,1]) (the K-transposed BHDS pattern),
     * returns the pre-permute variable name and the permute op name as [varName, opName].
     * Otherwise returns null. Used to absorb the K transposition in rank-4 attention fusion,
     * recovering the original BSHD key variable for dot_product_attention_v2.
     */
    static String[] absorbPermute0231(SameDiff sd, OptimizationHelper helper, String varName) {
        Variable v = getVariableWithFallback(helper, sd, varName);
        if (v == null) return null;
        String producerOpName = v.getOutputOfOp();
        if (producerOpName == null) return null;
        SameDiffOp producerOp = sd.getOps().get(producerOpName);
        if (producerOp == null) return null;
        if (checkPermute0231(producerOp)) {
            List<String> inputs = producerOp.getInputsToOp();
            if (inputs != null && !inputs.isEmpty()) {
                return new String[] { inputs.get(0), producerOpName };
            }
        }
        return null;
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

                // Rank-4 permute absorption: dot_product_attention_v2 expects BSHD format.
                // The pattern typically has permute([0,2,1,3]) converting BSHD→BHSD before
                // the attention matmuls. Absorb these permutes to pass BSHD directly.
                int qRank = inferVariableRank(sd, qSDVar);
                int kRank = inferVariableRank(sd, kSDVar);
                int vRank = inferVariableRank(sd, vSDVar);

                // Infer unknown ranks from known
                if (qRank == -1 && kRank > 0) qRank = kRank;
                if (qRank == -1 && vRank > 0) qRank = vRank;
                if (kRank == -1 && qRank > 0) kRank = qRank;
                if (vRank == -1 && qRank > 0) vRank = qRank;

                List<String> permuteOpsToRemove = new ArrayList<>();
                List<String> permuteVarsToRemove = new ArrayList<>();
                String downstreamPermuteOpName = null;
                String downstreamPermuteOutputVar = null;

                log.debug("[ATTN-CAUSAL] Rank check: Q={}, K={}, V={}", qRank, kRank, vRank);
                if (qRank == 4 || kRank == 4 || vRank == 4) {
                    // Absorb upstream permute([0,2,1,3]) on Q
                    String[] qAbsorbed = absorbPermute0213(sd, helper, components.queryVar);
                    if (qAbsorbed != null) {
                        permuteOpsToRemove.add(qAbsorbed[1]);
                        permuteVarsToRemove.add(components.queryVar);
                        components.queryVar = qAbsorbed[0];
                        qSDVar = sd.getVariable(components.queryVar);
                    }

                    // Absorb upstream permute on K.
                    // Two patterns are handled:
                    //   permute(0,2,1,3): BSHD→BHSD (standard multi-head layout swap)
                    //   permute(0,2,3,1): BSHD→BHDS (K transposed for Q@K^T matmul)
                    // In both cases absorbing recovers the original BSHD variable.
                    String[] kAbsorbed = absorbPermute0213(sd, helper, components.keyVar);
                    if (kAbsorbed == null) {
                        kAbsorbed = absorbPermute0231(sd, helper, components.keyVar);
                        if (kAbsorbed != null) {
                            log.debug("[ATTN-R4-CAUSAL] Absorbing K transpose-permute (0,2,3,1): {} -> {}", components.keyVar, kAbsorbed[0]);
                        }
                    }
                    if (kAbsorbed != null) {
                        permuteOpsToRemove.add(kAbsorbed[1]);
                        permuteVarsToRemove.add(components.keyVar);
                        components.keyVar = kAbsorbed[0];
                        kSDVar = sd.getVariable(components.keyVar);
                    }

                    // Absorb upstream permute([0,2,1,3]) on V
                    String[] vAbsorbed = absorbPermute0213(sd, helper, vVar);
                    if (vAbsorbed != null) {
                        permuteOpsToRemove.add(vAbsorbed[1]);
                        permuteVarsToRemove.add(vVar);
                        vVar = vAbsorbed[0];
                        vSDVar = sd.getVariable(vVar);
                    }

                    // Detect downstream permute([0,2,1,3]) on attention output (BHSD→BSHD).
                    // The fused op outputs BSHD directly, so we can absorb this permute.
                    Variable attOutVarInfo = getVariableWithFallback(helper, sd, attentionOutputVar);
                    if (attOutVarInfo != null) {
                        List<String> attUsers = attOutVarInfo.getInputsForOp();
                        if (attUsers != null && attUsers.size() == 1) {
                            SameDiffOp userOp = sd.getOps().get(attUsers.get(0));
                            if (userOp != null && checkPermute0213(userOp)) {
                                List<String> userOutputs = userOp.getOutputsOfOp();
                                if (userOutputs != null && !userOutputs.isEmpty()) {
                                    downstreamPermuteOpName = userOp.getName();
                                    downstreamPermuteOutputVar = userOutputs.get(0);
                                }
                            }
                        }
                    }
                }

                log.debug("[ATTN-CAUSAL] Final Q={} K={} V={} downPerm={} scale={}",
                        components.queryVar, components.keyVar, vVar, downstreamPermuteOutputVar, components.scaleFactor);

                // Create dot_product_attention_v2 with causal mask enabled
                SDVariable emptyQueryMask = sd.constant("attn_causal_empty_qmask_" + op.getName(), Nd4j.empty(DataType.FLOAT));
                SDVariable emptyValueMask = sd.constant("attn_causal_empty_vmask_" + op.getName(), Nd4j.empty(DataType.FLOAT));

                SDVariable fusedOutput = new DotProductAttentionV2(sd,
                        qSDVar,           // queries
                        vSDVar,           // values
                        kSDVar,           // keys
                        emptyQueryMask,   // queryMask (empty)
                        emptyValueMask,   // valueMask (empty)
                        null, null, null, null,  // no KV cache, no attention bias
                        components.scaleFactor,
                        0.0,              // no dropout for inference
                        true,             // use causal mask
                        false             // not training
                ).outputVariable();

                // For rank 4 with downstream permute absorption: replace the permute output's users
                if (downstreamPermuteOutputVar != null) {
                    OptimizationUtils.replaceOpInputsWith(sd, helper, downstreamPermuteOutputVar, fusedOutput.name());
                } else {
                    OptimizationUtils.replaceOpInputsWith(sd, helper, attentionOutputVar, fusedOutput.name());
                }

                // Remove old operations
                OptimizationUtils.removeOp(sd, helper, finalMatmulOp.getName());
                if (downstreamPermuteOpName != null) {
                    OptimizationUtils.removeOp(sd, helper, downstreamPermuteOpName);
                }
                OptimizationUtils.removeOp(sd, helper, softmaxOp.getName());
                OptimizationUtils.removeOp(sd, helper, op.getName()); // The add op
                if (components.scaleOpName != null) {
                    OptimizationUtils.removeOp(sd, helper, components.scaleOpName);
                }
                OptimizationUtils.removeOp(sd, helper, components.qkMatmulOpName);

                // Remove absorbed upstream permute ops
                for (String permuteOpName : permuteOpsToRemove) {
                    OptimizationUtils.removeOp(sd, helper, permuteOpName);
                }

                // Remove intermediate variables
                OptimizationUtils.removeVariable(sd, helper, softmaxOutput);
                OptimizationUtils.removeVariable(sd, helper, addOutput);
                if (downstreamPermuteOutputVar != null) {
                    OptimizationUtils.removeVariable(sd, helper, downstreamPermuteOutputVar);
                }
                for (String permuteVar : permuteVarsToRemove) {
                    OptimizationUtils.removeVariable(sd, helper, permuteVar);
                }
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
                // Always check for scalar scale value first — constants may not be in the helper
                // and would be silently skipped if we gated this check behind inputVar != null.
                SDVariable sdVar = sd.getVariable(input);
                if (sdVar != null && sdVar.getArr() != null) {
                    INDArray arr = sdVar.getArr();
                    if (arr.isScalar()) {
                        double val = arr.getDouble(0);
                        scaleFactor = isMul ? val : (1.0 / val);
                    }
                }

                // For matmul identification we need the Variable metadata (producer op).
                Variable inputVar = getVariableWithFallback(helper, sd, input);
                if (inputVar == null) continue;

                String inputOpName = inputVar.getOutputOfOp();
                if (inputOpName != null) {
                    SameDiffOp inputOp = sd.getOps().get(inputOpName);
                    if (inputOp != null && (inputOp.getOp() instanceof Mmul || inputOp.getOp() instanceof TensorMmul)) {
                        matmulOutputVar = input;
                    }
                }
            }

            if (matmulOutputVar == null) {
                return null;
            }

            Variable mmOutVar = getVariableWithFallback(helper, sd, matmulOutputVar);
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
                    if (kProducerOp != null &&
                        kProducerOp.getOp() instanceof Transpose) {
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
            return isCausalMaskArray(arr);
        }
    }

    /**
     * Checks if the given array looks like a causal (lower triangular) mask.
     * Handles arrays of any rank >= 2 by using full-dimensional indexing.
     */
    static boolean isCausalMaskArray(INDArray arr) {
        if (arr.rank() < 2) return false;

        long[] shape = arr.shape();
        long rows = shape[shape.length - 2];
        long cols = shape[shape.length - 1];

        if (rows != cols) return false;

        try {
            // Build full-dimensional indices: zeros for all leading dims,
            // then the target row/col for the last two dims
            long[] upperRightIdx = new long[shape.length];
            upperRightIdx[shape.length - 1] = cols - 1;
            // all other dims stay 0

            long[] lowerLeftIdx = new long[shape.length];
            lowerLeftIdx[shape.length - 2] = rows - 1;
            // last dim stays 0

            double upperRight = arr.getDouble(upperRightIdx);
            double lowerLeft = arr.getDouble(lowerLeftIdx);
            return upperRight < -1e4 && Math.abs(lowerLeft) < 1e-6;
        } catch (Exception e) {
            return false;
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

            // Detect BERT/encoder additive attention bias: (1 - attention_mask) * -large_negative_value
            // This pattern produces a float tensor with values {0.0, -3.4e38} (or similar large negative)
            // that is ADDED to QK^T scores before softmax to mask out padding positions.
            // DotProductAttentionV2 expects a boolean mask (non-zero=keep), NOT an additive bias,
            // so fusing this pattern would completely corrupt attention for all 12 BERT layers.
            // Return false here so the original Add→Softmax→@V graph is kept intact.
            if (isAdditiveBiasAttnMask(sd, helper, maskVar, constantArrays)) {
                log.debug("[ATTN-MASK] Skipping FuseAttentionWithMask: mask is BERT-style additive bias (large-negative Mul), not a boolean mask");
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

                // Rank-4 permute absorption: dot_product_attention_v2 expects BSHD format.
                int qRank = inferVariableRank(sd, qSDVar);
                int kRank = inferVariableRank(sd, kSDVar);
                int vRank = inferVariableRank(sd, vSDVar);

                if (qRank == -1 && kRank > 0) qRank = kRank;
                if (qRank == -1 && vRank > 0) qRank = vRank;
                if (kRank == -1 && qRank > 0) kRank = qRank;
                if (vRank == -1 && qRank > 0) vRank = qRank;

                List<String> permuteOpsToRemove = new ArrayList<>();
                List<String> permuteVarsToRemove = new ArrayList<>();
                String downstreamPermuteOpName = null;
                String downstreamPermuteOutputVar = null;

                if (qRank == 4 || kRank == 4 || vRank == 4) {
                    String[] qAbsorbed = absorbPermute0213(sd, helper, components.queryVar);
                    if (qAbsorbed != null) {
                        permuteOpsToRemove.add(qAbsorbed[1]);
                        permuteVarsToRemove.add(components.queryVar);
                        components.queryVar = qAbsorbed[0];
                        qSDVar = sd.getVariable(components.queryVar);
                    }

                    // Absorb upstream permute on K.
                    // Two patterns are handled:
                    //   permute(0,2,1,3): BSHD→BHSD (standard multi-head layout swap)
                    //   permute(0,2,3,1): BSHD→BHDS (K transposed for Q@K^T matmul)
                    String[] kAbsorbed = absorbPermute0213(sd, helper, components.keyVar);
                    if (kAbsorbed == null) {
                        kAbsorbed = absorbPermute0231(sd, helper, components.keyVar);
                        if (kAbsorbed != null) {
                            log.debug("[ATTN-R4-MASK] Absorbing K transpose-permute (0,2,3,1): {} -> {}", components.keyVar, kAbsorbed[0]);
                        }
                    }
                    if (kAbsorbed != null) {
                        permuteOpsToRemove.add(kAbsorbed[1]);
                        permuteVarsToRemove.add(components.keyVar);
                        components.keyVar = kAbsorbed[0];
                        kSDVar = sd.getVariable(components.keyVar);
                    }

                    String[] vAbsorbed = absorbPermute0213(sd, helper, vVar);
                    if (vAbsorbed != null) {
                        permuteOpsToRemove.add(vAbsorbed[1]);
                        permuteVarsToRemove.add(vVar);
                        vVar = vAbsorbed[0];
                        vSDVar = sd.getVariable(vVar);
                    }

                    Variable attOutVarInfo = getVariableWithFallback(helper, sd, attentionOutputVar);
                    if (attOutVarInfo != null) {
                        List<String> attUsers = attOutVarInfo.getInputsForOp();
                        if (attUsers != null && attUsers.size() == 1) {
                            SameDiffOp userOp = sd.getOps().get(attUsers.get(0));
                            if (userOp != null && checkPermute0213(userOp)) {
                                List<String> userOutputs = userOp.getOutputsOfOp();
                                if (userOutputs != null && !userOutputs.isEmpty()) {
                                    downstreamPermuteOpName = userOp.getName();
                                    downstreamPermuteOutputVar = userOutputs.get(0);
                                }
                            }
                        }
                    }
                }

                // Create dot_product_attention_v2 with value mask
                SDVariable emptyQueryMask = sd.constant("attn_masked_empty_qmask_" + op.getName(), Nd4j.empty(DataType.FLOAT));

                SDVariable fusedOutput = new DotProductAttentionV2(sd,
                        qSDVar,           // queries
                        vSDVar,           // values
                        kSDVar,           // keys
                        emptyQueryMask,   // queryMask (empty)
                        maskSDVar,        // valueMask - the additive mask
                        null, null, null, null,  // no KV cache, no attention bias
                        components.scaleFactor,
                        0.0,              // no dropout for inference
                        false,            // not causal mask
                        false             // not training
                ).outputVariable();

                // Wire fused output to downstream consumers
                if (downstreamPermuteOutputVar != null) {
                    OptimizationUtils.replaceOpInputsWith(sd, helper, downstreamPermuteOutputVar, fusedOutput.name());
                } else {
                    OptimizationUtils.replaceOpInputsWith(sd, helper, attentionOutputVar, fusedOutput.name());
                }

                // Remove old operations
                OptimizationUtils.removeOp(sd, helper, finalMatmulOp.getName());
                if (downstreamPermuteOpName != null) {
                    OptimizationUtils.removeOp(sd, helper, downstreamPermuteOpName);
                }
                OptimizationUtils.removeOp(sd, helper, softmaxOp.getName());
                OptimizationUtils.removeOp(sd, helper, op.getName()); // The add op
                if (components.scaleOpName != null) {
                    OptimizationUtils.removeOp(sd, helper, components.scaleOpName);
                }
                OptimizationUtils.removeOp(sd, helper, components.qkMatmulOpName);

                // Remove absorbed upstream permute ops
                for (String permuteOpName : permuteOpsToRemove) {
                    OptimizationUtils.removeOp(sd, helper, permuteOpName);
                }

                // Remove intermediate variables
                OptimizationUtils.removeVariable(sd, helper, softmaxOutput);
                OptimizationUtils.removeVariable(sd, helper, addOutput);
                if (components.scaleOutputVar != null) {
                    OptimizationUtils.removeVariable(sd, helper, components.scaleOutputVar);
                }
                OptimizationUtils.removeVariable(sd, helper, components.qkMatmulOutputVar);
                if (downstreamPermuteOutputVar != null) {
                    OptimizationUtils.removeVariable(sd, helper, downstreamPermuteOutputVar);
                }
                OptimizationUtils.removeVariable(sd, helper, attentionOutputVar);
                for (String permuteVarName : permuteVarsToRemove) {
                    OptimizationUtils.removeVariable(sd, helper, permuteVarName);
                }

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
                // Always check for scalar scale value first — constants may not be in the helper
                // and would be silently skipped if we gated this check behind inputVar != null.
                SDVariable sdVar = sd.getVariable(input);
                if (sdVar != null && sdVar.getArr() != null) {
                    INDArray arr = sdVar.getArr();
                    if (arr.isScalar()) {
                        double val = arr.getDouble(0);
                        scaleFactor = isMul ? val : (1.0 / val);
                    }
                }

                // For matmul identification we need the Variable metadata (producer op).
                Variable inputVar = getVariableWithFallback(helper, sd, input);
                if (inputVar == null) continue;

                String inputOpName = inputVar.getOutputOfOp();
                if (inputOpName != null) {
                    SameDiffOp inputOp = sd.getOps().get(inputOpName);
                    if (inputOp != null && (inputOp.getOp() instanceof Mmul || inputOp.getOp() instanceof TensorMmul)) {
                        matmulOutputVar = input;
                    }
                }
            }

            if (matmulOutputVar == null) {
                return null;
            }

            Variable mmOutVar = getVariableWithFallback(helper, sd, matmulOutputVar);
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
                    if (kProducerOp != null &&
                        kProducerOp.getOp() instanceof Transpose) {
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
            return isCausalMaskArray(arr);
        }

        /**
         * Returns true when the mask variable is produced by an additive attention bias subgraph
         * typical of BERT/encoder ONNX exports:
         *
         *   attention_mask (INT64 input)
         *     → Cast (→ FLOAT)
         *     → Sub  (1.0 - mask)
         *     → Mul  (* -3.4e38 or similar large negative)        ← "additive bias"
         *
         * Such a mask has float values {0.0, -3.4028235e38} and is ADDED to QK^T scores
         * before softmax to suppress padding positions.  It is NOT a boolean mask:
         * DotProductAttentionV2's valueMask argument expects non-zero = keep, so fusing
         * this pattern would invert and corrupt all attention heads.
         *
         * Detection: BFS up to 8 hops from maskVar; return true the moment we see a
         * MulOp or ScalarMultiplication whose constant input has absolute value ≥ 1e10.
         */
        private boolean isAdditiveBiasAttnMask(SameDiff sd, OptimizationHelper helper,
                                               String maskVar, ArrayHolder constantArrays) {
            if (maskVar == null) return false;

            Set<String> visited = new HashSet<>();
            List<String> queue = new ArrayList<>();
            queue.add(maskVar);
            int idx = 0;
            final int MAX_HOPS = 8;

            while (idx < queue.size() && idx < MAX_HOPS) {
                String current = queue.get(idx++);
                if (!visited.add(current)) continue;

                Variable v = getVariableWithFallback(helper, sd, current);
                if (v == null) continue;
                String producerName = v.getOutputOfOp();
                if (producerName == null) continue;

                SameDiffOp producerOp = sd.getOps().get(producerName);
                if (producerOp == null || producerOp.getOp() == null) continue;

                boolean isMulOp = producerOp.getOp() instanceof MulOp
                        || producerOp.getOp() instanceof ScalarMultiplication;
                if (isMulOp) {
                    // Check each input: if any is a constant scalar with |value| >= 1e10
                    // (e.g. -3.4e38, -10000.0, -1e9) this is an additive bias mask.
                    List<String> mulInputs = producerOp.getInputsToOp();
                    if (mulInputs != null) {
                        for (String mulIn : mulInputs) {
                            SDVariable mulInVar = sd.getVariable(mulIn);
                            INDArray arr = null;
                            if (mulInVar != null) {
                                arr = mulInVar.getArr();
                                if (arr == null && mulInVar.getVariableType() == VariableType.CONSTANT) {
                                    arr = constantArrays.getArray(mulIn);
                                }
                            }
                            if (arr != null && arr.isScalar()) {
                                double val = arr.getDouble(0);
                                if (Math.abs(val) >= 1e10) {
                                    return true;
                                }
                            }
                        }
                    }
                }

                // Enqueue this op's inputs for further tracing
                List<String> inputs = producerOp.getInputsToOp();
                if (inputs != null) {
                    for (String in : inputs) {
                        if (!visited.contains(in)) {
                            queue.add(in);
                        }
                    }
                }
            }

            return false;
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

    /**
     * Detects LLaMA-style attention blocks and fuses them into DotProductAttentionV2.
     * 
     * LLaMA attention pattern:
     *   input_layernorm → q_proj/k_proj/v_proj → attention_compute → o_proj → output
     *                     ↓                      (Q@K^T→softmax→@V)    ↑
     *                     └────────────────────────────────────────────┘
     * 
     * Where:
     * - q_proj, k_proj, v_proj: Linear projections from normalized input
     * - attention_compute: Q @ K^T → scale → softmax → @ V
     * - o_proj: Output projection
     * 
     * This optimizer detects the complete block and replaces ~6 ops with 1 fused op.
     */
    public static class FuseLLaMAAttentionBlock implements Optimizer {

        private static final Set<Class<? extends DifferentialFunction>> APPLICABLE_OPS = new HashSet<>();
        static {
            APPLICABLE_OPS.add(Mmul.class);
        }

        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return APPLICABLE_OPS;
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof Mmul)) {
                return false;
            }

            // Check if this is an o_proj matmul (output projection)
            if (!isOProjMatmul(op, sd)) {
                return false;
            }

            log.debug("[LLaMA-ATTN] Found o_proj candidate: {} inputs={} outputs={}",
                    op.getName(), op.getInputsToOp(), op.getOutputsOfOp());

            // Get o_proj inputs
            List<String> oProjInputs = op.getInputsToOp();
            if (oProjInputs == null || oProjInputs.size() < 2) {
                log.debug("[LLaMA-ATTN] o_proj {} has insufficient inputs", op.getName());
                return false;
            }

            // The first input should be the attention output (after @V and reshape/permute)
            String attentionOutputVar = oProjInputs.get(0);
            
            // Check if input comes from MultiHeadAttention (ONNX fused attention)
            Variable attnVar = helper.getVariable(attentionOutputVar);
            if (attnVar != null) {
                String attnProducerName = attnVar.getOutputOfOp();
                if (attnProducerName != null) {
                    SameDiffOp attnProducer = sd.getOps().get(attnProducerName);
                    if (attnProducer != null && attnProducer.getOp().getClass().getSimpleName().contains("MultiHeadAttention")) {
                        log.debug("[LLaMA-ATTN] Input to o_proj comes from MultiHeadAttention: {} — already fused", attnProducerName);
                        return false;
                    }
                }
            }
            
            // Trace back to find the attention computation (softmax @ V matmul)
            SameDiffOp[] attentionCompute = findAttentionCompute(sd, helper, attentionOutputVar);
            if (attentionCompute == null) {
                log.debug("[LLaMA-ATTN] Could not find attention compute pattern for {}", op.getName());
                return false;
            }
            
            SameDiffOp softmaxOp = attentionCompute[0];
            SameDiffOp attnMatmulOp = attentionCompute[1];

            log.debug("[LLaMA-ATTN] Found attention compute: softmax={}, matmul={}", 
                     softmaxOp.getName(), attnMatmulOp.getName());

            // Trace back from softmax to find Q @ K^T
            AttentionComponents components = traceQKFromSoftmax(sd, helper, softmaxOp);
            if (components == null) {
                log.debug("[LLaMA-ATTN] Could not trace Q/K from softmax for {}", op.getName());
                return false;
            }

            // Get V from the attention matmul
            List<String> attnMatmulInputs = attnMatmulOp.getInputsToOp();
            if (attnMatmulInputs == null || attnMatmulInputs.size() < 2) {
                log.debug("[LLaMA-ATTN] Attention matmul {} has insufficient inputs", attnMatmulOp.getName());
                return false;
            }
            String vVar = attnMatmulInputs.get(1);

            // Use Q, K, V variables at the attention compute stage (post-reshape/permute).
            // These are in BHSD format (after permute(0,2,1,3)) which is what the attention
            // core operates on. We absorb permutes to get BSHD for the fused op.
            // The projection matmuls and reshape ops remain in the graph.
            SDVariable qSDVar = sd.getVariable(components.queryVar);
            SDVariable kSDVar = sd.getVariable(components.keyVar);
            SDVariable vSDVar = sd.getVariable(vVar);

            if (qSDVar == null || kSDVar == null || vSDVar == null) {
                log.debug("[LLaMA-ATTN] Could not get SDVariables for Q/K/V at attention stage");
                return false;
            }

            // Absorb upstream permute(0,2,1,3) on Q/K/V to recover BSHD format
            // dot_product_attention_v2 expects BSHD, not BHSD
            List<String> permuteOpsToRemove = new ArrayList<>();
            List<String> permuteVarsToRemove = new ArrayList<>();

            String qVarName = components.queryVar;
            String kVarName = components.keyVar;
            String vVarName = vVar;

            int qRank = inferVariableRank(sd, qSDVar);
            if (qRank == 4 || inferVariableRank(sd, kSDVar) == 4 || inferVariableRank(sd, vSDVar) == 4) {
                String[] qAbsorbed = absorbPermute0213(sd, helper, qVarName);
                if (qAbsorbed != null) {
                    permuteOpsToRemove.add(qAbsorbed[1]);
                    permuteVarsToRemove.add(qVarName);
                    qVarName = qAbsorbed[0];
                    qSDVar = sd.getVariable(qVarName);
                }

                String[] kAbsorbed = absorbPermute0213(sd, helper, kVarName);
                if (kAbsorbed == null) {
                    kAbsorbed = absorbPermute0231(sd, helper, kVarName);
                }
                if (kAbsorbed != null) {
                    permuteOpsToRemove.add(kAbsorbed[1]);
                    permuteVarsToRemove.add(kVarName);
                    kVarName = kAbsorbed[0];
                    kSDVar = sd.getVariable(kVarName);
                }

                String[] vAbsorbed = absorbPermute0213(sd, helper, vVarName);
                if (vAbsorbed != null) {
                    permuteOpsToRemove.add(vAbsorbed[1]);
                    permuteVarsToRemove.add(vVarName);
                    vVarName = vAbsorbed[0];
                    vSDVar = sd.getVariable(vVarName);
                }
            }

            // Get the attention output variable — this is what feeds into the
            // downstream permute/reshape → o_proj chain
            String attOutVar = attentionOutputVar;

            // Detect downstream permute(0,2,1,3) on attention output (BHSD→BSHD)
            String downstreamPermuteOpName = null;
            String downstreamPermuteOutputVar = null;
            Variable attOutVarInfo = getVariableWithFallback(helper, sd, attOutVar);
            if (attOutVarInfo != null) {
                List<String> attUsers = attOutVarInfo.getInputsForOp();
                if (attUsers != null && attUsers.size() == 1) {
                    SameDiffOp userOp = sd.getOps().get(attUsers.get(0));
                    if (userOp != null && checkPermute0213(userOp)) {
                        List<String> userOutputs = userOp.getOutputsOfOp();
                        if (userOutputs != null && !userOutputs.isEmpty()) {
                            downstreamPermuteOpName = userOp.getName();
                            downstreamPermuteOutputVar = userOutputs.get(0);
                        }
                    }
                }
            }

            log.info("[LLaMA-ATTN] *** FUSING *** {}: Q={}, K={}, V={}, causal=true",
                    op.getName(), qVarName, kVarName, vVarName);

            try {
                // Create empty masks
                SDVariable emptyQueryMask = sd.constant("llama_attn_empty_qmask_" + op.getName(),
                                                       Nd4j.empty(qSDVar.dataType()));
                SDVariable emptyValueMask = sd.constant("llama_attn_empty_vmask_" + op.getName(),
                                                       Nd4j.empty(vSDVar.dataType()));

                // Create fused attention op with causal masking
                // Q/K/V are in BSHD format (after absorbing upstream permutes)
                SDVariable fusedOutput = new DotProductAttentionV2(sd,
                        qSDVar,              // queries (BSHD)
                        vSDVar,              // values (BSHD)
                        kSDVar,              // keys (BSHD)
                        emptyQueryMask,      // queryMask (empty)
                        emptyValueMask,      // valueMask (empty)
                        null, null, null, null,  // no KV cache, no attention bias
                        components.scaleFactor,  // scale factor (extracted from graph)
                        0.0,                 // no dropout
                        true,                // causal mask (LLaMA uses causal attention)
                        false                // not training
                ).outputVariable();

                // The fused op outputs BSHD. Replace the appropriate downstream variable:
                // - If there was a downstream permute (BHSD→BSHD), replace its output
                // - Otherwise, replace the attention output
                String replaceVar = (downstreamPermuteOutputVar != null) ?
                        downstreamPermuteOutputVar : attOutVar;
                OptimizationUtils.replaceOpInputsWith(sd, helper, replaceVar, fusedOutput.name());

                // Remove attention computation ops (but NOT projection matmuls or o_proj)
                OptimizationUtils.removeOp(sd, helper, attnMatmulOp.getName());
                OptimizationUtils.removeOp(sd, helper, softmaxOp.getName());
                if (components.scaleOpName != null) {
                    OptimizationUtils.removeOp(sd, helper, components.scaleOpName);
                }
                OptimizationUtils.removeOp(sd, helper, components.qkMatmulOpName);

                // Remove absorbed permute ops
                if (downstreamPermuteOpName != null) {
                    OptimizationUtils.removeOp(sd, helper, downstreamPermuteOpName);
                }
                for (String permuteOp : permuteOpsToRemove) {
                    OptimizationUtils.removeOp(sd, helper, permuteOp);
                }

                log.info("[LLaMA-ATTN] Successfully fused attention core for {}", op.getName());
                return true;

            } catch (Exception e) {
                log.warn("[LLaMA-ATTN] Failed to fuse attention block for {}: {}",
                        op.getName(), e.getMessage());
                return false;
            }
        }

        /**
         * Check if this matmul is an o_proj (output projection) by name pattern.
         * Checks op name, weight variable name, and output variable name.
         */
        private boolean isOProjMatmul(SameDiffOp op, SameDiff sd) {
            String opName = op.getName();
            
            // Check op name patterns
            if ((opName.contains("o_proj") && opName.contains("MatMul")) ||
                (opName.contains("attn") && opName.contains("o_proj")) ||
                opName.contains("/attn/o_proj/") ||
                opName.contains(".attn.o_proj.")) {
                log.debug("[LLaMA-ATTN] Matched by op name: {}", opName);
                return true;
            }
            
            // Check weight/input variable names (second input is usually the weight)
            List<String> inputs = op.getInputsToOp();
            if (inputs != null && inputs.size() >= 2) {
                String weightVar = inputs.get(1); // Second input is the weight
                if (weightVar != null && (
                    weightVar.contains("o_proj") || 
                    weightVar.contains("/attn/o_proj/") ||
                    weightVar.contains(".attn.o_proj.")
                )) {
                    log.debug("[LLaMA-ATTN] Matched by weight var: {}", weightVar);
                    return true;
                }
            }
            
            // Check output variable names
            List<String> outputs = op.getOutputsOfOp();
            if (outputs != null && !outputs.isEmpty()) {
                for (String output : outputs) {
                    if (output != null && (
                        output.contains("o_proj") ||
                        output.contains("/attn/o_proj/") ||
                        output.contains(".attn.o_proj.")
                    )) {
                        log.debug("[LLaMA-ATTN] Matched by output var: {}", output);
                        return true;
                    }
                }
            }
            
            return false;
        }
        



        /**
         * Trace back from attention output to find the attention computation.
         * Returns [softmaxOp, attnMatmulOp] or null if not found.
         */
        private SameDiffOp[] findAttentionCompute(SameDiff sd, OptimizationHelper helper, String varName) {
            Variable var = helper.getVariable(varName);
            if (var == null) return null;

            String producerOpName = var.getOutputOfOp();
            if (producerOpName == null) return null;

            SameDiffOp producerOp = sd.getOps().get(producerOpName);
            if (producerOp == null) return null;

            // The attention output may go through reshape/permute before o_proj
            // Keep tracing back until we find the softmax -> matmul pattern
            if (producerOp.getOp() instanceof Reshape || 
                producerOp.getOp() instanceof Permute ||
                producerOp.getOp() instanceof Transpose) {
                // Get the input to this op
                List<String> inputs = producerOp.getInputsToOp();
                if (inputs != null && !inputs.isEmpty()) {
                    return findAttentionCompute(sd, helper, inputs.get(0));
                }
                return null;
            }

            // Check if this is the attention matmul (softmax_output @ V)
            if (producerOp.getOp() instanceof Mmul || producerOp.getOp() instanceof TensorMmul) {
                // Check if first input comes from softmax
                List<String> matmulInputs = producerOp.getInputsToOp();
                if (matmulInputs == null || matmulInputs.size() < 2) return null;

                String firstInput = matmulInputs.get(0);
                Variable firstInputVar = helper.getVariable(firstInput);
                if (firstInputVar == null) return null;

                String firstInputProducerName = firstInputVar.getOutputOfOp();
                if (firstInputProducerName == null) return null;

                SameDiffOp firstInputProducer = sd.getOps().get(firstInputProducerName);
                if (firstInputProducer == null) return null;

                // Check if producer is softmax (directly or through reshape/permute)
                SameDiffOp softmaxOp = findSoftmaxProducer(sd, helper, firstInputProducer);
                if (softmaxOp != null) {
                    return new SameDiffOp[]{softmaxOp, producerOp};
                }
            }

            return null;
        }

        /**
         * Find the softmax op that produces the given op's output.
         * Handles reshape/permute between softmax and matmul.
         */
        private SameDiffOp findSoftmaxProducer(SameDiff sd, OptimizationHelper helper, SameDiffOp op) {
            if (op.getOp() instanceof SoftMax) {
                return op;
            }
            
            // Handle reshape/permute between softmax and matmul
            if (op.getOp() instanceof Reshape || 
                op.getOp() instanceof Permute ||
                op.getOp() instanceof Transpose) {
                List<String> inputs = op.getInputsToOp();
                if (inputs != null && !inputs.isEmpty()) {
                    Variable inputVar = helper.getVariable(inputs.get(0));
                    if (inputVar != null) {
                        String producerName = inputVar.getOutputOfOp();
                        if (producerName != null) {
                            SameDiffOp producer = sd.getOps().get(producerName);
                            if (producer != null) {
                                return findSoftmaxProducer(sd, helper, producer);
                            }
                        }
                    }
                }
            }
            
            return null;
        }

        /**
         * Trace back from softmax to find Q @ K^T pattern.
         */
        private AttentionComponents traceQKFromSoftmax(SameDiff sd, OptimizationHelper helper, 
                                                       SameDiffOp softmaxOp) {
            List<String> softmaxInputs = softmaxOp.getInputsToOp();
            if (softmaxInputs == null || softmaxInputs.isEmpty()) return null;

            String softmaxInput = softmaxInputs.get(0);
            Variable inputVar = helper.getVariable(softmaxInput);
            if (inputVar == null) return null;

            String producerName = inputVar.getOutputOfOp();
            if (producerName == null) return null;

            SameDiffOp producerOp = sd.getOps().get(producerName);
            if (producerOp == null) return null;

            // Check for scaled pattern: scores -> Mul/Div (scale) -> SoftMax
            if (producerOp.getOp() instanceof MulOp || 
                producerOp.getOp() instanceof ScalarMultiplication ||
                producerOp.getOp() instanceof DivOp || 
                producerOp.getOp() instanceof ScalarDivision) {
                
                AttentionComponents components = extractScaleAndTraceQK(sd, helper, producerOp, 
                                                                       softmaxInput, softmaxOp.getName());
                if (components != null) {
                    return components;
                }
            }

            // Direct matmul -> SoftMax (no scale)
            if (producerOp.getOp() instanceof Mmul || producerOp.getOp() instanceof TensorMmul) {
                return extractQKFromMatmul(sd, helper, producerOp, softmaxInput, null, null, 1.0);
            }

            return null;
        }

        /**
         * Extract scale factor and trace back to Q @ K^T matmul.
         */
        private AttentionComponents extractScaleAndTraceQK(SameDiff sd, OptimizationHelper helper,
                                                          SameDiffOp scaleOp, String scaleOutputVar,
                                                          String softmaxOpName) {
            List<String> scaleInputs = scaleOp.getInputsToOp();
            if (scaleInputs == null || scaleInputs.size() < 2) return null;

            boolean isMul = scaleOp.getOp() instanceof MulOp ||
                           scaleOp.getOp() instanceof ScalarMultiplication;
            String matmulOutputVar = null;
            double scaleFactor = 1.0;

            for (String input : scaleInputs) {
                // Always check for scalar scale value first — constants may not be in the helper
                // and would be silently skipped if we gated this check behind inputVar != null.
                SDVariable sdVar = sd.getVariable(input);
                if (sdVar != null && sdVar.getArr() != null) {
                    INDArray arr = sdVar.getArr();
                    if (arr.isScalar() || arr.length() == 1) {
                        double val = arr.getDouble(0);
                        scaleFactor = isMul ? val : (1.0 / val);
                    }
                }

                // For matmul identification we need the Variable metadata (producer op).
                Variable inputVar = getVariableWithFallback(helper, sd, input);
                if (inputVar == null) continue;

                String inputOpName = inputVar.getOutputOfOp();
                if (inputOpName != null) {
                    SameDiffOp inputOp = sd.getOps().get(inputOpName);
                    if (inputOp != null && (inputOp.getOp() instanceof Mmul ||
                                           inputOp.getOp() instanceof TensorMmul)) {
                        matmulOutputVar = input;
                    }
                }
            }

            if (matmulOutputVar == null) return null;

            Variable mmOutVar = getVariableWithFallback(helper, sd, matmulOutputVar);
            if (mmOutVar == null) return null;

            String mmOpName = mmOutVar.getOutputOfOp();
            if (mmOpName == null) return null;

            SameDiffOp mmOp = sd.getOps().get(mmOpName);
            if (mmOp == null) return null;

            return extractQKFromMatmul(sd, helper, mmOp, matmulOutputVar, 
                                      scaleOp.getName(), scaleOutputVar, scaleFactor);
        }

        /**
         * Extract Q and K from Q @ K^T matmul.
         */
        private AttentionComponents extractQKFromMatmul(SameDiff sd, OptimizationHelper helper,
                                                       SameDiffOp matmulOp, String matmulOutputVar,
                                                       String scaleOpName, String scaleOutputVar,
                                                       double scaleFactor) {
            List<String> mmInputs = matmulOp.getInputsToOp();
            if (mmInputs == null || mmInputs.size() < 2) return null;

            String qVar = mmInputs.get(0);
            String kVar = mmInputs.get(1);

            // Check if K goes through transpose (K^T)
            Variable kVariable = helper.getVariable(kVar);
            if (kVariable != null) {
                String kProducerName = kVariable.getOutputOfOp();
                if (kProducerName != null) {
                    SameDiffOp kProducerOp = sd.getOps().get(kProducerName);
                    if (kProducerOp != null &&
                        kProducerOp.getOp() instanceof Transpose) {
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
         * Trace a variable back to its projection output (q_proj, k_proj, or v_proj).
         * Handles reshape and permute ops between projection and attention.
         */
        private String traceBackToProjection(SameDiff sd, OptimizationHelper helper, 
                                            String varName, String projType) {
            Variable var = helper.getVariable(varName);
            if (var == null) return null;

            String producerName = var.getOutputOfOp();
            if (producerName == null) {
                // Check if var name itself contains projection pattern
                if (varName.contains(projType)) {
                    return varName;
                }
                return null;
            }

            SameDiffOp producerOp = sd.getOps().get(producerName);
            if (producerOp == null) return null;

            // Check if this is the projection matmul
            if (producerOp.getOp() instanceof Mmul || producerOp.getOp() instanceof TensorMmul) {
                String producerName2 = producerOp.getName();
                if (producerName2.contains(projType)) {
                    // Get the output of this projection matmul
                    List<String> outputs = producerOp.getOutputsOfOp();
                    if (outputs != null && !outputs.isEmpty()) {
                        return outputs.get(0);
                    }
                }
            }

            // Trace through reshape/permute
            if (producerOp.getOp() instanceof Reshape || 
                producerOp.getOp() instanceof Permute ||
                producerOp.getOp() instanceof Transpose) {
                List<String> inputs = producerOp.getInputsToOp();
                if (inputs != null && !inputs.isEmpty()) {
                    return traceBackToProjection(sd, helper, inputs.get(0), projType);
                }
            }

            // Check if current var name matches projection pattern
            if (varName.contains(projType) && varName.contains("MatMul")) {
                return varName;
            }

            return null;
        }

        /**
         * Verify that Q, K, V projections come from the same source (layernorm output).
         */
        private boolean verifyCommonSource(SameDiff sd, OptimizationHelper helper,
                                          String qProjOutput, String kProjOutput, String vProjOutput) {
            Variable qVar = helper.getVariable(qProjOutput);
            Variable kVar = helper.getVariable(kProjOutput);
            Variable vVar = helper.getVariable(vProjOutput);

            if (qVar == null || kVar == null || vVar == null) return false;

            // Get the producers of the projection outputs
            String qProducer = qVar.getOutputOfOp();
            String kProducer = kVar.getOutputOfOp();
            String vProducer = vVar.getOutputOfOp();

            if (qProducer == null || kProducer == null || vProducer == null) return false;

            SameDiffOp qProjOp = sd.getOps().get(qProducer);
            SameDiffOp kProjOp = sd.getOps().get(kProducer);
            SameDiffOp vProjOp = sd.getOps().get(vProducer);

            if (qProjOp == null || kProjOp == null || vProjOp == null) return false;

            // Get inputs to projection matmuls (should be the normalized input)
            List<String> qProjInputs = qProjOp.getInputsToOp();
            List<String> kProjInputs = kProjOp.getInputsToOp();
            List<String> vProjInputs = vProjOp.getInputsToOp();

            if (qProjInputs == null || kProjInputs == null || vProjInputs == null ||
                qProjInputs.isEmpty() || kProjInputs.isEmpty() || vProjInputs.isEmpty()) {
                return false;
            }

            // The first input to each projection should be the same source
            // (may go through reshape/permute)
            String qSource = traceToCommonSource(sd, helper, qProjInputs.get(0));
            String kSource = traceToCommonSource(sd, helper, kProjInputs.get(0));
            String vSource = traceToCommonSource(sd, helper, vProjInputs.get(0));

            if (qSource == null || kSource == null || vSource == null) return false;

            // Check if sources match
            return qSource.equals(kSource) && kSource.equals(vSource);
        }

        /**
         * Trace a variable back to find its ultimate source (handling reshape/permute).
         */
        private String traceToCommonSource(SameDiff sd, OptimizationHelper helper, String varName) {
            Variable var = helper.getVariable(varName);
            if (var == null) return varName;

            String producerName = var.getOutputOfOp();
            if (producerName == null) return varName;

            SameDiffOp producerOp = sd.getOps().get(producerName);
            if (producerOp == null) return varName;

            // If this is a layernorm/RMS norm output, return it
            if (producerName.contains("layernorm") || 
                producerName.contains("norm") ||
                producerOp.getOp() instanceof MulOp) {  // RMS norm is typically Mul
                return varName;
            }

            // Trace through reshape/permute
            if (producerOp.getOp() instanceof Reshape || 
                producerOp.getOp() instanceof Permute ||
                producerOp.getOp() instanceof Transpose) {
                List<String> inputs = producerOp.getInputsToOp();
                if (inputs != null && !inputs.isEmpty()) {
                    return traceToCommonSource(sd, helper, inputs.get(0));
                }
            }

            return varName;
        }

        /**
         * Remove projection matmul ops (q_proj, k_proj, v_proj).
         */
        private void removeProjectionOps(SameDiff sd, OptimizationHelper helper,
                                        String qProjOutput, String kProjOutput, String vProjOutput) {
            removeProjectionOp(sd, helper, qProjOutput);
            removeProjectionOp(sd, helper, kProjOutput);
            removeProjectionOp(sd, helper, vProjOutput);
        }

        private void removeProjectionOp(SameDiff sd, OptimizationHelper helper, String projOutput) {
            Variable var = helper.getVariable(projOutput);
            if (var == null) return;

            String producerName = var.getOutputOfOp();
            if (producerName == null) return;

            SameDiffOp projOp = sd.getOps().get(producerName);
            if (projOp == null) return;

            if (projOp.getOp() instanceof Mmul || projOp.getOp() instanceof TensorMmul) {
                OptimizationUtils.removeOp(sd, helper, projOp.getName());
            }
        }

        /**
         * Clean up intermediate variables.
         */
        private void cleanupIntermediateVars(SameDiff sd, OptimizationHelper helper,
                                            SameDiffOp oProjOp, SameDiffOp attnMatmulOp,
                                            SameDiffOp softmaxOp, AttentionComponents components,
                                            String attentionOutputVar, String oProjOutputVar) {
            // Remove softmax output
            List<String> softmaxOutputs = softmaxOp.getOutputsOfOp();
            if (softmaxOutputs != null && !softmaxOutputs.isEmpty()) {
                OptimizationUtils.removeVariable(sd, helper, softmaxOutputs.get(0));
            }

            // Remove attention matmul output
            List<String> attnMatmulOutputs = attnMatmulOp.getOutputsOfOp();
            if (attnMatmulOutputs != null && !attnMatmulOutputs.isEmpty()) {
                OptimizationUtils.removeVariable(sd, helper, attnMatmulOutputs.get(0));
            }

            // Remove Q @ K^T matmul output
            if (components.qkMatmulOutputVar != null) {
                OptimizationUtils.removeVariable(sd, helper, components.qkMatmulOutputVar);
            }

            // Remove scale output if present
            if (components.scaleOutputVar != null) {
                OptimizationUtils.removeVariable(sd, helper, components.scaleOutputVar);
            }

            // Remove o_proj output (replaced by fused output)
            OptimizationUtils.removeVariable(sd, helper, oProjOutputVar);
        }
    }
}

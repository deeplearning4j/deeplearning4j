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
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.autodiff.samediff.optimize.OptimizationHelper;
import org.nd4j.autodiff.samediff.optimize.Optimizer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce.Mmul;
import org.nd4j.linalg.api.ops.impl.reduce.TensorMmul;
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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

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
        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            // Look for the final matmul in attention: softmax_weights @ V
            if (!(op.getOp() instanceof Mmul) && !(op.getOp() instanceof TensorMmul)) {
                return false;
            }

            log.debug("FuseManualAttentionPattern: Checking matmul op: {}", op.getName());

            List<String> mmulInputs = op.getInputsToOp();
            if (mmulInputs == null || mmulInputs.size() < 2) {
                return false;
            }

            // Check if first input comes from softmax (attention weights)
            String potentialSoftmaxOutput = mmulInputs.get(0);
            Variable softmaxOutVar = sd.getVariables().get(potentialSoftmaxOutput);
            if (softmaxOutVar == null) return false;

            String softmaxOpName = softmaxOutVar.getOutputOfOp();
            if (softmaxOpName == null) return false;

            SameDiffOp softmaxOp = sd.getOps().get(softmaxOpName);
            if (softmaxOp == null || !(softmaxOp.getOp() instanceof SoftMax)) {
                log.debug("FuseManualAttentionPattern: First input {} is not from softmax (producer: {})",
                    potentialSoftmaxOutput, softmaxOp != null ? softmaxOp.getOp().getClass().getSimpleName() : "null");
                return false;
            }

            log.debug("FuseManualAttentionPattern: Found softmax {} before matmul {}", softmaxOpName, op.getName());
            // Found softmax! Now trace back to find Q @ K^T pattern
            List<String> softmaxInputs = softmaxOp.getInputsToOp();
            if (softmaxInputs == null || softmaxInputs.isEmpty()) {
                return false;
            }

            // The input to softmax could be:
            // 1. Direct matmul output (Q @ K^T)
            // 2. Scaled matmul output (Q @ K^T * scale or Q @ K^T / sqrt(d))
            // 3. Masked and/or scaled output

            String softmaxInput = softmaxInputs.get(0);
            AttentionComponents components = traceAttentionScores(sd, softmaxInput);
            if (components == null) {
                log.debug("FuseManualAttentionPattern: Could not trace attention scores from softmax input: {}", softmaxInput);
                return false;
            }
            log.debug("FuseManualAttentionPattern: Found Q={}, K={}, scale={}",
                components.queryVar, components.keyVar, components.scaleFactor);

            // Get V from the final matmul
            String vVar = mmulInputs.get(1);
            SDVariable vSDVar = sd.getVariable(vVar);
            if (vSDVar == null) return false;

            // Check that intermediate ops are only used once (safe to fuse)
            if (!canSafelyFuse(sd, softmaxOp, components)) {
                log.debug("FuseManualAttentionPattern: canSafelyFuse check failed for op: {}", op.getName());
                return false;
            }
            log.debug("FuseManualAttentionPattern: canSafelyFuse check PASSED for op: {}", op.getName());

            // Get the output of the attention (final matmul output)
            List<String> attentionOutputs = op.getOutputsOfOp();
            if (attentionOutputs == null || attentionOutputs.isEmpty()) {
                return false;
            }
            String attentionOutputVar = attentionOutputs.get(0);

            log.info("Fusing manual attention pattern: Q={}, K={}, V={} into dot_product_attention_v2 (causalMask={})",
                    components.queryVar, components.keyVar, vVar, components.useCausalMask);

            try {
                SDVariable qSDVar = sd.getVariable(components.queryVar);
                SDVariable kSDVar = sd.getVariable(components.keyVar);

                if (qSDVar == null || kSDVar == null) {
                    return false;
                }

                // Create dot_product_attention_v2 op
                // Note: We use keys=K (not K^T) because the op handles transposition internally
                // Create explicit empty mask constants to avoid array access issues with intermediate variables
                SDVariable emptyQueryMask = sd.constant("attn_empty_qmask_" + op.getName(), Nd4j.empty(DataType.FLOAT));
                SDVariable emptyValueMask = sd.constant("attn_empty_vmask_" + op.getName(), Nd4j.empty(DataType.FLOAT));

                SDVariable fusedOutput = new DotProductAttentionV2(sd,
                        qSDVar,           // queries
                        vSDVar,           // values
                        kSDVar,           // keys
                        emptyQueryMask,   // queryMask (empty)
                        emptyValueMask,   // valueMask (empty)
                        components.scaleFactor,  // scale factor
                        0.0,              // no dropout for inference
                        components.useCausalMask,  // use causal mask if detected
                        false             // not training
                ).outputVariable();

                // Replace all uses of the attention output with the fused output
                OptimizationUtils.replaceOpInputsWith(sd, attentionOutputVar, fusedOutput.name());

                // Remove old operations in reverse order
                OptimizationUtils.removeOp(sd, op.getName());  // final matmul
                OptimizationUtils.removeOp(sd, softmaxOp.getName());

                // Remove mask add op if it exists
                if (components.maskOpName != null) {
                    OptimizationUtils.removeOp(sd, components.maskOpName);
                }

                // Remove scale op if it exists
                if (components.scaleOpName != null) {
                    OptimizationUtils.removeOp(sd, components.scaleOpName);
                }

                // Remove Q @ K^T matmul
                OptimizationUtils.removeOp(sd, components.qkMatmulOpName);

                // Remove intermediate variables
                OptimizationUtils.removeVariable(sd, potentialSoftmaxOutput);
                OptimizationUtils.removeVariable(sd, softmaxInput);
                if (components.maskOutputVar != null) {
                    OptimizationUtils.removeVariable(sd, components.maskOutputVar);
                }
                if (components.scaleOutputVar != null) {
                    OptimizationUtils.removeVariable(sd, components.scaleOutputVar);
                }
                OptimizationUtils.removeVariable(sd, components.qkMatmulOutputVar);
                OptimizationUtils.removeVariable(sd, attentionOutputVar);

                return true;
            } catch (Exception e) {
                log.warn("Failed to fuse attention pattern: {}", e.getMessage());
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
        private AttentionComponents traceAttentionScores(SameDiff sd, String varName) {
            Variable v = sd.getVariables().get(varName);
            if (v == null) {
                log.debug("traceAttentionScores: Variable {} is null", varName);
                return null;
            }

            String producerOpName = v.getOutputOfOp();
            if (producerOpName == null) {
                log.debug("traceAttentionScores: Variable {} has no producer op", varName);
                return null;
            }

            SameDiffOp producerOp = sd.getOps().get(producerOpName);
            if (producerOp == null) {
                log.debug("traceAttentionScores: Producer op {} not found", producerOpName);
                return null;
            }

            log.debug("traceAttentionScores: Tracing from {} which is produced by {} (type: {})",
                varName, producerOpName, producerOp.getOp().getClass().getSimpleName());

            // Case 1: Direct matmul output (no scaling, no mask)
            if (producerOp.getOp() instanceof Mmul || producerOp.getOp() instanceof TensorMmul) {
                return extractQKFromMatmul(sd, producerOp, varName, null, null, 1.0);
            }

            // Case 2: Scaled output - look for mul/div by scalar
            if (producerOp.getOp() instanceof MulOp || producerOp.getOp() instanceof ScalarMultiplication) {
                return traceScaledMatmul(sd, producerOp, varName, true);
            }

            if (producerOp.getOp() instanceof DivOp || producerOp.getOp() instanceof ScalarDivision) {
                return traceScaledMatmul(sd, producerOp, varName, false);
            }

            // Case 3: Add operation - could be mask addition before softmax
            if (producerOp.getOp() instanceof AddOp) {
                return traceAttentionWithMask(sd, producerOp, varName);
            }

            return null;
        }

        /**
         * Traces through an add operation that applies a mask to attention scores.
         * Pattern: matmul(Q, K^T) [-> scale] -> add(mask) -> softmax
         */
        private AttentionComponents traceAttentionWithMask(SameDiff sd, SameDiffOp addOp, String addOutputVar) {
            List<String> addInputs = addOp.getInputsToOp();
            if (addInputs == null || addInputs.size() != 2) {
                return null;
            }

            // Find which input comes from matmul/scale (attention scores) and which is the mask
            String scoresVar = null;
            String maskVar = null;
            SameDiffOp scoresProducerOp = null;
            boolean isCausalMask = false;

            for (int i = 0; i < 2; i++) {
                String input = addInputs.get(i);
                Variable inputVar = sd.getVariables().get(input);
                if (inputVar == null) continue;

                String inputOpName = inputVar.getOutputOfOp();
                if (inputOpName != null) {
                    SameDiffOp inputOp = sd.getOps().get(inputOpName);
                    if (inputOp != null) {
                        // Check if this is matmul, scale op, or another pattern
                        if (inputOp.getOp() instanceof Mmul || inputOp.getOp() instanceof TensorMmul ||
                            inputOp.getOp() instanceof MulOp || inputOp.getOp() instanceof ScalarMultiplication ||
                            inputOp.getOp() instanceof DivOp || inputOp.getOp() instanceof ScalarDivision) {
                            scoresVar = input;
                            scoresProducerOp = inputOp;
                            maskVar = addInputs.get(1 - i);
                            break;
                        }
                    }
                }

                // Check if this could be the mask (constant array)
                SDVariable sdVar = sd.getVariable(input);
                if (sdVar != null && sdVar.getArr() != null) {
                    // This looks like a constant mask
                    if (isCausalMaskArray(sdVar.getArr())) {
                        maskVar = input;
                        scoresVar = addInputs.get(1 - i);
                        isCausalMask = true;
                    }
                }
            }

            if (scoresVar == null) {
                return null;
            }

            // Now trace back from scoresVar to find the actual Q @ K^T pattern
            AttentionComponents components = traceAttentionScoresWithoutMask(sd, scoresVar);
            if (components == null) {
                return null;
            }

            // Add mask information to components
            components.maskOpName = addOp.getName();
            components.maskOutputVar = addOutputVar;
            components.maskVar = maskVar;
            components.useCausalMask = isCausalMask;
            components.hasAdditiveMask = true;

            return components;
        }

        /**
         * Traces attention scores without considering mask add operations.
         * Used after we've already identified and removed the mask add layer.
         */
        private AttentionComponents traceAttentionScoresWithoutMask(SameDiff sd, String varName) {
            Variable v = sd.getVariables().get(varName);
            if (v == null) return null;

            String producerOpName = v.getOutputOfOp();
            if (producerOpName == null) return null;

            SameDiffOp producerOp = sd.getOps().get(producerOpName);
            if (producerOp == null) return null;

            // Direct matmul output
            if (producerOp.getOp() instanceof Mmul || producerOp.getOp() instanceof TensorMmul) {
                return extractQKFromMatmul(sd, producerOp, varName, null, null, 1.0);
            }

            // Scaled output
            if (producerOp.getOp() instanceof MulOp || producerOp.getOp() instanceof ScalarMultiplication) {
                return traceScaledMatmul(sd, producerOp, varName, true);
            }

            if (producerOp.getOp() instanceof DivOp || producerOp.getOp() instanceof ScalarDivision) {
                return traceScaledMatmul(sd, producerOp, varName, false);
            }

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
        private AttentionComponents traceScaledMatmul(SameDiff sd, SameDiffOp scaleOp,
                                                       String scaleOutputVar, boolean isMul) {
            List<String> scaleInputs = scaleOp.getInputsToOp();
            if (scaleInputs == null || scaleInputs.size() < 2) {
                return null;
            }

            // Find which input is the matmul output and which is the scale factor
            String matmulOutputVar = null;
            double scaleFactor = 1.0;

            for (String input : scaleInputs) {
                Variable inputVar = sd.getVariables().get(input);
                if (inputVar == null) continue;

                String inputOpName = inputVar.getOutputOfOp();
                if (inputOpName != null) {
                    SameDiffOp inputOp = sd.getOps().get(inputOpName);
                    if (inputOp != null && (inputOp.getOp() instanceof Mmul || inputOp.getOp() instanceof TensorMmul)) {
                        matmulOutputVar = input;
                    }
                }

                // Check if this is a constant scalar (scale factor)
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

            Variable mmOutVar = sd.getVariables().get(matmulOutputVar);
            if (mmOutVar == null) return null;

            String mmOpName = mmOutVar.getOutputOfOp();
            if (mmOpName == null) return null;

            SameDiffOp mmOp = sd.getOps().get(mmOpName);
            if (mmOp == null) return null;

            return extractQKFromMatmul(sd, mmOp, matmulOutputVar, scaleOp.getName(), scaleOutputVar, scaleFactor);
        }

        /**
         * Extracts Q and K variables from a matmul operation that computes Q @ K^T.
         */
        private AttentionComponents extractQKFromMatmul(SameDiff sd, SameDiffOp matmulOp,
                                                         String matmulOutputVar,
                                                         String scaleOpName, String scaleOutputVar,
                                                         double scaleFactor) {
            List<String> mmInputs = matmulOp.getInputsToOp();
            if (mmInputs == null || mmInputs.size() < 2) {
                return null;
            }

            String qVar = mmInputs.get(0);
            String kVar = mmInputs.get(1);

            // Check if K is transposed (either via Mmul transpose flag or explicit Transpose op)
            boolean kTransposed = false;

            if (matmulOp.getOp() instanceof Mmul) {
                Mmul mmul = (Mmul) matmulOp.getOp();
                // Check transpose flags via iArguments (index 1 is transposeB)
                if (mmul.numIArguments() > 1 && mmul.getIArgument(1) > 0) {
                    kTransposed = true;
                }
            }

            // Check for explicit transpose on K
            Variable kVariable = sd.getVariables().get(kVar);
            if (kVariable != null) {
                String kProducerName = kVariable.getOutputOfOp();
                if (kProducerName != null) {
                    SameDiffOp kProducerOp = sd.getOps().get(kProducerName);
                    if (kProducerOp != null && kProducerOp.getOp() instanceof Transpose) {
                        // K comes from a transpose - get the original K
                        List<String> transposeInputs = kProducerOp.getInputsToOp();
                        if (transposeInputs != null && !transposeInputs.isEmpty()) {
                            kVar = transposeInputs.get(0);
                            kTransposed = true;
                        }
                    }
                }
            }

            // For attention, K should be transposed (Q @ K^T)
            if (!kTransposed) {
                // This might not be an attention pattern
                log.debug("Matmul found but K is not transposed - may not be attention pattern");
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

        /**
         * Checks if intermediate operations can be safely removed (only used in this attention pattern).
         */
        private boolean canSafelyFuse(SameDiff sd, SameDiffOp softmaxOp, AttentionComponents components) {
            // Check softmax output is only used by one op (the final matmul)
            List<String> softmaxOutputs = softmaxOp.getOutputsOfOp();
            if (softmaxOutputs == null || softmaxOutputs.isEmpty()) {
                log.debug("canSafelyFuse: softmax has no outputs");
                return false;
            }

            Variable softmaxOutVar = sd.getVariables().get(softmaxOutputs.get(0));
            if (softmaxOutVar == null) {
                log.debug("canSafelyFuse: softmax output var is null");
                return false;
            }

            List<String> softmaxUsers = softmaxOutVar.getInputsForOp();
            if (softmaxUsers == null || softmaxUsers.size() != 1) {
                log.debug("canSafelyFuse: softmax output {} has {} users (expected 1): {}",
                    softmaxOutputs.get(0), softmaxUsers != null ? softmaxUsers.size() : 0, softmaxUsers);
                return false;
            }

            // Check Q @ K^T output is only used once
            Variable qkOutVar = sd.getVariables().get(components.qkMatmulOutputVar);
            if (qkOutVar == null) {
                log.debug("canSafelyFuse: Q@K^T output var {} is null", components.qkMatmulOutputVar);
                return false;
            }

            List<String> qkUsers = qkOutVar.getInputsForOp();
            if (qkUsers == null || qkUsers.size() != 1) {
                log.debug("canSafelyFuse: Q@K^T output {} has {} users (expected 1): {}",
                    components.qkMatmulOutputVar, qkUsers != null ? qkUsers.size() : 0, qkUsers);
                return false;
            }

            log.debug("canSafelyFuse: All checks passed");
            return true;
        }
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
            Variable attOutVar = sd.getVariables().get(attentionOutput);
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
            log.debug("Found potential attention+projection fusion opportunity at {}", op.getName());

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
                Variable v = sd.getVariables().get(input);
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
            Variable addOutVar = sd.getVariables().get(addOutput);
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
            Variable softmaxOutVar = sd.getVariables().get(softmaxOutput);
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
            AttentionComponents components = traceQKFromScores(sd, scoresVar);
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

            log.info("Fusing causal masked attention: Q={}, K={}, V={} into dot_product_attention_v2",
                    components.queryVar, components.keyVar, vVar);

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
                OptimizationUtils.replaceOpInputsWith(sd, attentionOutputVar, fusedOutput.name());

                // Remove old operations
                OptimizationUtils.removeOp(sd, finalMatmulOp.getName());
                OptimizationUtils.removeOp(sd, softmaxOp.getName());
                OptimizationUtils.removeOp(sd, op.getName()); // The add op
                if (components.scaleOpName != null) {
                    OptimizationUtils.removeOp(sd, components.scaleOpName);
                }
                OptimizationUtils.removeOp(sd, components.qkMatmulOpName);

                // Remove intermediate variables
                OptimizationUtils.removeVariable(sd, softmaxOutput);
                OptimizationUtils.removeVariable(sd, addOutput);
                if (components.scaleOutputVar != null) {
                    OptimizationUtils.removeVariable(sd, components.scaleOutputVar);
                }
                OptimizationUtils.removeVariable(sd, components.qkMatmulOutputVar);
                OptimizationUtils.removeVariable(sd, attentionOutputVar);

                return true;
            } catch (Exception e) {
                log.warn("Failed to fuse causal masked attention: {}", e.getMessage());
                return false;
            }
        }

        /**
         * Traces back from attention scores to find Q and K variables.
         */
        private AttentionComponents traceQKFromScores(SameDiff sd, String scoresVar) {
            Variable v = sd.getVariables().get(scoresVar);
            if (v == null) return null;

            String producerOpName = v.getOutputOfOp();
            if (producerOpName == null) return null;

            SameDiffOp producerOp = sd.getOps().get(producerOpName);
            if (producerOp == null) return null;

            // Direct matmul
            if (producerOp.getOp() instanceof Mmul || producerOp.getOp() instanceof TensorMmul) {
                return extractQKFromMatmul(sd, producerOp, scoresVar, null, null, 1.0);
            }

            // Scaled - trace through scale op
            if (producerOp.getOp() instanceof MulOp || producerOp.getOp() instanceof ScalarMultiplication ||
                producerOp.getOp() instanceof DivOp || producerOp.getOp() instanceof ScalarDivision) {
                return traceScaledQK(sd, producerOp, scoresVar);
            }

            return null;
        }

        /**
         * Traces through a scale operation to find Q @ K^T.
         */
        private AttentionComponents traceScaledQK(SameDiff sd, SameDiffOp scaleOp, String scaleOutputVar) {
            List<String> scaleInputs = scaleOp.getInputsToOp();
            if (scaleInputs == null || scaleInputs.size() < 2) {
                return null;
            }

            boolean isMul = scaleOp.getOp() instanceof MulOp || scaleOp.getOp() instanceof ScalarMultiplication;
            String matmulOutputVar = null;
            double scaleFactor = 1.0;

            for (String input : scaleInputs) {
                Variable inputVar = sd.getVariables().get(input);
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

            Variable mmOutVar = sd.getVariables().get(matmulOutputVar);
            if (mmOutVar == null) return null;

            String mmOpName = mmOutVar.getOutputOfOp();
            if (mmOpName == null) return null;

            SameDiffOp mmOp = sd.getOps().get(mmOpName);
            if (mmOp == null) return null;

            return extractQKFromMatmul(sd, mmOp, matmulOutputVar, scaleOp.getName(), scaleOutputVar, scaleFactor);
        }

        /**
         * Extracts Q and K from a matmul operation.
         */
        private AttentionComponents extractQKFromMatmul(SameDiff sd, SameDiffOp matmulOp,
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
            Variable kVariable = sd.getVariables().get(kVar);
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
                Variable v = sd.getVariables().get(input);
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
            Variable addOutVar = sd.getVariables().get(addOutput);
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
            Variable softmaxOutVar = sd.getVariables().get(softmaxOutput);
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
            AttentionComponents components = traceQKFromScoresInternal(sd, scoresVar);
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

            log.info("Fusing masked attention: Q={}, K={}, V={}, mask={} into dot_product_attention_v2",
                    components.queryVar, components.keyVar, vVar, maskVar);

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
                OptimizationUtils.replaceOpInputsWith(sd, attentionOutputVar, fusedOutput.name());

                // Remove old operations
                OptimizationUtils.removeOp(sd, finalMatmulOp.getName());
                OptimizationUtils.removeOp(sd, softmaxOp.getName());
                OptimizationUtils.removeOp(sd, op.getName()); // The add op
                if (components.scaleOpName != null) {
                    OptimizationUtils.removeOp(sd, components.scaleOpName);
                }
                OptimizationUtils.removeOp(sd, components.qkMatmulOpName);

                // Remove intermediate variables
                OptimizationUtils.removeVariable(sd, softmaxOutput);
                OptimizationUtils.removeVariable(sd, addOutput);
                if (components.scaleOutputVar != null) {
                    OptimizationUtils.removeVariable(sd, components.scaleOutputVar);
                }
                OptimizationUtils.removeVariable(sd, components.qkMatmulOutputVar);
                OptimizationUtils.removeVariable(sd, attentionOutputVar);

                return true;
            } catch (Exception e) {
                log.warn("Failed to fuse masked attention: {}", e.getMessage());
                return false;
            }
        }

        private AttentionComponents traceQKFromScoresInternal(SameDiff sd, String scoresVar) {
            Variable v = sd.getVariables().get(scoresVar);
            if (v == null) return null;

            String producerOpName = v.getOutputOfOp();
            if (producerOpName == null) return null;

            SameDiffOp producerOp = sd.getOps().get(producerOpName);
            if (producerOp == null) return null;

            if (producerOp.getOp() instanceof Mmul || producerOp.getOp() instanceof TensorMmul) {
                return extractQKFromMatmulInternal(sd, producerOp, scoresVar, null, null, 1.0);
            }

            if (producerOp.getOp() instanceof MulOp || producerOp.getOp() instanceof ScalarMultiplication ||
                producerOp.getOp() instanceof DivOp || producerOp.getOp() instanceof ScalarDivision) {
                return traceScaledQKInternal(sd, producerOp, scoresVar);
            }

            return null;
        }

        private AttentionComponents traceScaledQKInternal(SameDiff sd, SameDiffOp scaleOp, String scaleOutputVar) {
            List<String> scaleInputs = scaleOp.getInputsToOp();
            if (scaleInputs == null || scaleInputs.size() < 2) {
                return null;
            }

            boolean isMul = scaleOp.getOp() instanceof MulOp || scaleOp.getOp() instanceof ScalarMultiplication;
            String matmulOutputVar = null;
            double scaleFactor = 1.0;

            for (String input : scaleInputs) {
                Variable inputVar = sd.getVariables().get(input);
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

            Variable mmOutVar = sd.getVariables().get(matmulOutputVar);
            if (mmOutVar == null) return null;

            String mmOpName = mmOutVar.getOutputOfOp();
            if (mmOpName == null) return null;

            SameDiffOp mmOp = sd.getOps().get(mmOpName);
            if (mmOp == null) return null;

            return extractQKFromMatmulInternal(sd, mmOp, matmulOutputVar, scaleOp.getName(), scaleOutputVar, scaleFactor);
        }

        private AttentionComponents extractQKFromMatmulInternal(SameDiff sd, SameDiffOp matmulOp,
                                                                  String matmulOutputVar,
                                                                  String scaleOpName, String scaleOutputVar,
                                                                  double scaleFactor) {
            List<String> mmInputs = matmulOp.getInputsToOp();
            if (mmInputs == null || mmInputs.size() < 2) {
                return null;
            }

            String qVar = mmInputs.get(0);
            String kVar = mmInputs.get(1);

            Variable kVariable = sd.getVariables().get(kVar);
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

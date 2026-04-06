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
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.ArrayHolder;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.autodiff.samediff.optimize.OptimizationHelper;
import org.nd4j.autodiff.samediff.optimize.Optimizer;
import org.nd4j.linalg.api.ops.impl.reduce.Mmul;
import org.nd4j.linalg.api.ops.impl.reduce.TensorMmul;
import org.nd4j.linalg.api.ops.impl.transforms.custom.FusedGemmSwiglu;
import org.nd4j.linalg.api.ops.impl.transforms.custom.SwishMul;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MulOp;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Swish;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Sigmoid;

import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * Activation function fusion optimizations inspired by Luminal's RowSwishMul.
 * These optimizations fuse common LLM activation patterns for better performance.
 *
 * Supported fusions:
 * - swish(x) * y -> swiGLU pattern (common in LLaMA, Mistral)
 * - sigmoid(x) * x -> swish(x) (SiLU activation)
 *
 * Note: Full SwiGLU (swish(x @ W_gate) * (x @ W_up)) requires the fused op to be
 * implemented in libnd4j. This optimization prepares for that by detecting the pattern.
 */
@Slf4j
public class ActivationFusionOptimizations extends BaseOptimizerSet {

    /**
     * Fuses sigmoid(x) * x -> swish(x)
     *
     * This pattern is the definition of the Swish/SiLU activation:
     * swish(x) = x * sigmoid(x)
     *
     * When we see mul(sigmoid(x), x) or mul(x, sigmoid(x)), we can replace
     * it with the fused swish operation which is more efficient.
     */
    public static class FuseSigmoidMulToSwish implements Optimizer {
        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof MulOp)) {
                return false;
            }

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.size() != 2) {
                return false;
            }

            // Find sigmoid input and the raw input
            String sigmoidOutputVar = null;
            String rawInputVar = null;
            SameDiffOp sigmoidOp = null;

            for (int i = 0; i < 2; i++) {
                String inputVar = inputs.get(i);
                Variable v = helper != null ? helper.getVariable(inputVar) : sd.getVariables().get(inputVar);
                if (v == null) continue;

                String producerOpName = v.getOutputOfOp();
                if (producerOpName == null) continue;

                SameDiffOp producerOp = sd.getOps().get(producerOpName);
                if (producerOp != null && producerOp.getOp() instanceof Sigmoid) {
                    sigmoidOutputVar = inputVar;
                    sigmoidOp = producerOp;
                    rawInputVar = inputs.get(1 - i);
                    break;
                }
            }

            if (sigmoidOp == null || rawInputVar == null) {
                return false;
            }

            // Check that sigmoid input matches the raw input (sigmoid(x) * x pattern)
            List<String> sigmoidInputs = sigmoidOp.getInputsToOp();
            if (sigmoidInputs == null || sigmoidInputs.isEmpty()) {
                return false;
            }

            String sigmoidInput = sigmoidInputs.get(0);
            // Compare through cast/identity/reshape ops — mixed-precision models
            // insert casts (e.g., FP16→FP32 before sigmoid) that break exact matching
            String strippedSigmoidInput = stripTrivialOps(sd, helper, sigmoidInput);
            String strippedRawInput = stripTrivialOps(sd, helper, rawInputVar);
            if (!strippedSigmoidInput.equals(strippedRawInput)) {
                // This is sigmoid(a) * b where a != b, not the swish pattern
                return false;
            }

            // Check that sigmoid output is only used by this mul
            Variable sigmoidOutVariable = helper != null ?
                helper.getVariable(sigmoidOutputVar) : sd.getVariables().get(sigmoidOutputVar);
            if (sigmoidOutVariable == null) return false;

            List<String> sigmoidOutputUsers = sigmoidOutVariable.getInputsForOp();
            if (sigmoidOutputUsers == null || sigmoidOutputUsers.size() != 1) {
                return false;
            }

            // Get the mul output
            List<String> outputs = op.getOutputsOfOp();
            if (outputs == null || outputs.isEmpty()) {
                return false;
            }
            String mulOutputVar = outputs.get(0);

            log.info("Fusing sigmoid({}) * {} -> swish({})", rawInputVar, rawInputVar, rawInputVar);

            try {
                // Create swish operation
                SDVariable xVar = sd.getVariable(rawInputVar);
                if (xVar == null) return false;

                SDVariable swishOutput = sd.nn().swish(xVar);

                // Replace all uses of the mul output with swish output
                OptimizationUtils.replaceOpInputsWith(sd, helper, mulOutputVar, swishOutput.name());

                // Remove the old mul and sigmoid operations
                OptimizationUtils.removeOp(sd, helper, op.getName());
                OptimizationUtils.removeOp(sd, helper, sigmoidOp.getName());

                // Remove old variables
                OptimizationUtils.removeVariable(sd, helper, sigmoidOutputVar);
                OptimizationUtils.removeVariable(sd, helper, mulOutputVar);

                return true;
            } catch (Exception e) {
                log.warn("Failed to fuse sigmoid*x to swish: {}", e.getMessage());
                return false;
            }
        }
    }

    /**
     * Fuses SwiGLU pattern: swish(x) * y -> swish_mul(x, y)
     *
     * SwiGLU is used in modern LLMs like LLaMA:
     * output = swish(x @ W_gate) * (x @ W_up)
     *
     * This optimizer detects swish(x) * y patterns and fuses them into
     * a single swish_mul op for better performance.
     */
    public static class FuseSwiGLUPattern implements Optimizer {
        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof MulOp)) {
                return false;
            }

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.size() != 2) {
                return false;
            }

            // Find swish input
            String swishOutputVar = null;
            String otherInputVar = null;
            SameDiffOp swishOp = null;

            for (int i = 0; i < 2; i++) {
                String inputVar = inputs.get(i);
                Variable v = helper != null ? helper.getVariable(inputVar) : sd.getVariables().get(inputVar);
                if (v == null) continue;

                String producerOpName = v.getOutputOfOp();
                if (producerOpName == null) continue;

                SameDiffOp producerOp = sd.getOps().get(producerOpName);
                if (producerOp != null && producerOp.getOp() instanceof Swish) {
                    swishOutputVar = inputVar;
                    swishOp = producerOp;
                    otherInputVar = inputs.get(1 - i);
                    break;
                }
            }

            if (swishOp == null) {
                // No swish found - this is not a SwiGLU pattern
                return false;
            }

            // Check swish output is only used by this mul
            Variable swishOutVariable = helper != null ?
                helper.getVariable(swishOutputVar) : sd.getVariables().get(swishOutputVar);
            if (swishOutVariable == null) return false;

            List<String> swishOutputUsers = swishOutVariable.getInputsForOp();
            if (swishOutputUsers == null || swishOutputUsers.size() != 1) {
                return false;
            }

            // Get the swish input (x in swish(x))
            List<String> swishInputs = swishOp.getInputsToOp();
            if (swishInputs == null || swishInputs.isEmpty()) {
                return false;
            }
            String swishInputVar = swishInputs.get(0);

            // Get the mul output
            List<String> outputs = op.getOutputsOfOp();
            if (outputs == null || outputs.isEmpty()) {
                return false;
            }
            String mulOutputVar = outputs.get(0);

            log.info("Fusing SwiGLU pattern: swish({}) * {} -> swish_mul({}, {})",
                swishInputVar, otherInputVar, swishInputVar, otherInputVar);

            try {
                // Create fused swish_mul operation
                SDVariable xVar = sd.getVariable(swishInputVar);
                SDVariable yVar = sd.getVariable(otherInputVar);
                if (xVar == null || yVar == null) return false;

                // Create the fused SwishMul op - calling outputVariable() registers it with SameDiff
                SDVariable swishMulOutput = new SwishMul(sd, xVar, yVar).outputVariable();

                // Replace all uses of the mul output with swish_mul output
                OptimizationUtils.replaceOpInputsWith(sd, helper, mulOutputVar, swishMulOutput.name());

                // Remove the old mul and swish operations
                OptimizationUtils.removeOp(sd, helper, op.getName());
                OptimizationUtils.removeOp(sd, helper, swishOp.getName());

                // Remove old variables
                OptimizationUtils.removeVariable(sd, helper, swishOutputVar);
                OptimizationUtils.removeVariable(sd, helper, mulOutputVar);

                return true;
            } catch (Exception e) {
                log.warn("Failed to fuse SwiGLU pattern: {}", e.getMessage());
                return false;
            }
        }
    }

    /**
     * Fuses GatedMLP pattern: swish_mul(matmul(x, W_gate), matmul(x, W_up)) -> fused_gemm_swiglu(x, W_gate, W_up)
     *
     * This pattern occurs in LLaMA/Mistral feed-forward blocks after the earlier passes
     * have already fused sigmoid(x)*x -> swish and swish(x)*y -> swish_mul.
     * The resulting graph has:
     *   swish_mul(matmul(x, W_gate), matmul(x, W_up))
     *
     * where both matmuls share the same first input x. We fuse this into a single
     * fused_gemm_swiglu op that computes both GEMMs and the SwiGLU activation in one kernel.
     *
     * FusedGemmSwiglu semantics: output = (x @ W_gate) * silu(x @ W_up)
     * SwishMul semantics: silu(input0) * input1
     * Therefore: SwishMul.input0 = matmul(x, W_up) [silu path], SwishMul.input1 = matmul(x, W_gate) [direct path]
     */
    public static class FuseGatedMLPPattern implements Optimizer {

        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return Collections.singleton(SwishMul.class);
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof SwishMul)) {
                return false;
            }

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.size() != 2) {
                return false;
            }

            // SwishMul(input0, input1) = silu(input0) * input1
            // input0 is the silu path (produces x @ W_up in the FusedGemmSwiglu formula)
            // input1 is the direct multiply path (produces x @ W_gate in the FusedGemmSwiglu formula)
            String siluPathVar = inputs.get(0);   // output of matmul(x, W_up)
            String directPathVar = inputs.get(1);  // output of matmul(x, W_gate)

            // Strip through trivial ops (cast, identity, reshape) to find the actual matmul outputs
            String strippedSiluPath = stripTrivialOps(sd, helper, siluPathVar);
            String strippedDirectPath = stripTrivialOps(sd, helper, directPathVar);

            // Find the producing ops for both paths
            SameDiffOp siluMatmulOp = getProducingMatmulOp(sd, helper, strippedSiluPath);
            SameDiffOp directMatmulOp = getProducingMatmulOp(sd, helper, strippedDirectPath);

            if (siluMatmulOp == null || directMatmulOp == null) {
                return false;
            }

            // Both matmuls must have exactly 2 inputs: (x, W)
            List<String> siluMatmulInputs = siluMatmulOp.getInputsToOp();
            List<String> directMatmulInputs = directMatmulOp.getInputsToOp();
            if (siluMatmulInputs == null || siluMatmulInputs.size() != 2) {
                return false;
            }
            if (directMatmulInputs == null || directMatmulInputs.size() != 2) {
                return false;
            }

            // Verify both matmuls share the same first input (x), stripping through trivial ops
            String siluMatmulX = stripTrivialOps(sd, helper, siluMatmulInputs.get(0));
            String directMatmulX = stripTrivialOps(sd, helper, directMatmulInputs.get(0));
            if (!siluMatmulX.equals(directMatmulX)) {
                return false;
            }

            // Extract weight variables
            // siluMatmulOp computes x @ W_up (silu path), directMatmulOp computes x @ W_gate (direct path)
            String wUpVar = siluMatmulInputs.get(1);
            String wGateVar = directMatmulInputs.get(1);

            // The shared input x — use the actual (non-stripped) input from one of the matmuls
            String xVar = siluMatmulInputs.get(0);

            // Check that the matmul outputs have no other consumers (safe to remove)
            boolean siluMatmulExclusive = isExclusivelyConsumed(sd, helper, strippedSiluPath, op.getName());
            boolean directMatmulExclusive = isExclusivelyConsumed(sd, helper, strippedDirectPath, op.getName());

            // Get the SwishMul output variable
            List<String> swishMulOutputs = op.getOutputsOfOp();
            if (swishMulOutputs == null || swishMulOutputs.isEmpty()) {
                return false;
            }
            String swishMulOutputVar = swishMulOutputs.get(0);

            log.info("Fusing GatedMLP: swish_mul(mmul({}, {}), mmul({}, {})) -> fused_gemm_swiglu({}, {}, {})",
                    xVar, wUpVar, xVar, wGateVar, xVar, wGateVar, wUpVar);

            try {
                SDVariable x = sd.getVariable(xVar);
                SDVariable wGate = sd.getVariable(wGateVar);
                SDVariable wUp = sd.getVariable(wUpVar);
                if (x == null || wGate == null || wUp == null) {
                    return false;
                }

                // Create fused_gemm_swiglu: output = (x @ wGate) * silu(x @ wUp)
                SDVariable fusedOutput = new FusedGemmSwiglu(sd, x, wGate, wUp).outputVariable();

                // Replace all uses of SwishMul output with fused output
                OptimizationUtils.replaceOpInputsWith(sd, helper, swishMulOutputVar, fusedOutput.name());

                // Remove the SwishMul op
                OptimizationUtils.removeOp(sd, helper, op.getName());
                OptimizationUtils.removeVariable(sd, helper, swishMulOutputVar);

                // Remove matmul ops if they have no other consumers
                if (siluMatmulExclusive) {
                    OptimizationUtils.removeOp(sd, helper, siluMatmulOp.getName());
                    OptimizationUtils.removeVariable(sd, helper, strippedSiluPath);
                    // Also remove intermediate trivial op variables between matmul and SwishMul
                    if (!strippedSiluPath.equals(siluPathVar)) {
                        OptimizationUtils.removeVariable(sd, helper, siluPathVar);
                    }
                }
                if (directMatmulExclusive) {
                    OptimizationUtils.removeOp(sd, helper, directMatmulOp.getName());
                    OptimizationUtils.removeVariable(sd, helper, strippedDirectPath);
                    if (!strippedDirectPath.equals(directPathVar)) {
                        OptimizationUtils.removeVariable(sd, helper, directPathVar);
                    }
                }

                return true;
            } catch (Exception e) {
                log.warn("Failed to fuse GatedMLP pattern: {}", e.getMessage());
                return false;
            }
        }

        /**
         * Finds the producing op for a variable if it is a matmul (Mmul or TensorMmul).
         * Returns null if the variable is not produced by a matmul.
         */
        private SameDiffOp getProducingMatmulOp(SameDiff sd, OptimizationHelper helper, String varName) {
            Variable v = helper != null ? helper.getVariable(varName) : sd.getVariables().get(varName);
            if (v == null) return null;

            String producerOpName = v.getOutputOfOp();
            if (producerOpName == null) return null;

            SameDiffOp producerOp = sd.getOps().get(producerOpName);
            if (producerOp == null || producerOp.getOp() == null) return null;

            if (producerOp.getOp() instanceof Mmul || producerOp.getOp() instanceof TensorMmul) {
                return producerOp;
            }
            return null;
        }

        /**
         * Checks if the given variable is exclusively consumed by the specified op
         * (i.e., no other ops use this variable as input). This determines whether
         * the producing op can be safely removed.
         */
        private boolean isExclusivelyConsumed(SameDiff sd, OptimizationHelper helper, String varName, String consumerOpName) {
            Variable v = helper != null ? helper.getVariable(varName) : sd.getVariables().get(varName);
            if (v == null) return false;

            List<String> consumers = v.getInputsForOp();
            if (consumers == null) return true;

            // The variable should only be consumed by the SwishMul op (or trivial ops leading to it)
            for (String consumer : consumers) {
                if (!consumer.equals(consumerOpName)) {
                    return false;
                }
            }
            return true;
        }
    }

    /**
     * Strips through cast, identity, expand_dims, squeeze, reshape ops to find the
     * underlying variable. Used to compare variables across mixed-precision cast boundaries.
     */
    private static String stripTrivialOps(SameDiff sd, OptimizationHelper helper, String varName) {
        String current = varName;
        for (int i = 0; i < 8; i++) {
            Variable v = helper != null ? helper.getVariable(current) : sd.getVariables().get(current);
            if (v == null) break;
            String producerOpName = v.getOutputOfOp();
            if (producerOpName == null) break;
            SameDiffOp p = sd.getOps().get(producerOpName);
            if (p == null || p.getOp() == null || p.getInputsToOp() == null || p.getInputsToOp().isEmpty()) break;
            String opName = p.getOp().opName();
            if (opName == null) break;
            opName = opName.toLowerCase();
            if ("cast".equals(opName) || "identity".equals(opName) || "expand_dims".equals(opName)
                    || "squeeze".equals(opName) || "reshape".equals(opName)) {
                current = p.getInputsToOp().get(0);
            } else {
                break;
            }
        }
        return current;
    }
}

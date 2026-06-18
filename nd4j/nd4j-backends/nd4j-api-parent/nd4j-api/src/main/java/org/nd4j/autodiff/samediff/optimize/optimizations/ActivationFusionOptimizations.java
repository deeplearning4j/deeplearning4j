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
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.linalg.api.ops.BaseReduceOp;
import org.nd4j.linalg.api.ops.impl.reduce.same.Max;
import org.nd4j.linalg.api.ops.impl.reduce.same.Sum;
import org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax;
import org.nd4j.linalg.api.ops.impl.transforms.custom.SwishMul;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.DivOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MulOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.SubOp;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Exp;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Swish;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Sigmoid;

import java.util.List;

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
                String swishName = swishOutput.name();

                // Replace all uses of the mul output with swish output
                OptimizationUtils.replaceOpInputsWith(sd, helper, mulOutputVar, swishName);

                // Temporarily remove mulOutputVar from graph outputs so removeOp/
                // removeVariable guards don't refuse deletion. Without this, when
                // mulOutputVar is a graph output, removeVariable is refused and the
                // subsequent renameVariable fails with a Preconditions exception.
                List<String> graphOutputs = sd.outputs();
                boolean wasOutput = graphOutputs != null && graphOutputs.remove(mulOutputVar);

                // Remove the old mul and sigmoid operations
                OptimizationUtils.removeOp(sd, helper, op.getName());
                OptimizationUtils.removeOp(sd, helper, sigmoidOp.getName());

                // Remove old variables
                OptimizationUtils.removeVariable(sd, helper, sigmoidOutputVar);
                OptimizationUtils.removeVariable(sd, helper, mulOutputVar);

                // Rename fused output to match original output name
                if (!swishName.equals(mulOutputVar)) {
                    sd.renameVariable(swishName, mulOutputVar);
                }
                if (wasOutput) {
                    graphOutputs.add(mulOutputVar);
                }

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
                String swishMulName = swishMulOutput.name();

                // Replace all uses of the mul output with swish_mul output
                OptimizationUtils.replaceOpInputsWith(sd, helper, mulOutputVar, swishMulName);

                // Temporarily remove mulOutputVar from graph outputs so removeOp/
                // removeVariable guards don't refuse deletion. Without this, when
                // mulOutputVar is a graph output, removeVariable is refused and the
                // subsequent renameVariable fails with a Preconditions exception.
                List<String> graphOutputs = sd.outputs();
                boolean wasOutput = graphOutputs != null && graphOutputs.remove(mulOutputVar);

                // Remove the old mul and swish operations
                OptimizationUtils.removeOp(sd, helper, op.getName());
                OptimizationUtils.removeOp(sd, helper, swishOp.getName());

                // Remove old variables
                OptimizationUtils.removeVariable(sd, helper, swishOutputVar);
                OptimizationUtils.removeVariable(sd, helper, mulOutputVar);

                // Rename fused output to match original output name
                if (!swishMulName.equals(mulOutputVar)) {
                    sd.renameVariable(swishMulName, mulOutputVar);
                }
                if (wasOutput) {
                    graphOutputs.add(mulOutputVar);
                }

                return true;
            } catch (Exception e) {
                log.warn("Failed to fuse SwiGLU pattern: {}", e.getMessage());
                return false;
            }
        }
    }

    /**
     * Fuses decomposed softmax pattern into a single softmax op.
     *
     * Pattern: div(exp(sub(x, reduce_max(x, axis, keepDims=true))),
     *              reduce_sum(exp(sub(x, reduce_max(x, axis, keepDims=true))), axis, keepDims=true))
     *
     * Anchors on DivOp: numerator must be exp(...), denominator must be
     * reduce_sum(exp(...)) where both exp ops share the same shifted input.
     */
    public static class FuseSoftmaxPattern implements Optimizer {
        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof DivOp)) {
                return false;
            }

            List<String> divInputs = op.getInputsToOp();
            if (divInputs == null || divInputs.size() != 2) {
                return false;
            }

            String numeratorVar = divInputs.get(0);
            String denominatorVar = divInputs.get(1);

            // Numerator must be produced by exp
            SameDiffOp expOp = producerOp(sd, helper, numeratorVar);
            if (expOp == null || !(expOp.getOp() instanceof Exp)) {
                return false;
            }

            // Denominator must be produced by reduce_sum
            SameDiffOp sumOp = producerOp(sd, helper, denominatorVar);
            if (sumOp == null || !(sumOp.getOp() instanceof Sum)) {
                return false;
            }

            // reduce_sum's input must be the same exp output (numerator)
            List<String> sumInputs = sumOp.getInputsToOp();
            if (sumInputs == null || sumInputs.isEmpty() || !sumInputs.get(0).equals(numeratorVar)) {
                return false;
            }

            // exp's input must come from sub
            List<String> expInputs = expOp.getInputsToOp();
            if (expInputs == null || expInputs.isEmpty()) {
                return false;
            }
            String shiftedVar = expInputs.get(0);
            SameDiffOp subOp = producerOp(sd, helper, shiftedVar);
            if (subOp == null || !(subOp.getOp() instanceof SubOp)) {
                return false;
            }

            // sub's inputs: (x, reduce_max(x))
            List<String> subInputs = subOp.getInputsToOp();
            if (subInputs == null || subInputs.size() != 2) {
                return false;
            }
            String xVar = subInputs.get(0);
            String maxVar = subInputs.get(1);

            // maxVar must come from reduce_max
            SameDiffOp maxOp = producerOp(sd, helper, maxVar);
            if (maxOp == null || !(maxOp.getOp() instanceof Max)) {
                return false;
            }

            // reduce_max's input must be x
            List<String> maxInputs = maxOp.getInputsToOp();
            if (maxInputs == null || maxInputs.isEmpty() || !maxInputs.get(0).equals(xVar)) {
                return false;
            }

            // Verify single-consumer chains (each intermediate feeds only the next op)
            if (!hasOnlyConsumer(sd, helper, maxVar, subOp.getName())) return false;
            if (!hasOnlyConsumer(sd, helper, shiftedVar, expOp.getName())) return false;
            // exp output feeds both the div numerator AND the sum — must have exactly 2 consumers
            Variable expOutVar = helper != null ? helper.getVariable(numeratorVar) : sd.getVariables().get(numeratorVar);
            if (expOutVar == null) return false;
            List<String> expUsers = expOutVar.getInputsForOp();
            if (expUsers == null || expUsers.size() != 2) return false;
            if (!hasOnlyConsumer(sd, helper, denominatorVar, op.getName())) return false;

            // Extract the dimension from reduce_max (it and reduce_sum should use the same axis)
            int dimension = -1;
            DifferentialFunction maxFunc = maxOp.getOp();
            if (maxFunc instanceof BaseReduceOp) {
                long[] dims = ((BaseReduceOp) maxFunc).dimensions().toLongVector();
                if (dims != null && dims.length == 1) {
                    dimension = (int) dims[0];
                }
            }

            // Get div output
            List<String> divOutputs = op.getOutputsOfOp();
            if (divOutputs == null || divOutputs.isEmpty()) return false;
            String divOutputVar = divOutputs.get(0);

            log.info("Fusing decomposed softmax: x={}, dim={}", xVar, dimension);

            try {
                SDVariable x = sd.getVariable(xVar);
                if (x == null) return false;

                SDVariable fused = new SoftMax(sd, new SDVariable[]{x}, dimension).outputVariable();

                OptimizationUtils.replaceOpInputsWith(sd, helper, divOutputVar, fused.name());

                List<String> graphOutputs = sd.outputs();
                boolean wasOutput = graphOutputs != null && graphOutputs.remove(divOutputVar);

                // Remove all ops in the chain
                OptimizationUtils.removeOp(sd, helper, op.getName());       // div
                OptimizationUtils.removeOp(sd, helper, sumOp.getName());    // reduce_sum
                OptimizationUtils.removeOp(sd, helper, expOp.getName());    // exp
                OptimizationUtils.removeOp(sd, helper, subOp.getName());    // sub
                OptimizationUtils.removeOp(sd, helper, maxOp.getName());    // reduce_max

                // Remove intermediate variables (not x — it's the input)
                OptimizationUtils.removeVariable(sd, helper, divOutputVar);
                OptimizationUtils.removeVariable(sd, helper, denominatorVar);
                OptimizationUtils.removeVariable(sd, helper, numeratorVar);
                OptimizationUtils.removeVariable(sd, helper, shiftedVar);
                OptimizationUtils.removeVariable(sd, helper, maxVar);

                sd.renameVariable(fused.name(), divOutputVar);
                if (wasOutput) {
                    graphOutputs.add(divOutputVar);
                }

                return true;
            } catch (Exception e) {
                log.warn("Failed to fuse softmax pattern: {}", e.getMessage());
                return false;
            }
        }

        private SameDiffOp producerOp(SameDiff sd, OptimizationHelper helper, String varName) {
            Variable v = helper != null ? helper.getVariable(varName) : sd.getVariables().get(varName);
            if (v == null) return null;
            String opName = v.getOutputOfOp();
            if (opName == null) return null;
            return sd.getOps().get(opName);
        }

        private boolean hasOnlyConsumer(SameDiff sd, OptimizationHelper helper, String varName, String expectedConsumerOp) {
            Variable v = helper != null ? helper.getVariable(varName) : sd.getVariables().get(varName);
            if (v == null) return false;
            List<String> users = v.getInputsForOp();
            return users != null && users.size() == 1 && expectedConsumerOp.equals(users.get(0));
        }
    }

    // FuseGatedMLPPattern DISABLED: absorbing two matmuls into fused_gemm_swiglu
    // regresses throughput because (a) the C++ generic impl allocates temp buffers
    // and does 2 sequential MmulHelper::mmul calls, (b) it's MATMUL category which
    // breaks elementwise Triton island chains, and (c) DSP can no longer batch the
    // two separate matmul ops. The chain stops at swish_mul (BINARY_ELEMENTWISE)
    // which stays in Triton islands and lets DSP batch the matmuls.

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

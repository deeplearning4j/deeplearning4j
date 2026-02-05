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
import org.nd4j.linalg.api.ops.impl.reduce.floating.Mean;
import org.nd4j.linalg.api.ops.impl.reduce.floating.Norm2;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MulOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.AddOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.DivOp;
import org.nd4j.linalg.api.ops.impl.transforms.floating.RSqrt;
import org.nd4j.linalg.api.ops.impl.transforms.floating.Sqrt;
import org.nd4j.linalg.api.ops.impl.scalar.RectifiedLinear;
import org.nd4j.linalg.api.ops.impl.transforms.custom.MeanSquare;

import java.util.List;

/**
 * Normalization fusion optimizations inspired by Luminal's RowRMSNorm.
 * These optimizations detect and potentially fuse normalization patterns
 * that are common in modern transformer architectures.
 *
 * RMSNorm pattern (used in LLaMA, Mistral, etc.):
 *   output = x * weight / sqrt(mean(x^2) + eps)
 *
 * This is more efficient than LayerNorm as it doesn't require computing
 * the mean for centering, only the root mean square for scaling.
 *
 * The full pattern detection looks for:
 *   1. Square: x * x
 *   2. Mean: reduce_mean(x*x, axis=-1)
 *   3. Add eps: mean + eps
 *   4. Rsqrt: rsqrt(mean + eps) OR 1/sqrt(mean + eps)
 *   5. Scale: x * rsqrt_result
 *   6. Apply weight: scaled * weight
 */
@Slf4j
public class NormalizationFusionOptimizations extends BaseOptimizerSet {

    /**
     * Detects RMSNorm pattern and logs it for debugging.
     *
     * RMSNorm: output = x * weight / sqrt(mean(x^2) + eps)
     *
     * This is a detection-only optimizer. Full fusion would require:
     * 1. A native fused RMSNorm op in libnd4j
     * 2. Java wrapper for that op
     *
     * The pattern we look for (backwards from the final mul with weight):
     * mul(weight, mul(x, rsqrt(add(mean(mul(x, x)), eps))))
     */
    public static class DetectRMSNormPattern implements Optimizer {
        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            // Look for the final multiplication (x_normed * weight)
            if (!(op.getOp() instanceof MulOp)) {
                return false;
            }

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.size() != 2) {
                return false;
            }

            // Try to find: one input is weight (constant/variable), other is normalized x
            String weightVar = null;
            String normalizedVar = null;
            SameDiffOp normalizeOp = null;

            for (int i = 0; i < 2; i++) {
                String inputVar = inputs.get(i);
                Variable v = helper != null ? helper.getVariable(inputVar) : sd.getVariables().get(inputVar);
                if (v == null) continue;

                SDVariable sdVar = sd.getVariable(inputVar);
                if (sdVar == null) continue;

                // Check if this looks like a weight (constant or variable, 1D shape)
                if (sdVar.getVariableType() == VariableType.CONSTANT ||
                    sdVar.getVariableType() == VariableType.VARIABLE) {
                    long[] shape = sdVar.getShape();
                    if (shape != null && shape.length == 1) {
                        weightVar = inputVar;
                        normalizedVar = inputs.get(1 - i);
                        break;
                    }
                }
            }

            if (weightVar == null || normalizedVar == null) {
                return false;
            }

            // Now trace back from normalizedVar to see if it matches RMSNorm pattern
            // normalizedVar should come from mul(x, rsqrt(...))
            Variable normalizedVariable = helper != null ?
                helper.getVariable(normalizedVar) : sd.getVariables().get(normalizedVar);
            if (normalizedVariable == null) return false;

            String producerOpName = normalizedVariable.getOutputOfOp();
            if (producerOpName == null) return false;

            SameDiffOp producerOp = sd.getOps().get(producerOpName);
            if (producerOp == null || !(producerOp.getOp() instanceof MulOp)) {
                return false;
            }

            // This mul should be x * rsqrt_result
            List<String> mulInputs = producerOp.getInputsToOp();
            if (mulInputs == null || mulInputs.size() != 2) {
                return false;
            }

            // One of these should be an rsqrt output
            String xVar = null;
            String rsqrtVar = null;

            for (int i = 0; i < 2; i++) {
                String inputVar = mulInputs.get(i);
                Variable v = helper != null ? helper.getVariable(inputVar) : sd.getVariables().get(inputVar);
                if (v == null) continue;

                String opName = v.getOutputOfOp();
                if (opName == null) {
                    // This might be the x input (placeholder or variable)
                    xVar = inputVar;
                    continue;
                }

                SameDiffOp inputOp = sd.getOps().get(opName);
                if (inputOp != null && inputOp.getOp() instanceof RSqrt) {
                    rsqrtVar = inputVar;
                    if (xVar == null) xVar = mulInputs.get(1 - i);
                }
            }

            if (rsqrtVar == null) {
                // No rsqrt found, might use div/sqrt pattern instead
                return false;
            }

            // Trace rsqrt input - should be add(mean(x*x), eps)
            Variable rsqrtVariable = helper != null ?
                helper.getVariable(rsqrtVar) : sd.getVariables().get(rsqrtVar);
            if (rsqrtVariable == null) return false;

            SameDiffOp rsqrtOp = sd.getOps().get(rsqrtVariable.getOutputOfOp());
            if (rsqrtOp == null) return false;

            List<String> rsqrtInputs = rsqrtOp.getInputsToOp();
            if (rsqrtInputs == null || rsqrtInputs.isEmpty()) return false;

            String addEpsVar = rsqrtInputs.get(0);
            Variable addEpsVariable = helper != null ?
                helper.getVariable(addEpsVar) : sd.getVariables().get(addEpsVar);
            if (addEpsVariable == null) return false;

            String addOpName = addEpsVariable.getOutputOfOp();
            if (addOpName == null) return false;

            SameDiffOp addOp = sd.getOps().get(addOpName);
            if (addOp == null || !(addOp.getOp() instanceof AddOp)) {
                return false;
            }

            // One of add inputs should be mean(x*x), other should be epsilon constant
            List<String> addInputs = addOp.getInputsToOp();
            if (addInputs == null || addInputs.size() != 2) return false;

            String meanVar = null;
            for (String addInput : addInputs) {
                Variable addInputVar = helper != null ?
                    helper.getVariable(addInput) : sd.getVariables().get(addInput);
                if (addInputVar == null) continue;

                String meanOpName = addInputVar.getOutputOfOp();
                if (meanOpName == null) continue;

                SameDiffOp meanOp = sd.getOps().get(meanOpName);
                if (meanOp != null && meanOp.getOp() instanceof Mean) {
                    meanVar = addInput;
                    break;
                }
            }

            if (meanVar == null) {
                return false;
            }

            // We detected RMSNorm pattern!
            log.debug("Detected RMSNorm pattern: x={}, weight={}", xVar, weightVar);

            // For now, just detection. Full fusion would require native op.
            // TODO: Implement fused RMSNorm op and return true when fusing
            return false;
        }
    }

    /**
     * Fuses mean(x^2) pattern: mean(mul(x, x)) -> mean_square(x)
     *
     * This is a component of RMSNorm that can be optimized.
     * The pattern mean(x * x) is fused into a single mean_square op.
     */
    public static class FuseMeanSquarePattern implements Optimizer {
        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof Mean)) {
                return false;
            }

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.isEmpty()) {
                return false;
            }

            // Check if input comes from mul(x, x)
            String inputVar = inputs.get(0);
            Variable v = helper != null ? helper.getVariable(inputVar) : sd.getVariables().get(inputVar);
            if (v == null) return false;

            String producerOpName = v.getOutputOfOp();
            if (producerOpName == null) return false;

            SameDiffOp producerOp = sd.getOps().get(producerOpName);
            if (producerOp == null || !(producerOp.getOp() instanceof MulOp)) {
                return false;
            }

            // Check if mul is x * x (same variable both inputs)
            List<String> mulInputs = producerOp.getInputsToOp();
            if (mulInputs == null || mulInputs.size() != 2) return false;

            if (!mulInputs.get(0).equals(mulInputs.get(1))) {
                return false;
            }

            String xVar = mulInputs.get(0);

            // Check that the mul output is only used by this mean
            Variable mulOutVariable = helper != null ?
                helper.getVariable(inputVar) : sd.getVariables().get(inputVar);
            if (mulOutVariable == null) return false;

            List<String> mulOutputUsers = mulOutVariable.getInputsForOp();
            if (mulOutputUsers == null || mulOutputUsers.size() != 1) {
                return false;
            }

            // Get the mean output
            List<String> outputs = op.getOutputsOfOp();
            if (outputs == null || outputs.isEmpty()) {
                return false;
            }
            String meanOutputVar = outputs.get(0);

            log.info("Fusing mean({}^2) -> mean_square({})", xVar, xVar);

            try {
                // Create fused mean_square operation
                SDVariable xSdVar = sd.getVariable(xVar);
                if (xSdVar == null) return false;

                // Create the fused MeanSquare op - calling outputVariable() registers it with SameDiff
                SDVariable meanSquareOutput = new MeanSquare(sd, xSdVar, true).outputVariable();

                // Replace all uses of the mean output with mean_square output
                OptimizationUtils.replaceOpInputsWith(sd, helper, meanOutputVar, meanSquareOutput.name());

                // Remove the old mean and mul operations
                OptimizationUtils.removeOp(sd, helper, op.getName());
                OptimizationUtils.removeOp(sd, helper, producerOp.getName());

                // Remove old variables
                OptimizationUtils.removeVariable(sd, helper, inputVar);
                OptimizationUtils.removeVariable(sd, helper, meanOutputVar);

                return true;
            } catch (Exception e) {
                log.warn("Failed to fuse mean(x^2) pattern: {}", e.getMessage());
                return false;
            }
        }
    }
}

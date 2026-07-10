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
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.optimize.OptimizationHelper;
import org.nd4j.autodiff.samediff.optimize.Optimizer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.scalar.Pow;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.DivOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MulOp;

import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Strength reduction optimizations: replacing expensive ops with cheaper equivalents.
 *
 * Patterns:
 * - pow(x, 2) → square(x)          [avoids transcendental function]
 * - pow(x, 0.5) → sqrt(x)          [dedicated fast path]
 * - pow(x, -1) → reciprocal(x)     [dedicated fast path]
 * - pow(x, 1) → x                  [identity elimination]
 * - pow(x, 0) → 1                  [constant result]
 * - div(x, const) → mul(x, 1/const) [multiply is faster than divide]
 */
@Slf4j
public class StrengthReductionOptimizations extends BaseOptimizerSet {

    /**
     * pow(x, 2) → square(x)
     * pow(x, 0.5) → sqrt(x)
     * pow(x, -1) → reciprocal(x)
     * pow(x, 1) → x
     *
     * Targets the scalar Pow op where the exponent is a known constant.
     */
    public static class ReduceScalarPow implements Optimizer {

        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return Collections.singleton(Pow.class);
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof Pow)) return false;

            Pow powOp = (Pow) op.getOp();
            double exponent = powOp.scalar().getDouble(0);

            List<String> inputs = op.getInputsToOp();
            List<String> outputs = op.getOutputsOfOp();
            if (inputs == null || inputs.isEmpty() || outputs == null || outputs.isEmpty()) return false;

            String inputVar = inputs.get(0);
            String outputVar = outputs.get(0);

            if (Math.abs(exponent - 2.0) < 1e-10) {
                // pow(x, 2) → square(x)
                log.debug("Strength reduction: pow({}, 2) → square({})", inputVar, inputVar);
                SDVariable squareOut = sd.math().square(inputVar + "__sr_sq", sd.getVariable(inputVar));
                return replaceAndCleanup(sd, helper, op, outputVar, squareOut.name());
            }

            if (Math.abs(exponent - 0.5) < 1e-10) {
                // pow(x, 0.5) → sqrt(x)
                log.debug("Strength reduction: pow({}, 0.5) → sqrt({})", inputVar, inputVar);
                SDVariable sqrtOut = sd.math().sqrt(inputVar + "__sr_sqrt", sd.getVariable(inputVar));
                return replaceAndCleanup(sd, helper, op, outputVar, sqrtOut.name());
            }

            if (Math.abs(exponent - (-1.0)) < 1e-10) {
                // pow(x, -1) → reciprocal(x)
                log.debug("Strength reduction: pow({}, -1) → reciprocal({})", inputVar, inputVar);
                SDVariable recipOut = sd.math().reciprocal(inputVar + "__sr_recip", sd.getVariable(inputVar));
                return replaceAndCleanup(sd, helper, op, outputVar, recipOut.name());
            }

            if (Math.abs(exponent - 1.0) < 1e-10) {
                // pow(x, 1) → x (identity)
                log.debug("Strength reduction: pow({}, 1) → {} (identity)", inputVar, inputVar);
                OptimizationUtils.replaceOpInputsWith(sd, helper, outputVar, inputVar);
                List<String> graphOutputs = sd.outputs();
                boolean wasOutput = false;
                if (graphOutputs != null) {
                    for (int i = 0; i < graphOutputs.size(); i++) {
                        if (graphOutputs.get(i).equals(outputVar)) {
                            graphOutputs.set(i, inputVar);
                            wasOutput = true;
                        }
                    }
                }
                OptimizationUtils.removeOp(sd, helper, op.getName());
                if (!wasOutput) OptimizationUtils.removeVariable(sd, helper, outputVar);
                return true;
            }

            if (Math.abs(exponent) < 1e-10) {
                // pow(x, 0) → 1
                log.debug("Strength reduction: pow({}, 0) → 1", inputVar);
                SDVariable inputSdVar = sd.getVariable(inputVar);
                if (inputSdVar == null) return false;
                SDVariable one = sd.constant("one_pow0_" + System.nanoTime(),
                        org.nd4j.linalg.factory.Nd4j.ones(inputSdVar.dataType(), 1));
                OptimizationUtils.replaceOpInputsWith(sd, helper, outputVar, one.name());
                List<String> graphOutputs = sd.outputs();
                boolean wasOutput = false;
                if (graphOutputs != null) {
                    for (int i = 0; i < graphOutputs.size(); i++) {
                        if (graphOutputs.get(i).equals(outputVar)) {
                            graphOutputs.set(i, one.name());
                            wasOutput = true;
                        }
                    }
                }
                OptimizationUtils.removeOp(sd, helper, op.getName());
                if (!wasOutput) OptimizationUtils.removeVariable(sd, helper, outputVar);
                return true;
            }

            return false;
        }

        private boolean replaceAndCleanup(SameDiff sd, OptimizationHelper helper,
                                          SameDiffOp op, String oldOutput, String newOutput) {
            OptimizationUtils.replaceOpInputsWith(sd, helper, oldOutput, newOutput);
            List<String> graphOutputs = sd.outputs();
            boolean wasOutput = graphOutputs != null && graphOutputs.remove(oldOutput);
            OptimizationUtils.removeOp(sd, helper, op.getName());
            OptimizationUtils.removeVariable(sd, helper, oldOutput);
            sd.renameVariable(newOutput, oldOutput);
            if (wasOutput) graphOutputs.add(oldOutput);
            return true;
        }
    }

    /**
     * div(x, constant) → mul(x, 1/constant) when the divisor is a scalar constant.
     * Multiplication is faster than division on both CPU and GPU.
     */
    public static class DivByConstantToMul implements Optimizer {

        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return Collections.singleton(DivOp.class);
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof DivOp)) return false;

            List<String> inputs = op.getInputsToOp();
            List<String> outputs = op.getOutputsOfOp();
            if (inputs == null || inputs.size() != 2 || outputs == null || outputs.isEmpty()) return false;

            String dividend = inputs.get(0);
            String divisor = inputs.get(1);
            String outputVar = outputs.get(0);

            // Divisor must be a constant
            SDVariable divisorSdVar = sd.getVariable(divisor);
            if (divisorSdVar == null || divisorSdVar.getVariableType() != VariableType.CONSTANT) return false;

            INDArray divisorArr = constantArrays.getArray(divisor);
            if (divisorArr == null) return false;

            // Only handle scalar or single-element divisors
            if (!divisorArr.isScalar() && divisorArr.length() != 1) return false;

            double divisorVal = divisorArr.getDouble(0);
            // Don't convert if divisor is 0, 1, or -1 (special cases handled elsewhere or would cause issues)
            if (divisorVal == 0.0 || Math.abs(divisorVal) == 1.0) return false;

            // Check for near-zero divisors that would produce infinity when inverted
            if (Math.abs(divisorVal) < 1e-30) return false;

            double reciprocal = 1.0 / divisorVal;
            log.debug("Strength reduction: div({}, {}) → mul({}, {})", dividend, divisorVal, dividend, reciprocal);

            // Create the reciprocal constant
            String recipName = divisor + "__sr_recip";
            int suffix = 0;
            while (sd.hasVariable(recipName)) recipName = divisor + "__sr_recip_" + (suffix++);
            sd.constant(recipName, org.nd4j.linalg.factory.Nd4j.scalar(divisorArr.dataType(), reciprocal));

            // Create mul(x, 1/const)
            String mulName = outputVar + "__sr_mul";
            suffix = 0;
            while (sd.hasVariable(mulName)) mulName = outputVar + "__sr_mul_" + (suffix++);
            sd.getVariable(dividend).mul(mulName, sd.getVariable(recipName));

            // Replace and cleanup
            OptimizationUtils.replaceOpInputsWith(sd, helper, outputVar, mulName);
            List<String> graphOutputs = sd.outputs();
            boolean wasOutput = graphOutputs != null && graphOutputs.remove(outputVar);
            OptimizationUtils.removeOp(sd, helper, op.getName());
            OptimizationUtils.removeVariable(sd, helper, outputVar);
            sd.renameVariable(mulName, outputVar);
            if (wasOutput) graphOutputs.add(outputVar);
            return true;
        }
    }

    /**
     * Reduces the custom Pow op (two-input: pow(x, y)) when y is a constant.
     * pow(x, 2) → square(x), pow(x, 0.5) → sqrt(x), pow(x, -1) → reciprocal(x)
     */
    public static class ReduceCustomPow implements Optimizer {

        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            Set<Class<? extends DifferentialFunction>> types = new HashSet<>();
            types.add(org.nd4j.linalg.api.ops.impl.transforms.custom.Pow.class);
            return types;
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof org.nd4j.linalg.api.ops.impl.transforms.custom.Pow)) return false;

            List<String> inputs = op.getInputsToOp();
            List<String> outputs = op.getOutputsOfOp();
            if (inputs == null || inputs.size() != 2 || outputs == null || outputs.isEmpty()) return false;

            String baseVar = inputs.get(0);
            String expVar = inputs.get(1);
            String outputVar = outputs.get(0);

            // Exponent must be a constant scalar
            SDVariable expSdVar = sd.getVariable(expVar);
            if (expSdVar == null || expSdVar.getVariableType() != VariableType.CONSTANT) return false;

            INDArray expArr = constantArrays.getArray(expVar);
            if (expArr == null || (!expArr.isScalar() && expArr.length() != 1)) return false;

            double exponent = expArr.getDouble(0);

            if (Math.abs(exponent - 2.0) < 1e-10) {
                log.debug("Strength reduction: Pow({}, 2) → square({})", baseVar, baseVar);
                SDVariable squareOut = sd.math().square(baseVar + "__sr_sq", sd.getVariable(baseVar));
                return replaceAndCleanup(sd, helper, op, outputVar, squareOut.name());
            }

            if (Math.abs(exponent - 0.5) < 1e-10) {
                log.debug("Strength reduction: Pow({}, 0.5) → sqrt({})", baseVar, baseVar);
                SDVariable sqrtOut = sd.math().sqrt(baseVar + "__sr_sqrt", sd.getVariable(baseVar));
                return replaceAndCleanup(sd, helper, op, outputVar, sqrtOut.name());
            }

            if (Math.abs(exponent - (-1.0)) < 1e-10) {
                log.debug("Strength reduction: Pow({}, -1) → reciprocal({})", baseVar, baseVar);
                SDVariable recipOut = sd.math().reciprocal(baseVar + "__sr_recip", sd.getVariable(baseVar));
                return replaceAndCleanup(sd, helper, op, outputVar, recipOut.name());
            }

            if (Math.abs(exponent - 1.0) < 1e-10) {
                log.debug("Strength reduction: Pow({}, 1) → {} (identity)", baseVar, baseVar);
                OptimizationUtils.replaceOpInputsWith(sd, helper, outputVar, baseVar);
                List<String> graphOutputs = sd.outputs();
                boolean wasOutput = false;
                if (graphOutputs != null) {
                    for (int i = 0; i < graphOutputs.size(); i++) {
                        if (graphOutputs.get(i).equals(outputVar)) {
                            graphOutputs.set(i, baseVar);
                            wasOutput = true;
                        }
                    }
                }
                OptimizationUtils.removeOp(sd, helper, op.getName());
                if (!wasOutput) OptimizationUtils.removeVariable(sd, helper, outputVar);
                return true;
            }

            if (Math.abs(exponent) < 1e-10) {
                // Pow(x, 0) → 1
                log.debug("Strength reduction: Pow({}, 0) → 1", baseVar);
                SDVariable baseSdVar = sd.getVariable(baseVar);
                if (baseSdVar == null) return false;
                SDVariable one = sd.constant("one_pow0_" + System.nanoTime(),
                        org.nd4j.linalg.factory.Nd4j.ones(baseSdVar.dataType(), 1));
                OptimizationUtils.replaceOpInputsWith(sd, helper, outputVar, one.name());
                List<String> graphOutputs = sd.outputs();
                boolean wasOutput = false;
                if (graphOutputs != null) {
                    for (int i = 0; i < graphOutputs.size(); i++) {
                        if (graphOutputs.get(i).equals(outputVar)) {
                            graphOutputs.set(i, one.name());
                            wasOutput = true;
                        }
                    }
                }
                OptimizationUtils.removeOp(sd, helper, op.getName());
                if (!wasOutput) OptimizationUtils.removeVariable(sd, helper, outputVar);
                return true;
            }

            return false;
        }

        private boolean replaceAndCleanup(SameDiff sd, OptimizationHelper helper,
                                          SameDiffOp op, String oldOutput, String newOutput) {
            OptimizationUtils.replaceOpInputsWith(sd, helper, oldOutput, newOutput);
            List<String> graphOutputs = sd.outputs();
            boolean wasOutput = graphOutputs != null && graphOutputs.remove(oldOutput);
            OptimizationUtils.removeOp(sd, helper, op.getName());
            OptimizationUtils.removeVariable(sd, helper, oldOutput);
            sd.renameVariable(newOutput, oldOutput);
            if (wasOutput) graphOutputs.add(oldOutput);
            return true;
        }
    }
}

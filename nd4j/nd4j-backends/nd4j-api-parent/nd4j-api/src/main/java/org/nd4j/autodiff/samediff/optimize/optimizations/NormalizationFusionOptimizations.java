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
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.api.ops.impl.reduce.floating.Mean;
import org.nd4j.linalg.api.ops.impl.transforms.custom.MeanSquare;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MulOp;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Normalization-related graph fusions.
 *
 * Supported RMSNorm decomposition patterns:
 * 1) rsqrt form: x * rsqrt(mean(x*x) + eps) * gamma
 * 2) sqrt/div form: (x / sqrt(mean(pow(x,2)) + eps)) * gamma
 */
@Slf4j
public class NormalizationFusionOptimizations extends BaseOptimizerSet {

    /**
     * Fuse decomposed RMSNorm chains into native rms_norm op.
     */
    public static class FuseRMSNormPattern implements Optimizer {

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (op == null || op.getOp() == null || !(op.getOp() instanceof MulOp)) {
                return false;
            }

            List<String> finalInputs = op.getInputsToOp();
            List<String> finalOutputs = op.getOutputsOfOp();
            if (finalInputs == null || finalInputs.size() != 2 || finalOutputs == null || finalOutputs.isEmpty()) {
                return false;
            }

            String finalOutputVar = finalOutputs.get(0);
            RmsNormMatch match = matchRmsNorm(sd, helper, op, finalOutputVar);
            if (match == null) {
                return false;
            }

            try {
                SDVariable x = sd.getVariable(match.xVar);
                SDVariable gamma = sd.getVariable(match.gammaVar);
                if (x == null || gamma == null) {
                    return false;
                }

                SDVariable fused = sd.nn().rmsNorm(x, gamma, match.epsilon);
                OptimizationUtils.replaceOpInputsWith(sd, helper, finalOutputVar, fused.name());

                for (String opName : match.opsToRemove) {
                    OptimizationUtils.removeOp(sd, helper, opName);
                }
                for (String varName : match.varsToRemove) {
                    if (!match.xVar.equals(varName) && !match.gammaVar.equals(varName)) {
                        OptimizationUtils.removeVariable(sd, helper, varName);
                    }
                }

                log.info("Fused RMSNorm pattern: x={}, gamma={}, eps={}", match.xVar, match.gammaVar, match.epsilon);
                return true;
            } catch (Exception e) {
                log.debug("Failed to fuse RMSNorm pattern at op {}: {}", op.getName(), e.getMessage());
                return false;
            }
        }

        private RmsNormMatch matchRmsNorm(SameDiff sd, OptimizationHelper helper, SameDiffOp finalMul, String finalOutputVar) {
            List<String> finalInputs = finalMul.getInputsToOp();
            String gammaVar = null;
            String normalizedVar = null;

            for (int i = 0; i < 2; i++) {
                String candidate = finalInputs.get(i);
                if (isLikelyGamma(sd, candidate)) {
                    gammaVar = candidate;
                    normalizedVar = finalInputs.get(1 - i);
                    break;
                }
            }

            if (gammaVar == null || normalizedVar == null) {
                return null;
            }
            if (!hasOnlyConsumer(sd, helper, normalizedVar, finalMul.getName())) {
                return null;
            }

            SameDiffOp normalizedOp = producerOp(sd, helper, normalizedVar);
            if (normalizedOp == null || normalizedOp.getInputsToOp() == null || normalizedOp.getInputsToOp().size() != 2) {
                return null;
            }

            String normalizedOpName = normalizedOp.getName();
            String normalizedType = opName(normalizedOp);
            String xVar;
            String normFactorVar;
            boolean rsqrtPath;
            boolean divSqrtPath;

            if ("mul".equals(normalizedType)) {
                String in0 = normalizedOp.getInputsToOp().get(0);
                String in1 = normalizedOp.getInputsToOp().get(1);
                SameDiffOp p0 = producerOp(sd, helper, in0);
                SameDiffOp p1 = producerOp(sd, helper, in1);
                if (p0 != null && "rsqrt".equals(opName(p0))) {
                    normFactorVar = in0;
                    xVar = in1;
                } else if (p1 != null && "rsqrt".equals(opName(p1))) {
                    normFactorVar = in1;
                    xVar = in0;
                } else {
                    return null;
                }
                rsqrtPath = true;
                divSqrtPath = false;
            } else if ("div".equals(normalizedType)) {
                String numerator = normalizedOp.getInputsToOp().get(0);
                String denominator = normalizedOp.getInputsToOp().get(1);
                SameDiffOp denomOp = producerOp(sd, helper, denominator);
                if (denomOp == null || !"sqrt".equals(opName(denomOp))) {
                    return null;
                }
                xVar = numerator;
                normFactorVar = denominator;
                rsqrtPath = false;
                divSqrtPath = true;
            } else {
                return null;
            }

            SameDiffOp normFactorOp = producerOp(sd, helper, normFactorVar);
            if (normFactorOp == null || normFactorOp.getInputsToOp() == null || normFactorOp.getInputsToOp().isEmpty()) {
                return null;
            }
            if (!hasOnlyConsumer(sd, helper, normFactorVar, normalizedOpName)) {
                return null;
            }

            String factorType = opName(normFactorOp);
            if (rsqrtPath && !"rsqrt".equals(factorType)) {
                return null;
            }
            if (divSqrtPath && !"sqrt".equals(factorType)) {
                return null;
            }

            String addVar = normFactorOp.getInputsToOp().get(0);
            if (!hasOnlyConsumer(sd, helper, addVar, normFactorOp.getName())) {
                return null;
            }
            SameDiffOp addOp = producerOp(sd, helper, addVar);
            if (addOp == null) {
                return null;
            }

            String meanVar = null;
            Double epsilon = null;
            String addType = opName(addOp);
            if ("add".equals(addType)) {
                List<String> addInputs = addOp.getInputsToOp();
                if (addInputs == null || addInputs.size() != 2) return null;

                for (String in : addInputs) {
                    SameDiffOp p = producerOp(sd, helper, in);
                    if (p != null && isMeanLike(p)) {
                        meanVar = in;
                    } else {
                        epsilon = scalarFromVariable(sd, in);
                    }
                }
            } else if ("add_scalar".equals(addType)) {
                List<String> addInputs = addOp.getInputsToOp();
                if (addInputs == null || addInputs.size() != 1) return null;
                meanVar = addInputs.get(0);
                epsilon = scalarFromScalarOp(addOp);
            } else {
                return null;
            }

            if (meanVar == null || epsilon == null) {
                return null;
            }
            if (!hasOnlyConsumer(sd, helper, meanVar, addOp.getName())) {
                return null;
            }

            SameDiffOp meanOp = producerOp(sd, helper, meanVar);
            if (meanOp == null) {
                return null;
            }

            String expectedX = stripTrivial(sd, helper, xVar);
            String meanInputVar;
            String squareVar = null;
            SameDiffOp squareOp = null;

            if ("mean_square".equals(opName(meanOp))) {
                List<String> meanInputs = meanOp.getInputsToOp();
                if (meanInputs == null || meanInputs.isEmpty()) return null;
                meanInputVar = stripTrivial(sd, helper, meanInputs.get(0));
            } else if ("reduce_mean".equals(opName(meanOp)) || meanOp.getOp() instanceof Mean) {
                List<String> meanInputs = meanOp.getInputsToOp();
                if (meanInputs == null || meanInputs.isEmpty()) return null;
                squareVar = meanInputs.get(0);
                if (!hasOnlyConsumer(sd, helper, squareVar, meanOp.getName())) {
                    return null;
                }

                squareOp = producerOp(sd, helper, squareVar);
                if (squareOp == null) {
                    return null;
                }
                String squareType = opName(squareOp);
                if ("mul".equals(squareType)) {
                    List<String> sqInputs = squareOp.getInputsToOp();
                    if (sqInputs == null || sqInputs.size() != 2 || !sqInputs.get(0).equals(sqInputs.get(1))) {
                        return null;
                    }
                    meanInputVar = stripTrivial(sd, helper, sqInputs.get(0));
                } else if ("pow".equals(squareType) || "pow_scalar".equals(squareType)) {
                    List<String> sqInputs = squareOp.getInputsToOp();
                    if (sqInputs == null || sqInputs.isEmpty()) return null;
                    meanInputVar = stripTrivial(sd, helper, sqInputs.get(0));
                    Double powVal = null;
                    if (sqInputs.size() >= 2) {
                        powVal = scalarFromVariable(sd, sqInputs.get(1));
                    }
                    if (powVal == null) {
                        powVal = scalarFromScalarOp(squareOp);
                    }
                    if (powVal == null || Math.abs(powVal - 2.0) > 1e-6) {
                        return null;
                    }
                } else if ("square".equals(squareType)) {
                    List<String> sqInputs = squareOp.getInputsToOp();
                    if (sqInputs == null || sqInputs.isEmpty()) return null;
                    meanInputVar = stripTrivial(sd, helper, sqInputs.get(0));
                } else {
                    return null;
                }
            } else {
                return null;
            }

            if (!expectedX.equals(meanInputVar)) {
                return null;
            }

            RmsNormMatch m = new RmsNormMatch();
            m.xVar = xVar;
            m.gammaVar = gammaVar;
            m.epsilon = epsilon;

            m.opsToRemove.add(finalMul.getName());
            m.opsToRemove.add(normalizedOpName);
            m.opsToRemove.add(normFactorOp.getName());
            m.opsToRemove.add(addOp.getName());
            m.opsToRemove.add(meanOp.getName());
            if (squareOp != null) {
                m.opsToRemove.add(squareOp.getName());
            }

            m.varsToRemove.add(finalOutputVar);
            m.varsToRemove.add(normalizedVar);
            m.varsToRemove.add(normFactorVar);
            m.varsToRemove.add(addVar);
            m.varsToRemove.add(meanVar);
            if (squareVar != null) {
                m.varsToRemove.add(squareVar);
            }

            return m;
        }

        private String stripTrivial(SameDiff sd, OptimizationHelper helper, String varName) {
            String current = varName;
            for (int i = 0; i < 8; i++) {
                SameDiffOp p = producerOp(sd, helper, current);
                if (p == null || p.getInputsToOp() == null || p.getInputsToOp().isEmpty()) {
                    break;
                }
                String n = opName(p);
                if ("cast".equals(n) || "identity".equals(n)) {
                    current = p.getInputsToOp().get(0);
                } else {
                    break;
                }
            }
            return current;
        }

        private boolean isLikelyGamma(SameDiff sd, String varName) {
            SDVariable v = sd.getVariable(varName);
            if (v == null) return false;
            if (v.getVariableType() != VariableType.CONSTANT && v.getVariableType() != VariableType.VARIABLE) {
                return false;
            }
            long[] shape = v.getShape();
            return shape != null && shape.length == 1 && shape[0] > 1;
        }

        private boolean hasOnlyConsumer(SameDiff sd, OptimizationHelper helper, String varName, String expectedConsumerOp) {
            Variable v = helper != null ? helper.getVariable(varName) : null;
            if (v == null) {
                v = sd.getVariables().get(varName);
            }
            if (v == null) return false;
            List<String> users = v.getInputsForOp();
            return users != null && users.size() == 1 && expectedConsumerOp.equals(users.get(0));
        }

        private SameDiffOp producerOp(SameDiff sd, OptimizationHelper helper, String varName) {
            Variable v = helper != null ? helper.getVariable(varName) : null;
            if (v == null) {
                v = sd.getVariables().get(varName);
            }
            if (v == null) return null;
            String opName = v.getOutputOfOp();
            if (opName == null) return null;
            return sd.getOps().get(opName);
        }

        private boolean isMeanLike(SameDiffOp op) {
            String n = opName(op);
            return "reduce_mean".equals(n) || "mean_square".equals(n) || op.getOp() instanceof Mean;
        }

        private String opName(SameDiffOp op) {
            return op != null && op.getOp() != null && op.getOp().opName() != null
                    ? op.getOp().opName().toLowerCase() : "";
        }

        private Double scalarFromVariable(SameDiff sd, String varName) {
            SDVariable v = sd.getVariable(varName);
            if (v == null || v.getArr() == null || !v.getArr().isScalar()) {
                return null;
            }
            return v.getArr().getDouble(0);
        }

        private Double scalarFromScalarOp(SameDiffOp op) {
            if (op == null || op.getOp() == null || !(op.getOp() instanceof ScalarOp)) {
                return null;
            }
            INDArray scalar = ((ScalarOp) op.getOp()).scalar();
            if (scalar == null || !scalar.isScalar()) {
                return null;
            }
            return scalar.getDouble(0);
        }
    }

    /**
     * Fuses mean(x^2) pattern: mean(mul(x, x)) -> mean_square(x).
     * This is useful both standalone and as an enabler for RMSNorm fusion.
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

            String inputVar = inputs.get(0);
            Variable v = helper != null ? helper.getVariable(inputVar) : sd.getVariables().get(inputVar);
            if (v == null) return false;

            String producerOpName = v.getOutputOfOp();
            if (producerOpName == null) return false;

            SameDiffOp producerOp = sd.getOps().get(producerOpName);
            if (producerOp == null || !(producerOp.getOp() instanceof MulOp)) {
                return false;
            }

            List<String> mulInputs = producerOp.getInputsToOp();
            if (mulInputs == null || mulInputs.size() != 2 || !mulInputs.get(0).equals(mulInputs.get(1))) {
                return false;
            }

            Variable mulOutVariable = helper != null ? helper.getVariable(inputVar) : sd.getVariables().get(inputVar);
            if (mulOutVariable == null) return false;

            List<String> mulOutputUsers = mulOutVariable.getInputsForOp();
            if (mulOutputUsers == null || mulOutputUsers.size() != 1) {
                return false;
            }

            List<String> outputs = op.getOutputsOfOp();
            if (outputs == null || outputs.isEmpty()) {
                return false;
            }
            String meanOutputVar = outputs.get(0);
            String xVar = mulInputs.get(0);

            try {
                SDVariable xSdVar = sd.getVariable(xVar);
                if (xSdVar == null) return false;

                SDVariable meanSquareOutput = new MeanSquare(sd, xSdVar, true).outputVariable();
                OptimizationUtils.replaceOpInputsWith(sd, helper, meanOutputVar, meanSquareOutput.name());

                OptimizationUtils.removeOp(sd, helper, op.getName());
                OptimizationUtils.removeOp(sd, helper, producerOp.getName());
                OptimizationUtils.removeVariable(sd, helper, inputVar);
                OptimizationUtils.removeVariable(sd, helper, meanOutputVar);

                return true;
            } catch (Exception e) {
                log.warn("Failed to fuse mean(x^2) pattern: {}", e.getMessage());
                return false;
            }
        }
    }

    private static class RmsNormMatch {
        String xVar;
        String gammaVar;
        double epsilon;
        final Set<String> opsToRemove = new LinkedHashSet<>();
        final Set<String> varsToRemove = new LinkedHashSet<>();
    }
}

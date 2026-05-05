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
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.api.ops.impl.reduce.floating.Mean;
import org.nd4j.linalg.api.ops.impl.scalar.Pow;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarAdd;
import org.nd4j.linalg.api.ops.impl.transforms.custom.MeanSquare;
import org.nd4j.linalg.api.ops.impl.transforms.floating.RSqrt;
import org.nd4j.linalg.api.ops.impl.transforms.floating.Sqrt;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.DivOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MulOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.PowPairwise;
import org.nd4j.linalg.api.ops.impl.transforms.same.Square;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.impl.reduce.Mmul;
import org.nd4j.linalg.api.ops.impl.reduce.TensorMmul;
import org.nd4j.linalg.api.ops.impl.transforms.custom.RmsNorm;
import org.nd4j.linalg.api.ops.impl.transforms.custom.RmsNormLinear;

import java.util.ArrayList;
import java.util.HashSet;
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

                // Temporarily remove finalOutputVar from graph outputs so removeVariable
                // won't refuse to delete it (mirrors FuseRMSNormLinearPattern logic)
                List<String> graphOutputs = sd.outputs();
                boolean wasOutput = graphOutputs != null && graphOutputs.remove(finalOutputVar);

                for (String varName : match.varsToRemove) {
                    if (!match.xVar.equals(varName) && !match.gammaVar.equals(varName)) {
                        OptimizationUtils.removeVariable(sd, helper, varName);
                    }
                }

                // Rename fused variable back to the original name so that downstream
                // name-based lookups (DSP plan compilation, decode loop state resolution,
                // external output requests) continue to work correctly.
                sd.renameVariable(fused.name(), finalOutputVar);
                if (wasOutput) {
                    graphOutputs.add(finalOutputVar);
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
                log.debug("RMSNorm: no gamma found for mul op {} inputs={}", finalMul.getName(), finalInputs);
                return null;
            }
            if (!hasOnlyConsumer(sd, helper, normalizedVar, finalMul.getName())) {
                log.debug("RMSNorm: normalizedVar {} has multiple consumers (expected only {})", normalizedVar, finalMul.getName());
                return null;
            }

            SameDiffOp normalizedOp = producerOp(sd, helper, normalizedVar);
            if (normalizedOp == null || normalizedOp.getInputsToOp() == null) {
                log.debug("RMSNorm: normalizedOp null for var {}", normalizedVar);
                return null;
            }

            // Strip cast ops wrapping the normalization output (mixed-precision models
            // cast FP32→FP16 after div/mul before the final gamma multiply)
            Set<String> castOpsToRemove = new LinkedHashSet<>();
            Set<String> castVarsToRemove = new LinkedHashSet<>();
            for (int depth = 0; depth < 4; depth++) {
                if ("cast".equals(opName(normalizedOp))) {
                    castOpsToRemove.add(normalizedOp.getName());
                    castVarsToRemove.add(normalizedVar);
                    normalizedVar = normalizedOp.getInputsToOp().get(0);
                    normalizedOp = producerOp(sd, helper, normalizedVar);
                    if (normalizedOp == null || normalizedOp.getInputsToOp() == null) {
                        log.debug("RMSNorm: normalizedOp null after stripping cast for var {}", normalizedVar);
                        return null;
                    }
                } else {
                    break;
                }
            }

            if (normalizedOp.getInputsToOp().size() != 2) {
                log.debug("RMSNorm: normalizedOp wrong input count for var {}", normalizedVar);
                return null;
            }

            String normalizedOpName = normalizedOp.getName();
            DifferentialFunction normalizedFunc = normalizedOp.getOp();
            String xVar;
            String normFactorVar;
            boolean rsqrtPath;
            boolean divSqrtPath;

            if (normalizedFunc instanceof MulOp) {
                String in0 = normalizedOp.getInputsToOp().get(0);
                String in1 = normalizedOp.getInputsToOp().get(1);
                SameDiffOp p0 = producerOp(sd, helper, in0);
                SameDiffOp p1 = producerOp(sd, helper, in1);
                if (p0 != null && p0.getOp() instanceof RSqrt) {
                    normFactorVar = in0;
                    xVar = in1;
                } else if (p1 != null && p1.getOp() instanceof RSqrt) {
                    normFactorVar = in1;
                    xVar = in0;
                } else {
                    return null;
                }
                rsqrtPath = true;
                divSqrtPath = false;
            } else if (normalizedFunc instanceof DivOp) {
                String numerator = normalizedOp.getInputsToOp().get(0);
                String denominator = normalizedOp.getInputsToOp().get(1);
                SameDiffOp denomOp = producerOp(sd, helper, denominator);
                if (denomOp == null || !(denomOp.getOp() instanceof Sqrt)) {
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
            // Note: normFactorVar (sqrt/rsqrt output) may have multiple consumers
            // (e.g., reciprocal for inv_std_var output in SimplifiedLayerNormalization).
            // We still fuse, but skip removing normFactorVar/normFactorOp if they have other consumers.
            boolean normFactorHasOnlyConsumer = hasOnlyConsumer(sd, helper, normFactorVar, normalizedOpName);

            if (rsqrtPath && !(normFactorOp.getOp() instanceof RSqrt)) {
                return null;
            }

            // For div/sqrt path, traverse through expand_dims/reshape/squeeze to find actual sqrt
            Set<String> shapeOpsToRemove = new LinkedHashSet<>();
            Set<String> shapeVarsToRemove = new LinkedHashSet<>();
            if (divSqrtPath && !(normFactorOp.getOp() instanceof Sqrt)) {
                // Try to look through shape-preserving-for-broadcast ops
                SameDiffOp currentOp = normFactorOp;
                String currentVar = normFactorVar;
                boolean foundSqrt = false;
                for (int depth = 0; depth < 4; depth++) {
                    DifferentialFunction currentFunc = currentOp.getOp();
                    String currentType = opName(currentOp);
                    if ("expand_dims".equals(currentType) || "reshape".equals(currentType) || "squeeze".equals(currentType)) {
                        shapeOpsToRemove.add(currentOp.getName());
                        shapeVarsToRemove.add(currentVar);
                        String nextVar = currentOp.getInputsToOp().get(0);
                        SameDiffOp nextOp = producerOp(sd, helper, nextVar);
                        if (nextOp == null) break;
                        currentVar = nextVar;
                        currentOp = nextOp;
                        if (nextOp.getOp() instanceof Sqrt) {
                            normFactorOp = nextOp;
                            normFactorVar = currentVar;
                            foundSqrt = true;
                            break;
                        }
                    } else {
                        break;
                    }
                }
                if (!foundSqrt) {
                    return null;
                }
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
            DifferentialFunction addFunc = addOp.getOp();
            if (addFunc instanceof ScalarAdd) {
                // ScalarAdd: single input + scalar arg
                List<String> addInputs = addOp.getInputsToOp();
                if (addInputs == null || addInputs.size() != 1) return null;
                meanVar = addInputs.get(0);
                epsilon = scalarFromScalarOp(addOp);
            } else if (addFunc instanceof org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.AddOp) {
                // Pairwise AddOp: two inputs
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

            DifferentialFunction meanFunc = meanOp.getOp();
            if (meanFunc instanceof MeanSquare) {
                List<String> meanInputs = meanOp.getInputsToOp();
                if (meanInputs == null || meanInputs.isEmpty()) return null;
                meanInputVar = stripTrivial(sd, helper, meanInputs.get(0));
            } else if (meanFunc instanceof Mean) {
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
                DifferentialFunction squareFunc = squareOp.getOp();
                if (squareFunc instanceof MulOp) {
                    // x * x pattern (self-multiply = square)
                    List<String> sqInputs = squareOp.getInputsToOp();
                    if (sqInputs == null || sqInputs.size() != 2 || !sqInputs.get(0).equals(sqInputs.get(1))) {
                        return null;
                    }
                    meanInputVar = stripTrivial(sd, helper, sqInputs.get(0));
                } else if (squareFunc instanceof Pow || squareFunc instanceof PowPairwise
                           || squareFunc instanceof org.nd4j.linalg.api.ops.impl.transforms.custom.Pow) {
                    // pow(x, 2) pattern
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
                } else if (squareFunc instanceof Square) {
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
            // Use the pre-cast x so fused rms_norm produces output in the original type
            // (e.g., FP16 if the model is mixed-precision). The intermediate FP32 cast
            // and all internal ops will be removed; rms_norm handles precision internally.
            m.xVar = stripTrivial(sd, helper, xVar);
            m.gammaVar = gammaVar;
            m.epsilon = epsilon;

            m.opsToRemove.add(finalMul.getName());
            m.opsToRemove.addAll(castOpsToRemove);
            m.opsToRemove.add(normalizedOpName);
            if (normFactorHasOnlyConsumer) {
                m.opsToRemove.add(normFactorOp.getName());
            }
            m.opsToRemove.add(addOp.getName());
            m.opsToRemove.add(meanOp.getName());
            if (squareOp != null) {
                m.opsToRemove.add(squareOp.getName());
            }
            m.opsToRemove.addAll(shapeOpsToRemove);

            m.varsToRemove.add(finalOutputVar);
            m.varsToRemove.addAll(castVarsToRemove);
            m.varsToRemove.add(normalizedVar);
            if (normFactorHasOnlyConsumer) {
                m.varsToRemove.add(normFactorVar);
            }
            m.varsToRemove.add(addVar);
            m.varsToRemove.add(meanVar);
            if (squareVar != null) {
                m.varsToRemove.add(squareVar);
            }
            m.varsToRemove.addAll(shapeVarsToRemove);

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
                // NEVER strip through reshape — it changes tensor shape, causing fused rms_norm
                // to operate on wrong-shaped tensors (e.g., [B,L,512] instead of [B,L,2,256]).
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
            VariableType vt = v.getVariableType();
            if (vt != VariableType.CONSTANT && vt != VariableType.VARIABLE) {
                return false;
            }
            // Get actual array from the appropriate holder — getShape() is unreliable
            INDArray arr = null;
            if (vt == VariableType.CONSTANT) {
                arr = sd.getConstantArrays().getArray(varName);
            } else if (vt == VariableType.VARIABLE) {
                arr = sd.getVariablesArrays().getArray(varName);
            }
            if (arr == null) return false;
            long[] shape = arr.shape();
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
            DifferentialFunction f = op.getOp();
            return f instanceof Mean || f instanceof MeanSquare;
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

    /**
     * Fuse rms_norm(x, gamma, eps) followed by matmul(normalized, W) into a single
     * rms_norm_linear(x, gamma, W, eps) op.
     *
     * This pattern is extremely common in transformer models where every RMSNorm
     * is immediately followed by a linear projection (Q/K/V projections, FFN layers).
     * The fused op avoids materializing the intermediate normalized tensor.
     *
     * Detection starts from Mmul/TensorMmul and walks backward to find an RmsNorm producer.
     * Cast/identity ops between rms_norm and matmul are stripped (mixed-precision models
     * insert FP16/FP32 casts).
     */
    public static class FuseRMSNormLinearPattern implements Optimizer {

        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            Set<Class<? extends DifferentialFunction>> types = new HashSet<>();
            types.add(Mmul.class);
            types.add(TensorMmul.class);
            return types;
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (op == null || op.getOp() == null) {
                return false;
            }
            DifferentialFunction func = op.getOp();
            if (!(func instanceof Mmul) && !(func instanceof TensorMmul)) {
                return false;
            }

            List<String> matmulInputs = op.getInputsToOp();
            List<String> matmulOutputs = op.getOutputsOfOp();
            if (matmulInputs == null || matmulInputs.size() != 2 || matmulOutputs == null || matmulOutputs.isEmpty()) {
                return false;
            }

            String matmulOutputVar = matmulOutputs.get(0);

            // Try both input positions: rms_norm output could be either input to matmul
            for (int i = 0; i < 2; i++) {
                String rmsNormCandidateVar = matmulInputs.get(i);
                String weightVar = matmulInputs.get(1 - i);

                // Strip through cast/identity ops between rms_norm and matmul.
                // IMPORTANT: Do NOT strip precision-widening casts (e.g., HALF→FLOAT32).
                // These casts exist for numerical accuracy — the rms_norm_linear fused op
                // would operate in the narrower type, losing significant precision in the
                // vocabulary projection (248K-wide matmul amplifies half-precision errors).
                String strippedVar = rmsNormCandidateVar;
                Set<String> intermediateOps = new LinkedHashSet<>();
                Set<String> intermediateVars = new LinkedHashSet<>();
                boolean precisionWideningDetected = false;
                for (int depth = 0; depth < 4; depth++) {
                    SameDiffOp producer = producerOp(sd, helper, strippedVar);
                    if (producer == null || producer.getOp() == null) break;
                    String producerOpName = producer.getOp().opName();
                    if (producerOpName == null) break;
                    producerOpName = producerOpName.toLowerCase();
                    if ("cast".equals(producerOpName) || "identity".equals(producerOpName)) {
                        List<String> pInputs = producer.getInputsToOp();
                        if (pInputs == null || pInputs.isEmpty()) break;
                        // Reject precision-widening casts: if the cast output has more bits
                        // than its input, it exists for numerical precision — do not strip.
                        if ("cast".equals(producerOpName)) {
                            // Get the target type from the cast op's output variable dtype.
                            // The output variable of the cast is strippedVar (current variable).
                            SDVariable castOutputVar = sd.getVariable(strippedVar);
                            SDVariable castInputVar = sd.getVariable(pInputs.get(0));
                            if (castOutputVar != null && castInputVar != null) {
                                DataType inputDt = castInputVar.dataType();
                                DataType targetDt = castOutputVar.dataType();
                                if (inputDt != null && targetDt != null && targetDt != inputDt
                                        && inputDt.isFPType() && targetDt.isFPType()) {
                                    int inputBits = inputDt.width() * 8;
                                    int targetBits = targetDt.width() * 8;
                                    if (targetBits > inputBits) {
                                        // This is a precision-widening cast (e.g., HALF→FLOAT32).
                                        // Abort fusion to preserve numerical accuracy.
                                        log.debug("FuseRMSNormLinear: skipping fusion — precision-widening cast {} → {} detected", inputDt, targetDt);
                                        precisionWideningDetected = true;
                                        break;
                                    }
                                }
                            }
                        }
                        // Only strip if this intermediate has exactly 1 consumer
                        if (!hasOnlyConsumer(sd, helper, strippedVar, depth == 0 ? op.getName() : getConsumerOpName(intermediateOps))) {
                            break;
                        }
                        intermediateOps.add(producer.getName());
                        intermediateVars.add(strippedVar);
                        strippedVar = pInputs.get(0);
                    } else {
                        break;
                    }
                }
                if (precisionWideningDetected) {
                    continue;
                }

                // Check if the (possibly stripped) variable is produced by an RmsNorm op
                SameDiffOp rmsNormOp = producerOp(sd, helper, strippedVar);
                if (rmsNormOp == null || !(rmsNormOp.getOp() instanceof RmsNorm)) {
                    continue;
                }

                // Verify the rms_norm output feeds ONLY into this matmul (through intermediates)
                String rmsNormOutVar = strippedVar;
                String expectedConsumer = intermediateOps.isEmpty() ? op.getName() :
                        intermediateOps.iterator().next(); // first intermediate op consumes rms_norm output
                // If there are intermediates, rms_norm output should feed the first intermediate
                // If no intermediates, rms_norm output should feed the matmul directly
                // Verify the rms_norm output has EXACTLY 1 consumer (this matmul's chain).
                // Always check the LIVE graph, not the helper cache — prior optimizations
                // in this pass may have altered consumer lists. In models like Qwen where
                // Q/K/V projections share a normalization, rms_norm_N feeds multiple matmuls
                // and MUST NOT be fused+removed.
                {
                    Variable rmsOutVariable = sd.getVariables().get(rmsNormOutVar);
                    if (rmsOutVariable == null) continue;
                    List<String> rmsOutUsers = rmsOutVariable.getInputsForOp();
                    if (rmsOutUsers == null || rmsOutUsers.size() != 1) {
                        continue;
                    }
                    // Also verify the single consumer is the expected op in our chain
                    String expectedConsumerForRms = intermediateOps.isEmpty() ? op.getName()
                            : intermediateOps.iterator().next();
                    if (!expectedConsumerForRms.equals(rmsOutUsers.get(0))) {
                        continue;
                    }
                }

                // Extract RmsNorm components
                RmsNorm rmsNorm = (RmsNorm) rmsNormOp.getOp();
                double epsilon = rmsNorm.getEpsilon();
                List<String> rmsNormInputs = rmsNormOp.getInputsToOp();
                if (rmsNormInputs == null || rmsNormInputs.size() < 2) {
                    // Need both x and gamma
                    continue;
                }

                String xVar = rmsNormInputs.get(0);
                String gammaVar = rmsNormInputs.get(1);

                try {
                    SDVariable x = sd.getVariable(xVar);
                    SDVariable gamma = sd.getVariable(gammaVar);
                    SDVariable w = sd.getVariable(weightVar);
                    if (x == null || gamma == null || w == null) {
                        continue;
                    }

                    // Create fused RmsNormLinear op
                    SDVariable fused = sd.nn().rmsNormLinear(x, gamma, w, epsilon);

                    // Replace all uses of matmul output with fused output
                    OptimizationUtils.replaceOpInputsWith(sd, helper, matmulOutputVar, fused.name());

                    // Remove the old matmul op
                    OptimizationUtils.removeOp(sd, helper, op.getName());

                    // Remove intermediate cast/identity ops and vars
                    for (String intermediateOp : intermediateOps) {
                        OptimizationUtils.removeOp(sd, helper, intermediateOp);
                    }
                    for (String intermediateVar : intermediateVars) {
                        OptimizationUtils.removeVariable(sd, helper, intermediateVar);
                    }

                    // Remove the rms_norm op first. Its output variable cleanup is
                    // deferred until AFTER the matmul is removed (see below).
                    OptimizationUtils.removeOp(sd, helper, rmsNormOp.getName());

                    // Remove matmul output variable, then rename fused to match the
                    // original name (preserves downstream name lookups like "lm_logits").
                    // Must temporarily remove from outputs since removeVariable/removeOp
                    // guard against deleting registered graph outputs.
                    List<String> graphOutputs = sd.outputs();
                    boolean wasOutput = graphOutputs != null && graphOutputs.remove(matmulOutputVar);

                    // Re-attempt removing the matmul op now that its output is no longer
                    // a registered graph output. The first removeOp call (line 736) may
                    // have been refused because matmulOutputVar was still in outputs().
                    // Without this, the dangling matmul op persists in the graph alongside
                    // the fused rms_norm_linear, causing incorrect execution.
                    OptimizationUtils.removeOp(sd, helper, op.getName());

                    OptimizationUtils.removeVariable(sd, helper, matmulOutputVar);
                    sd.renameVariable(fused.name(), matmulOutputVar);
                    if (wasOutput) {
                        graphOutputs.add(matmulOutputVar);
                    }

                    // NOW clean up rmsNormOutVar — deferred to here because the matmul
                    // (which was rmsNormOutVar's only consumer) had to be removed first.
                    // Before matmul removal, rmsNormOutVar's consumer list was non-empty
                    // (containing matmul_186), preventing removal and leaving an orphaned
                    // variable in the graph that corrupts the DSP plan on CUDA.
                    if (!rmsNormOutVar.equals(xVar) && !rmsNormOutVar.equals(gammaVar)) {
                        Variable rmsVar = sd.getVariables().get(rmsNormOutVar);
                        List<String> remainingUsers = rmsVar != null ? rmsVar.getInputsForOp() : null;
                        if (remainingUsers == null || remainingUsers.isEmpty()) {
                            OptimizationUtils.removeVariable(sd, helper, rmsNormOutVar);
                        } else {
                            log.debug("FuseRMSNormLinear: keeping rms_norm output '{}' — still consumed by {} op(s)",
                                    rmsNormOutVar, remainingUsers.size());
                        }
                    }

                    log.info("Fused RMSNorm+Linear pattern: x={}, gamma={}, W={}, eps={}, output={}",
                            xVar, gammaVar, weightVar, epsilon, matmulOutputVar);
                    return true;
                } catch (Exception e) {
                    log.debug("Failed to fuse RMSNorm+Linear pattern at op {}: {}", op.getName(), e.getMessage());
                    return false;
                }
            }

            return false;
        }

        private String getConsumerOpName(Set<String> ops) {
            // Return the last op in the set (the one closest to rms_norm)
            String last = null;
            for (String s : ops) {
                last = s;
            }
            return last;
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
    }

    private static class RmsNormMatch {
        String xVar;
        String gammaVar;
        double epsilon;
        final Set<String> opsToRemove = new LinkedHashSet<>();
        final Set<String> varsToRemove = new LinkedHashSet<>();
    }
}

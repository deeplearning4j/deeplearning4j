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
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.autodiff.samediff.optimize.OptimizationHelper;
import org.nd4j.autodiff.samediff.optimize.Optimizer;
import org.nd4j.linalg.api.ops.impl.scalar.RectifiedLinear;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Max;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Min;
import org.nd4j.linalg.api.ops.impl.transforms.same.Abs;
import org.nd4j.linalg.api.ops.impl.transforms.same.Negative;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Exp;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Log;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.AddOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MulOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.SubOp;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.Set;

/**
 * Peephole optimizations — small, local rewrites that eliminate redundant
 * operations based on algebraic identities.
 *
 * <ul>
 *   <li><b>Idempotent ops</b>: {@code relu(relu(x)) → relu(x)},
 *       {@code abs(abs(x)) → abs(x)}</li>
 *   <li><b>Inverse function pairs</b>: {@code exp(log(x)) → x},
 *       {@code log(exp(x)) → x}</li>
 *   <li><b>Redundant binary same-input</b>: {@code min(x, x) → x},
 *       {@code max(x, x) → x}</li>
 *   <li><b>Negation propagation</b>: {@code neg(sub(x, y)) → sub(y, x)},
 *       {@code sub(x, neg(y)) → add(x, y)}</li>
 * </ul>
 */
@Slf4j
public class PeepholeOptimizations extends BaseOptimizerSet {

    // ─── Helper: bypass an op when its input was produced by the same op type ──

    /**
     * For idempotent unary ops (f(f(x)) = f(x)), replace the outer
     * with its inner's output.
     */
    private static boolean bypassIfProducerMatches(SameDiff sd, OptimizationHelper helper,
                                                    SameDiffOp op, Class<?> opType) {
        List<String> inputs = op.getInputsToOp();
        if (inputs == null || inputs.size() != 1) return false;
        List<String> outputs = op.getOutputsOfOp();
        if (outputs == null || outputs.isEmpty()) return false;

        String inputVar = inputs.get(0);
        String outputVar = outputs.get(0);

        Variable inputVariable = sd.getVariables().get(inputVar);
        if (inputVariable == null) return false;
        String producerOpName = inputVariable.getOutputOfOp();
        if (producerOpName == null) return false;

        SameDiffOp producerOp = sd.getOps().get(producerOpName);
        if (producerOp == null || !opType.isInstance(producerOp.getOp())) return false;

        // f(f(x)) → f(x): the outer op's output becomes the inner op's output
        // i.e., just bypass the outer op
        log.debug("Idempotent elimination: {}({}) -> {}", op.getOp().opName(), inputVar, inputVar);

        OptimizationUtils.replaceOpInputsWith(sd, helper, outputVar, inputVar);

        List<String> graphOutputs = sd.outputs();
        if (graphOutputs != null) {
            for (int i = 0; i < graphOutputs.size(); i++) {
                if (graphOutputs.get(i).equals(outputVar)) {
                    graphOutputs.set(i, inputVar);
                }
            }
        }

        OptimizationUtils.removeOp(sd, helper, op.getName());
        OptimizationUtils.removeVariable(sd, helper, outputVar);
        return true;
    }

    /**
     * For inverse function pairs (g(f(x)) = x), bypass both ops
     * and point consumers at the original input.
     */
    private static boolean bypassInversePair(SameDiff sd, OptimizationHelper helper,
                                              SameDiffOp outerOp, Class<?> innerOpType) {
        List<String> outerInputs = outerOp.getInputsToOp();
        if (outerInputs == null || outerInputs.size() != 1) return false;
        List<String> outerOutputs = outerOp.getOutputsOfOp();
        if (outerOutputs == null || outerOutputs.isEmpty()) return false;

        String intermediateVar = outerInputs.get(0);
        String outerOutputVar = outerOutputs.get(0);

        Variable intermediateVariable = sd.getVariables().get(intermediateVar);
        if (intermediateVariable == null) return false;
        String producerOpName = intermediateVariable.getOutputOfOp();
        if (producerOpName == null) return false;

        SameDiffOp producerOp = sd.getOps().get(producerOpName);
        if (producerOp == null || !innerOpType.isInstance(producerOp.getOp())) return false;

        // The intermediate must only be consumed by the outer op
        List<String> intermediateUsers = intermediateVariable.getInputsForOp();
        if (intermediateUsers == null || intermediateUsers.size() != 1) return false;

        // Get the original input before the inner op
        List<String> innerInputs = producerOp.getInputsToOp();
        if (innerInputs == null || innerInputs.size() != 1) return false;
        String originalInput = innerInputs.get(0);

        log.debug("Inverse pair elimination: {}({}({})) -> {}",
                outerOp.getOp().opName(), producerOp.getOp().opName(), originalInput, originalInput);

        OptimizationUtils.replaceOpInputsWith(sd, helper, outerOutputVar, originalInput);

        List<String> graphOutputs = sd.outputs();
        if (graphOutputs != null) {
            for (int i = 0; i < graphOutputs.size(); i++) {
                if (graphOutputs.get(i).equals(outerOutputVar)) {
                    graphOutputs.set(i, originalInput);
                }
            }
        }

        OptimizationUtils.removeOp(sd, helper, outerOp.getName());
        OptimizationUtils.removeVariable(sd, helper, outerOutputVar);

        // Clean up inner op if it has no remaining consumers
        Variable intermediateAfter = sd.getVariables().get(intermediateVar);
        if (intermediateAfter != null) {
            List<String> remaining = intermediateAfter.getInputsForOp();
            if (remaining == null || remaining.isEmpty()) {
                OptimizationUtils.removeOp(sd, helper, producerOpName);
                OptimizationUtils.removeVariable(sd, helper, intermediateVar);
            }
        }

        return true;
    }

    // ─── Idempotent ops ────────────────────────────────────────────────────

    /**
     * relu(relu(x)) → relu(x). ReLU is idempotent: max(0, max(0, x)) = max(0, x).
     */
    public static class IdempotentRelu implements Optimizer {
        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return Set.of(RectifiedLinear.class);
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof RectifiedLinear)) return false;
            return bypassIfProducerMatches(sd, helper, op, RectifiedLinear.class);
        }
    }

    /**
     * abs(abs(x)) → abs(x). Absolute value is idempotent.
     */
    public static class IdempotentAbs implements Optimizer {
        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return Set.of(Abs.class);
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof Abs)) return false;
            return bypassIfProducerMatches(sd, helper, op, Abs.class);
        }
    }

    // ─── Inverse function pairs ────────────────────────────────────────────

    /**
     * exp(log(x)) → x. Exponential is the inverse of natural log.
     */
    public static class InverseExpLog implements Optimizer {
        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return Set.of(Exp.class);
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof Exp)) return false;
            return bypassInversePair(sd, helper, op, Log.class);
        }
    }

    /**
     * log(exp(x)) → x. Natural log is the inverse of exponential.
     */
    public static class InverseLogExp implements Optimizer {
        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return Set.of(Log.class);
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof Log)) return false;
            return bypassInversePair(sd, helper, op, Exp.class);
        }
    }

    // ─── Redundant binary same-input ───────────────────────────────────────

    /**
     * min(x, x) → x. Minimum of identical inputs is identity.
     */
    public static class RedundantMin implements Optimizer {
        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return Set.of(Min.class);
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof Min)) return false;
            return eliminateSameInputBinary(sd, helper, op);
        }
    }

    /**
     * max(x, x) → x. Maximum of identical inputs is identity.
     */
    public static class RedundantMax implements Optimizer {
        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return Set.of(Max.class);
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof Max)) return false;
            return eliminateSameInputBinary(sd, helper, op);
        }
    }

    private static boolean eliminateSameInputBinary(SameDiff sd, OptimizationHelper helper, SameDiffOp op) {
        List<String> inputs = op.getInputsToOp();
        if (inputs == null || inputs.size() != 2) return false;
        if (!inputs.get(0).equals(inputs.get(1))) return false;

        List<String> outputs = op.getOutputsOfOp();
        if (outputs == null || outputs.isEmpty()) return false;

        String inputVar = inputs.get(0);
        String outputVar = outputs.get(0);

        log.debug("Redundant {} elimination: {}({}, {}) -> {}",
                op.getOp().opName(), op.getOp().opName(), inputVar, inputVar, inputVar);

        OptimizationUtils.replaceOpInputsWith(sd, helper, outputVar, inputVar);

        List<String> graphOutputs = sd.outputs();
        if (graphOutputs != null) {
            for (int i = 0; i < graphOutputs.size(); i++) {
                if (graphOutputs.get(i).equals(outputVar)) {
                    graphOutputs.set(i, inputVar);
                }
            }
        }

        OptimizationUtils.removeOp(sd, helper, op.getName());
        OptimizationUtils.removeVariable(sd, helper, outputVar);
        return true;
    }

    // ─── Negation propagation ──────────────────────────────────────────────

    /**
     * sub(x, neg(y)) → add(x, y).
     * Avoids the extra negation op.
     */
    public static class SubNegToAdd implements Optimizer {
        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return Set.of(SubOp.class);
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof SubOp)) return false;

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.size() != 2) return false;
            List<String> outputs = op.getOutputsOfOp();
            if (outputs == null || outputs.isEmpty()) return false;

            String xVar = inputs.get(0);
            String negYVar = inputs.get(1);
            String outputVar = outputs.get(0);

            // Check if second input is produced by a neg op
            Variable negYVariable = sd.getVariables().get(negYVar);
            if (negYVariable == null) return false;
            String producerOpName = negYVariable.getOutputOfOp();
            if (producerOpName == null) return false;

            SameDiffOp producerOp = sd.getOps().get(producerOpName);
            if (producerOp == null || !(producerOp.getOp() instanceof Negative)) return false;

            // The neg output must only be consumed by this sub
            List<String> negUsers = negYVariable.getInputsForOp();
            if (negUsers == null || negUsers.size() != 1) return false;

            List<String> negInputs = producerOp.getInputsToOp();
            if (negInputs == null || negInputs.size() != 1) return false;
            String yVar = negInputs.get(0);

            log.debug("Negation propagation: sub({}, neg({})) -> add({}, {})", xVar, yVar, xVar, yVar);

            // Rewire: sub(x, neg(y)) → add(x, y)
            // Create add(x, y) with the output name
            sd.getVariable(xVar).add(outputVar + "__peep_add", sd.getVariable(yVar));
            String addOutputName = outputVar + "__peep_add";

            OptimizationUtils.replaceOpInputsWith(sd, helper, outputVar, addOutputName);

            List<String> graphOutputs = sd.outputs();
            boolean wasOutput = graphOutputs != null && graphOutputs.remove(outputVar);
            OptimizationUtils.removeOp(sd, helper, op.getName());
            OptimizationUtils.removeVariable(sd, helper, outputVar);
            sd.renameVariable(addOutputName, outputVar);
            if (wasOutput) graphOutputs.add(outputVar);

            // Clean up neg if no remaining consumers
            Variable negAfter = sd.getVariables().get(negYVar);
            if (negAfter != null) {
                List<String> remaining = negAfter.getInputsForOp();
                if (remaining == null || remaining.isEmpty()) {
                    OptimizationUtils.removeOp(sd, helper, producerOpName);
                    OptimizationUtils.removeVariable(sd, helper, negYVar);
                }
            }

            return true;
        }
    }

    // ─── add(x, neg(y)) → sub(x, y) ──────────────────────────────────────

    /**
     * add(x, neg(y)) → sub(x, y).
     * Complement of SubNegToAdd.
     */
    public static class AddNegToSub implements Optimizer {
        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return Set.of(AddOp.class);
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof AddOp)) return false;

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.size() != 2) return false;
            List<String> outputs = op.getOutputsOfOp();
            if (outputs == null || outputs.isEmpty()) return false;

            String outputVar = outputs.get(0);

            // Check either input for neg producer
            for (int negIdx = 0; negIdx < 2; negIdx++) {
                String candidateVar = inputs.get(negIdx);
                String otherVar = inputs.get(1 - negIdx);

                Variable candidateVariable = sd.getVariables().get(candidateVar);
                if (candidateVariable == null) continue;
                String producerOpName = candidateVariable.getOutputOfOp();
                if (producerOpName == null) continue;

                SameDiffOp producerOp = sd.getOps().get(producerOpName);
                if (producerOp == null || !(producerOp.getOp() instanceof Negative)) continue;

                List<String> negUsers = candidateVariable.getInputsForOp();
                if (negUsers == null || negUsers.size() != 1) continue;

                List<String> negInputs = producerOp.getInputsToOp();
                if (negInputs == null || negInputs.size() != 1) continue;
                String yVar = negInputs.get(0);

                log.debug("Negation propagation: add({}, neg({})) -> sub({}, {})",
                        otherVar, yVar, otherVar, yVar);

                // Create sub(other, y)
                String subName = outputVar + "__peep_sub";
                sd.getVariable(otherVar).sub(subName, sd.getVariable(yVar));

                OptimizationUtils.replaceOpInputsWith(sd, helper, outputVar, subName);
                List<String> graphOutputs = sd.outputs();
                boolean wasOutput = graphOutputs != null && graphOutputs.remove(outputVar);
                OptimizationUtils.removeOp(sd, helper, op.getName());
                OptimizationUtils.removeVariable(sd, helper, outputVar);
                sd.renameVariable(subName, outputVar);
                if (wasOutput) graphOutputs.add(outputVar);

                // Clean up neg if no remaining consumers
                Variable negAfter = sd.getVariables().get(candidateVar);
                if (negAfter != null) {
                    List<String> remaining = negAfter.getInputsForOp();
                    if (remaining == null || remaining.isEmpty()) {
                        OptimizationUtils.removeOp(sd, helper, producerOpName);
                        OptimizationUtils.removeVariable(sd, helper, candidateVar);
                    }
                }
                return true;
            }
            return false;
        }
    }

    // ─── mul(x, -1) → neg(x) ─────────────────────────────────────────────

    /**
     * mul(x, -1) → neg(x) or mul(-1, x) → neg(x).
     * Replaces multiplication by scalar -1 with a negation.
     */
    public static class MulNegOneToNeg implements Optimizer {
        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return Set.of(MulOp.class);
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof MulOp)) return false;

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.size() != 2) return false;
            List<String> outputs = op.getOutputsOfOp();
            if (outputs == null || outputs.isEmpty()) return false;

            String outputVar = outputs.get(0);

            // Check either input for scalar -1 constant
            for (int constIdx = 0; constIdx < 2; constIdx++) {
                String constVarName = inputs.get(constIdx);
                String otherVarName = inputs.get(1 - constIdx);

                SDVariable constVar = sd.getVariable(constVarName);
                if (constVar == null || constVar.getVariableType() != VariableType.CONSTANT) continue;

                INDArray arr = constantArrays.getArray(constVarName);
                if (arr == null) arr = sd.getArrForVarName(constVarName);
                if (arr == null) continue;

                // Must be scalar or [1] with value -1
                if (arr.length() != 1) continue;
                if (Math.abs(arr.getDouble(0) - (-1.0)) > 1e-10) continue;

                log.debug("MulNegOne: mul({}, -1) -> neg({})", otherVarName, otherVarName);

                // Create neg(other)
                String negName = outputVar + "__peep_neg";
                sd.getVariable(otherVarName).neg(negName);

                OptimizationUtils.replaceOpInputsWith(sd, helper, outputVar, negName);
                List<String> graphOutputs = sd.outputs();
                boolean wasOutput = graphOutputs != null && graphOutputs.remove(outputVar);
                OptimizationUtils.removeOp(sd, helper, op.getName());
                OptimizationUtils.removeVariable(sd, helper, outputVar);
                sd.renameVariable(negName, outputVar);
                if (wasOutput) graphOutputs.add(outputVar);
                return true;
            }
            return false;
        }
    }

    // ─── sub(0, x) → neg(x) ──────────────────────────────────────────────

    /**
     * sub(0, x) → neg(x).
     * Subtraction from zero is negation.
     */
    public static class SubZeroToNeg implements Optimizer {
        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return Set.of(SubOp.class);
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof SubOp)) return false;

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.size() != 2) return false;
            List<String> outputs = op.getOutputsOfOp();
            if (outputs == null || outputs.isEmpty()) return false;

            String firstVarName = inputs.get(0);
            String secondVarName = inputs.get(1);
            String outputVar = outputs.get(0);

            // First input must be scalar 0 constant
            SDVariable firstVar = sd.getVariable(firstVarName);
            if (firstVar == null || firstVar.getVariableType() != VariableType.CONSTANT) return false;

            INDArray arr = constantArrays.getArray(firstVarName);
            if (arr == null) arr = sd.getArrForVarName(firstVarName);
            if (arr == null) return false;

            if (arr.length() != 1) return false;
            if (Math.abs(arr.getDouble(0)) > 1e-10) return false;

            log.debug("SubZero: sub(0, {}) -> neg({})", secondVarName, secondVarName);

            // Create neg(second)
            String negName = outputVar + "__peep_neg";
            sd.getVariable(secondVarName).neg(negName);

            OptimizationUtils.replaceOpInputsWith(sd, helper, outputVar, negName);
            List<String> graphOutputs = sd.outputs();
            boolean wasOutput = graphOutputs != null && graphOutputs.remove(outputVar);
            OptimizationUtils.removeOp(sd, helper, op.getName());
            OptimizationUtils.removeVariable(sd, helper, outputVar);
            sd.renameVariable(negName, outputVar);
            if (wasOutput) graphOutputs.add(outputVar);
            return true;
        }
    }
}

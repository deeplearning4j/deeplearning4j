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
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastTo;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.AddOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MulOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.SubOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.DivOp;
import org.nd4j.linalg.api.ops.impl.transforms.same.Negative;

import java.util.List;
import java.util.Set;

/**
 * Broadcast elimination optimizations.
 *
 * <p>Removes explicit {@code broadcast_to} ops when every downstream consumer
 * already handles broadcasting natively. Most arithmetic ops (add, sub, mul, div)
 * and element-wise transforms broadcast internally via Nd4j's shape broadcasting
 * rules, making an explicit {@code broadcast_to} redundant overhead.</p>
 *
 * <p>Also eliminates redundant broadcast_to ops where the input shape already
 * matches the target shape (no-op broadcasts).</p>
 */
@Slf4j
public class BroadcastEliminationOptimizations extends BaseOptimizerSet {

    /** Ops that handle broadcasting internally and don't need explicit broadcast_to. */
    private static final Set<Class<? extends DifferentialFunction>> BROADCAST_AWARE_OPS = Set.of(
            AddOp.class, SubOp.class, MulOp.class, DivOp.class
    );

    /**
     * Remove broadcast_to(x, shape) when all consumers handle broadcasting natively.
     *
     * <p>Pattern: broadcast_to(x, shape) → consumers that broadcast internally.
     * The broadcast_to output is replaced with its first input (the data tensor).</p>
     */
    public static class RemoveRedundantBroadcast implements Optimizer {

        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return Set.of(BroadcastTo.class);
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof BroadcastTo)) {
                return false;
            }

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.isEmpty()) {
                return false;
            }

            List<String> outputs = op.getOutputsOfOp();
            if (outputs == null || outputs.isEmpty()) {
                return false;
            }

            String broadcastOutput = outputs.get(0);
            String dataInput = inputs.get(0);

            // Check all consumers of the broadcast output
            Variable outputVar = sd.getVariables().get(broadcastOutput);
            if (outputVar == null) return false;

            List<String> consumers = outputVar.getInputsForOp();
            if (consumers == null || consumers.isEmpty()) {
                // No consumers -- the broadcast is dead code. DCE handles this,
                // but we can safely remove it too.
                return eliminateBroadcast(sd, helper, op, broadcastOutput, dataInput);
            }

            // Check if every consumer handles broadcasting internally
            for (String consumerOpName : consumers) {
                SameDiffOp consumerOp = sd.getOps().get(consumerOpName);
                if (consumerOp == null || consumerOp.getOp() == null) {
                    return false;
                }

                boolean isBroadcastAware = false;
                for (Class<? extends DifferentialFunction> cls : BROADCAST_AWARE_OPS) {
                    if (cls.isInstance(consumerOp.getOp())) {
                        isBroadcastAware = true;
                        break;
                    }
                }

                if (!isBroadcastAware) {
                    return false;
                }
            }

            return eliminateBroadcast(sd, helper, op, broadcastOutput, dataInput);
        }

        private boolean eliminateBroadcast(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                           String broadcastOutput, String dataInput) {
            log.debug("Eliminating redundant broadcast_to: {} -> {}", broadcastOutput, dataInput);

            // Replace all uses of broadcast output with the data input
            OptimizationUtils.replaceOpInputsWith(sd, helper, broadcastOutput, dataInput);

            // Update graph outputs if broadcast output was one
            List<String> graphOutputs = sd.outputs();
            if (graphOutputs != null) {
                for (int i = 0; i < graphOutputs.size(); i++) {
                    if (graphOutputs.get(i).equals(broadcastOutput)) {
                        graphOutputs.set(i, dataInput);
                    }
                }
            }

            // Remove the broadcast op
            OptimizationUtils.removeOp(sd, helper, op.getName());
            OptimizationUtils.removeVariable(sd, helper, broadcastOutput);
            return true;
        }
    }

    /**
     * Eliminate double negation: neg(neg(x)) → x.
     *
     * <p>Checks if a neg op's input is itself the output of another neg op.
     * If so, bypasses both negations.</p>
     */
    public static class EliminateDoubleNegation implements Optimizer {

        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return Set.of(Negative.class);
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof Negative)) {
                return false;
            }

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.size() != 1) {
                return false;
            }

            List<String> outputs = op.getOutputsOfOp();
            if (outputs == null || outputs.isEmpty()) {
                return false;
            }

            String innerVarName = inputs.get(0);
            Variable innerVar = sd.getVariables().get(innerVarName);
            if (innerVar == null) return false;

            // Check if the input was produced by another neg
            String producerOpName = innerVar.getOutputOfOp();
            if (producerOpName == null) return false;

            SameDiffOp producerOp = sd.getOps().get(producerOpName);
            if (producerOp == null || !(producerOp.getOp() instanceof Negative)) {
                return false;
            }

            // Found neg(neg(x)). Get x.
            List<String> innerInputs = producerOp.getInputsToOp();
            if (innerInputs == null || innerInputs.size() != 1) {
                return false;
            }

            String originalInput = innerInputs.get(0);
            String outerOutput = outputs.get(0);

            log.debug("Eliminating double negation: neg(neg({})) -> {}", originalInput, originalInput);

            // Replace all uses of the outer neg output with the original input
            OptimizationUtils.replaceOpInputsWith(sd, helper, outerOutput, originalInput);

            // Update graph outputs — track wasOutput so we can leave outerOutput as a
            // zombie variable when it was a registered graph output. A zombie prevents
            // restoreRequiredOutputNames from re-inserting an unwanted identity op.
            List<String> graphOutputs = sd.outputs();
            boolean wasOutput = false;
            if (graphOutputs != null) {
                for (int i = 0; i < graphOutputs.size(); i++) {
                    if (graphOutputs.get(i).equals(outerOutput)) {
                        graphOutputs.set(i, originalInput);
                        wasOutput = true;
                    }
                }
            }

            // Remove the outer neg op
            OptimizationUtils.removeOp(sd, helper, op.getName());
            if (!wasOutput) OptimizationUtils.removeVariable(sd, helper, outerOutput);

            // If the inner neg now has no consumers, remove it too
            Variable innerVarAfter = sd.getVariables().get(innerVarName);
            if (innerVarAfter != null) {
                List<String> remainingConsumers = innerVarAfter.getInputsForOp();
                if (remainingConsumers == null || remainingConsumers.isEmpty()) {
                    OptimizationUtils.removeOp(sd, helper, producerOpName);
                    OptimizationUtils.removeVariable(sd, helper, innerVarName);
                }
            }

            return true;
        }
    }

    /**
     * Canonicalize commutative ops so constants are always on the right.
     * This normalizes the graph for downstream constant-folding and algebraic
     * simplification passes.
     *
     * <p>Pattern: add(const, x) → add(x, const), mul(const, x) → mul(x, const).</p>
     */
    public static class CanonicalizeCommutativeOps implements Optimizer {

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            DifferentialFunction df = op.getOp();
            if (df == null) return false;

            // Only commutative ops: add, mul
            if (!(df instanceof AddOp) && !(df instanceof MulOp)) {
                return false;
            }

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.size() != 2) {
                return false;
            }

            String left = inputs.get(0);
            String right = inputs.get(1);

            // If left is constant and right is not, swap them
            var leftVar = sd.getVariable(left);
            var rightVar = sd.getVariable(right);
            if (leftVar == null || rightVar == null) return false;

            boolean leftConst = leftVar.isConstant();
            boolean rightConst = rightVar.isConstant();

            if (leftConst && !rightConst) {
                inputs.set(0, right);
                inputs.set(1, left);
                log.debug("Canonicalized commutative op {}: swapped {} and {}",
                        op.getName(), left, right);
                return true;
            }

            return false;
        }
    }
}

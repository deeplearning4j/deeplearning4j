/*
 *  ******************************************************************************
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
import org.nd4j.linalg.api.ops.impl.shape.Concat;
import org.nd4j.linalg.api.ops.impl.shape.Split;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Optimizations for concat and split operations.
 *
 * <ul>
 *   <li><b>FlattenNestedConcat</b>: {@code concat(concat(a, b), c, axis=N) → concat(a, b, c, axis=N)}
 *       when inner and outer concats use the same axis and the inner result has only one consumer.</li>
 *   <li><b>EliminateConcatSplit</b>: {@code split(concat(a, b), numSplit=2, axis=N) → [a, b]}
 *       when the split undoes the concat exactly (same axis, same number of pieces).</li>
 * </ul>
 */
@Slf4j
public class ConcatSplitOptimizations extends BaseOptimizerSet {

    /**
     * Flatten nested concat ops on the same axis.
     *
     * <p>{@code concat(concat(a, b, axis=N), c, axis=N) → concat(a, b, c, axis=N)}</p>
     *
     * <p>This reduces the number of concat ops and avoids an intermediate
     * allocation for the inner concat result. Only applies when the inner
     * concat's output has exactly one consumer (the outer concat).</p>
     */
    public static class FlattenNestedConcat implements Optimizer {
        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return Set.of(Concat.class);
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof Concat)) return false;

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.size() < 2) return false;

            // Get outer concat axis
            long[] outerIArgs = ((Concat) op.getOp()).iArgs();
            if (outerIArgs == null || outerIArgs.length == 0) return false;
            long outerAxis = outerIArgs[0];

            boolean applied = false;

            // Check each input: if it's produced by a Concat on the same axis
            // with a single consumer, inline its inputs
            for (int i = 0; i < inputs.size(); i++) {
                String inputVar = inputs.get(i);
                Variable inputVariable = sd.getVariables().get(inputVar);
                if (inputVariable == null) continue;

                String producerOpName = inputVariable.getOutputOfOp();
                if (producerOpName == null) continue;

                SameDiffOp producerOp = sd.getOps().get(producerOpName);
                if (producerOp == null || !(producerOp.getOp() instanceof Concat)) continue;

                // Inner concat must have the same axis
                long[] innerIArgs = ((Concat) producerOp.getOp()).iArgs();
                if (innerIArgs == null || innerIArgs.length == 0) continue;
                if (innerIArgs[0] != outerAxis) continue;

                // Inner result must only be consumed by this outer concat
                List<String> innerUsers = inputVariable.getInputsForOp();
                if (innerUsers == null || innerUsers.size() != 1) continue;

                List<String> innerInputs = producerOp.getInputsToOp();
                if (innerInputs == null || innerInputs.isEmpty()) continue;

                log.debug("FlattenNestedConcat: inlining inner concat at input {} with {} inputs",
                        i, innerInputs.size());

                // Replace the single inner-concat output with all inner-concat inputs
                // Remove the inner result from the outer input list and insert inner inputs
                inputs.remove(i);
                inputs.addAll(i, new ArrayList<>(innerInputs));

                // Register all inner inputs as consumers of the outer concat op
                for (String innerInput : innerInputs) {
                    Variable innerInputVar = sd.getVariables().get(innerInput);
                    if (innerInputVar != null && innerInputVar.getInputsForOp() != null
                            && !innerInputVar.getInputsForOp().contains(op.getName())) {
                        innerInputVar.getInputsForOp().add(op.getName());
                    }
                }

                // Remove inner concat op and its output variable
                OptimizationUtils.removeOp(sd, helper, producerOpName);
                OptimizationUtils.removeVariable(sd, helper, inputVar);

                applied = true;
                break; // Re-scan after modification (optimizer will be called again)
            }

            return applied;
        }
    }

    /**
     * Eliminate split(concat(a, b, ...), numSplit=N, axis) when the split
     * perfectly undoes the concat (same axis, same number of pieces, and each
     * concat input has equal size along the split axis).
     *
     * <p>Replaces each split output with the corresponding concat input directly.</p>
     */
    public static class EliminateConcatSplit implements Optimizer {
        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return Set.of(Split.class);
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof Split)) return false;

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.isEmpty()) return false;

            // Split iArgs: [numSplit, splitDim]
            long[] splitIArgs = ((Split) op.getOp()).iArgs();
            if (splitIArgs == null || splitIArgs.length < 2) return false;
            int numSplit = (int) splitIArgs[0];
            long splitDim = splitIArgs[1];

            // The input to split must be produced by a concat
            String splitInput = inputs.get(0);
            Variable splitInputVar = sd.getVariables().get(splitInput);
            if (splitInputVar == null) return false;

            String concatOpName = splitInputVar.getOutputOfOp();
            if (concatOpName == null) return false;

            SameDiffOp concatOp = sd.getOps().get(concatOpName);
            if (concatOp == null || !(concatOp.getOp() instanceof Concat)) return false;

            // Concat must have the same axis
            long[] concatIArgs = ((Concat) concatOp.getOp()).iArgs();
            if (concatIArgs == null || concatIArgs.length == 0) return false;
            if (concatIArgs[0] != splitDim) return false;

            List<String> concatInputs = concatOp.getInputsToOp();
            if (concatInputs == null || concatInputs.size() != numSplit) return false;

            // Split outputs must match concat inputs count
            List<String> splitOutputs = op.getOutputsOfOp();
            if (splitOutputs == null || splitOutputs.size() != numSplit) return false;

            // The concat result must only be consumed by this split
            List<String> concatUsers = splitInputVar.getInputsForOp();
            if (concatUsers == null || concatUsers.size() != 1) return false;

            // Verify that concat inputs have equal sizes along the split dimension
            // (split divides evenly, so all pieces must be the same size)
            for (String concatInput : concatInputs) {
                org.nd4j.autodiff.samediff.SDVariable sdv = sd.getVariable(concatInput);
                if (sdv == null) return false;
                long[] shape = sdv.getShape();
                if (shape == null) return false;
                int axis = (int) splitDim;
                if (axis < 0) axis += shape.length;
                if (axis < 0 || axis >= shape.length) return false;
                if (shape[axis] < 0) return false; // Dynamic — can't verify
            }

            // Check that all concat inputs have the same size along the split axis
            long[] firstShape = sd.getVariable(concatInputs.get(0)).getShape();
            int axis = (int) splitDim;
            if (axis < 0) axis += firstShape.length;
            long expectedSize = firstShape[axis];
            for (int i = 1; i < concatInputs.size(); i++) {
                long[] shape = sd.getVariable(concatInputs.get(i)).getShape();
                int ax = (int) splitDim;
                if (ax < 0) ax += shape.length;
                if (shape[ax] != expectedSize) return false; // Unequal sizes — can't be a uniform split
            }

            log.debug("EliminateConcatSplit: split(concat({} inputs), numSplit={}) eliminated",
                    concatInputs.size(), numSplit);

            // Replace each split output with the corresponding concat input
            for (int i = 0; i < numSplit; i++) {
                String splitOutput = splitOutputs.get(i);
                String concatInput = concatInputs.get(i);

                OptimizationUtils.replaceOpInputsWith(sd, helper, splitOutput, concatInput);

                List<String> graphOutputs = sd.outputs();
                if (graphOutputs != null) {
                    for (int j = 0; j < graphOutputs.size(); j++) {
                        if (graphOutputs.get(j).equals(splitOutput)) {
                            graphOutputs.set(j, concatInput);
                        }
                    }
                }
            }

            // Remove split op and its output variables
            OptimizationUtils.removeOp(sd, helper, op.getName());
            for (String splitOutput : splitOutputs) {
                OptimizationUtils.removeVariable(sd, helper, splitOutput);
            }

            // Remove concat op and its output variable (only consumed by the now-removed split)
            OptimizationUtils.removeOp(sd, helper, concatOpName);
            OptimizationUtils.removeVariable(sd, helper, splitInput);

            return true;
        }
    }
}

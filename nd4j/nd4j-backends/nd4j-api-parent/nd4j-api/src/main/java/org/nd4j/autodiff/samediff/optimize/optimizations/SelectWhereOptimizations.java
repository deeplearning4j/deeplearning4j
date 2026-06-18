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
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.optimize.OptimizationHelper;
import org.nd4j.autodiff.samediff.optimize.Optimizer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.controlflow.Select;
import org.nd4j.linalg.api.ops.impl.controlflow.Where;
import org.nd4j.linalg.api.ops.impl.controlflow.WhereNumpy;

import java.util.List;
import java.util.Set;

/**
 * Optimizations for select/where operations with constant conditions.
 *
 * <ul>
 *   <li><b>ConstantConditionSelect</b>: {@code select(true, x, y) → x} or
 *       {@code select(false, x, y) → y} when the condition is a compile-time constant.</li>
 *   <li><b>ConstantConditionWhereNumpy</b>: same optimization for the numpy-style where op.</li>
 * </ul>
 */
@Slf4j
public class SelectWhereOptimizations extends BaseOptimizerSet {

    /**
     * Check if all elements of a boolean/numeric constant are true (nonzero).
     */
    private static Boolean isConstantCondition(SameDiff sd, ArrayHolder constantArrays, String varName) {
        SDVariable sdVar = sd.getVariable(varName);
        if (sdVar == null || sdVar.getVariableType() != VariableType.CONSTANT) return null;

        INDArray arr = constantArrays.getArray(varName);
        if (arr == null) arr = sd.getArrForVarName(varName);
        if (arr == null) return null;

        // Only optimize small constant conditions to avoid scanning large masks
        if (arr.length() > 10000) return null;

        boolean allTrue = true;
        boolean allFalse = true;
        for (long i = 0; i < arr.length(); i++) {
            double v = arr.getDouble(i);
            if (v == 0.0) {
                allTrue = false;
            } else {
                allFalse = false;
            }
        }

        if (allTrue) return Boolean.TRUE;
        if (allFalse) return Boolean.FALSE;
        return null; // Mixed — can't simplify
    }

    /**
     * Replace all downstream consumers of an op's output with a replacement variable,
     * then remove the op and its output variable.
     */
    private static void replaceWithBranch(SameDiff sd, OptimizationHelper helper,
                                          SameDiffOp op, String outputVar, String replacement) {
        OptimizationUtils.replaceOpInputsWith(sd, helper, outputVar, replacement);

        List<String> graphOutputs = sd.outputs();
        if (graphOutputs != null) {
            for (int i = 0; i < graphOutputs.size(); i++) {
                if (graphOutputs.get(i).equals(outputVar)) {
                    graphOutputs.set(i, replacement);
                }
            }
        }

        OptimizationUtils.removeOp(sd, helper, op.getName());
        OptimizationUtils.removeVariable(sd, helper, outputVar);
    }

    /**
     * {@code select(cond, x, y)} with constant all-true condition → x,
     * with constant all-false condition → y.
     *
     * <p>Select inputs are [condition, x, y]. When the condition tensor is a
     * compile-time constant that is uniformly true or false, the entire select
     * can be replaced by the corresponding branch.</p>
     */
    public static class ConstantConditionSelect implements Optimizer {
        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return Set.of(Select.class);
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof Select)) return false;

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.size() != 3) return false;
            List<String> outputs = op.getOutputsOfOp();
            if (outputs == null || outputs.isEmpty()) return false;

            String condVar = inputs.get(0);
            String xVar = inputs.get(1);
            String yVar = inputs.get(2);
            String outputVar = outputs.get(0);

            Boolean condValue = isConstantCondition(sd, constantArrays, condVar);
            if (condValue == null) return false;

            String replacement = condValue ? xVar : yVar;
            log.debug("ConstantConditionSelect: select({}, {}, {}) -> {} (condition all-{})",
                    condVar, xVar, yVar, replacement, condValue);

            replaceWithBranch(sd, helper, op, outputVar, replacement);
            return true;
        }
    }

    /**
     * {@code where(cond, x, y)} with constant all-true condition → x,
     * with constant all-false condition → y.
     *
     * <p>Where inputs are [condition, x, y]. This is the op created by
     * {@code sd.where(name, x, y, condition)}.</p>
     */
    public static class ConstantConditionWhere implements Optimizer {
        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return Set.of(Where.class);
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof Where)) return false;

            List<String> inputs = op.getInputsToOp();
            // Only handle the 3-arg form (condition, x, y)
            if (inputs == null || inputs.size() != 3) return false;
            List<String> outputs = op.getOutputsOfOp();
            if (outputs == null || outputs.isEmpty()) return false;

            // Where inputs: [condition, x, y]
            String condVar = inputs.get(0);
            String xVar = inputs.get(1);
            String yVar = inputs.get(2);
            String outputVar = outputs.get(0);

            Boolean condValue = isConstantCondition(sd, constantArrays, condVar);
            if (condValue == null) return false;

            String replacement = condValue ? xVar : yVar;
            log.debug("ConstantConditionWhere: where({}, {}, {}) -> {} (condition all-{})",
                    condVar, xVar, yVar, replacement, condValue);

            replaceWithBranch(sd, helper, op, outputVar, replacement);
            return true;
        }
    }

    /**
     * {@code where_np(cond, x, y)} with constant all-true condition → x,
     * with constant all-false condition → y.
     *
     * <p>Same as ConstantConditionSelect but for the numpy-style where op
     * which has inputs [x, y, condition] (note different order).</p>
     */
    public static class ConstantConditionWhereNumpy implements Optimizer {
        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return Set.of(WhereNumpy.class);
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof WhereNumpy)) return false;

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.size() != 3) return false;
            List<String> outputs = op.getOutputsOfOp();
            if (outputs == null || outputs.isEmpty()) return false;

            // WhereNumpy inputs: [x, y, condition]
            String xVar = inputs.get(0);
            String yVar = inputs.get(1);
            String condVar = inputs.get(2);
            String outputVar = outputs.get(0);

            Boolean condValue = isConstantCondition(sd, constantArrays, condVar);
            if (condValue == null) return false;

            String replacement = condValue ? xVar : yVar;
            log.debug("ConstantConditionWhereNumpy: where_np({}, {}, {}) -> {} (condition all-{})",
                    condVar, xVar, yVar, replacement, condValue);

            replaceWithBranch(sd, helper, op, outputVar, replacement);
            return true;
        }
    }
}

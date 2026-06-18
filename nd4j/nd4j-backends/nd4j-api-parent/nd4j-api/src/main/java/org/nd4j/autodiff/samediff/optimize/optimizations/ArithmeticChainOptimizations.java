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
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.autodiff.samediff.optimize.OptimizationHelper;
import org.nd4j.autodiff.samediff.optimize.Optimizer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.AddOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MulOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.Set;

/**
 * Fold chains of arithmetic operations with constants.
 *
 * <ul>
 *   <li><b>FoldAddChain</b>: {@code add(add(x, c1), c2) → add(x, c1+c2)}
 *       Pre-computes the sum of two constant addends.</li>
 *   <li><b>FoldMulChain</b>: {@code mul(mul(x, c1), c2) → mul(x, c1*c2)}
 *       Pre-computes the product of two constant factors.</li>
 * </ul>
 */
@Slf4j
public class ArithmeticChainOptimizations extends BaseOptimizerSet {

    /**
     * Get the scalar constant value from a variable, or null if not a scalar constant.
     */
    private static INDArray getScalarConstant(SameDiff sd, ArrayHolder constantArrays, String varName) {
        SDVariable sdVar = sd.getVariable(varName);
        if (sdVar == null || sdVar.getVariableType() != VariableType.CONSTANT) return null;

        INDArray arr = constantArrays.getArray(varName);
        if (arr == null) arr = sd.getArrForVarName(varName);
        if (arr == null) return null;

        if (arr.isScalar() || arr.length() == 1) return arr;
        return null;
    }

    /**
     * {@code add(add(x, c1), c2) → add(x, c1+c2)}
     *
     * <p>Folds a chain of two add ops where the outer add has a scalar constant
     * and the inner add also has a scalar constant. Pre-computes their sum at
     * compile time. Only applies when the inner add result has exactly one consumer.</p>
     */
    public static class FoldAddChain implements Optimizer {
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

            // Find which input is the constant (could be either position)
            INDArray outerConst = null;
            int outerConstIdx = -1;
            int outerDynIdx = -1;
            for (int i = 0; i < 2; i++) {
                INDArray c = getScalarConstant(sd, constantArrays, inputs.get(i));
                if (c != null) {
                    outerConst = c;
                    outerConstIdx = i;
                    outerDynIdx = 1 - i;
                }
            }
            if (outerConst == null || outerConstIdx == -1) return false;

            // The dynamic input must be produced by another AddOp
            String innerResultVar = inputs.get(outerDynIdx);
            Variable innerResultVariable = sd.getVariables().get(innerResultVar);
            if (innerResultVariable == null) return false;

            String innerOpName = innerResultVariable.getOutputOfOp();
            if (innerOpName == null) return false;

            SameDiffOp innerOp = sd.getOps().get(innerOpName);
            if (innerOp == null || !(innerOp.getOp() instanceof AddOp)) return false;

            // Inner result must only be consumed by the outer add
            List<String> innerUsers = innerResultVariable.getInputsForOp();
            if (innerUsers == null || innerUsers.size() != 1) return false;

            List<String> innerInputs = innerOp.getInputsToOp();
            if (innerInputs == null || innerInputs.size() != 2) return false;

            // Find the constant in the inner add
            INDArray innerConst = null;
            int innerConstIdx = -1;
            int innerDynIdx = -1;
            for (int i = 0; i < 2; i++) {
                INDArray c = getScalarConstant(sd, constantArrays, innerInputs.get(i));
                if (c != null) {
                    innerConst = c;
                    innerConstIdx = i;
                    innerDynIdx = 1 - i;
                }
            }
            if (innerConst == null || innerConstIdx == -1) return false;

            // Pre-compute c1 + c2
            double sumValue = innerConst.getDouble(0) + outerConst.getDouble(0);
            INDArray foldedArr = Nd4j.scalar(outerConst.dataType(), sumValue);
            String foldedName = "add_folded_" + System.identityHashCode(foldedArr);
            sd.constant(foldedName, foldedArr);

            // Register consumer tracking for the new constant
            Variable foldedVariable = sd.getVariables().get(foldedName);
            if (foldedVariable != null && foldedVariable.getInputsForOp() != null) {
                foldedVariable.getInputsForOp().add(op.getName());
            }

            String xVarName = innerInputs.get(innerDynIdx); // The dynamic (non-constant) input

            log.debug("FoldAddChain: add(add({}, {}), {}) -> add({}, {}) [sum={}]",
                    xVarName, innerConst.getDouble(0), outerConst.getDouble(0),
                    xVarName, sumValue, foldedName);

            // Rewire outer add: inputs become [x, folded_constant]
            inputs.set(outerDynIdx, xVarName);
            inputs.set(outerConstIdx, foldedName);

            // Register x as input to outer op
            Variable xVariable = sd.getVariables().get(xVarName);
            if (xVariable != null && xVariable.getInputsForOp() != null
                    && !xVariable.getInputsForOp().contains(op.getName())) {
                xVariable.getInputsForOp().add(op.getName());
            }

            // Remove inner add op and intermediate variable
            OptimizationUtils.removeOp(sd, helper, innerOpName);
            OptimizationUtils.removeVariable(sd, helper, innerResultVar);

            return true;
        }
    }

    /**
     * {@code mul(mul(x, c1), c2) → mul(x, c1*c2)}
     *
     * <p>Folds a chain of two mul ops where the outer mul has a scalar constant
     * and the inner mul also has a scalar constant. Pre-computes their product at
     * compile time. Only applies when the inner mul result has exactly one consumer.</p>
     */
    public static class FoldMulChain implements Optimizer {
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

            // Find which input is the constant
            INDArray outerConst = null;
            int outerConstIdx = -1;
            int outerDynIdx = -1;
            for (int i = 0; i < 2; i++) {
                INDArray c = getScalarConstant(sd, constantArrays, inputs.get(i));
                if (c != null) {
                    outerConst = c;
                    outerConstIdx = i;
                    outerDynIdx = 1 - i;
                }
            }
            if (outerConst == null || outerConstIdx == -1) return false;

            // The dynamic input must be produced by another MulOp
            String innerResultVar = inputs.get(outerDynIdx);
            Variable innerResultVariable = sd.getVariables().get(innerResultVar);
            if (innerResultVariable == null) return false;

            String innerOpName = innerResultVariable.getOutputOfOp();
            if (innerOpName == null) return false;

            SameDiffOp innerOp = sd.getOps().get(innerOpName);
            if (innerOp == null || !(innerOp.getOp() instanceof MulOp)) return false;

            // Inner result must only be consumed by the outer mul
            List<String> innerUsers = innerResultVariable.getInputsForOp();
            if (innerUsers == null || innerUsers.size() != 1) return false;

            List<String> innerInputs = innerOp.getInputsToOp();
            if (innerInputs == null || innerInputs.size() != 2) return false;

            // Find the constant in the inner mul
            INDArray innerConst = null;
            int innerConstIdx = -1;
            int innerDynIdx = -1;
            for (int i = 0; i < 2; i++) {
                INDArray c = getScalarConstant(sd, constantArrays, innerInputs.get(i));
                if (c != null) {
                    innerConst = c;
                    innerConstIdx = i;
                    innerDynIdx = 1 - i;
                }
            }
            if (innerConst == null || innerConstIdx == -1) return false;

            // Pre-compute c1 * c2
            double productValue = innerConst.getDouble(0) * outerConst.getDouble(0);
            INDArray foldedArr = Nd4j.scalar(outerConst.dataType(), productValue);
            String foldedName = "mul_folded_" + System.identityHashCode(foldedArr);
            sd.constant(foldedName, foldedArr);

            // Register consumer tracking for the new constant
            Variable foldedVariable = sd.getVariables().get(foldedName);
            if (foldedVariable != null && foldedVariable.getInputsForOp() != null) {
                foldedVariable.getInputsForOp().add(op.getName());
            }

            String xVarName = innerInputs.get(innerDynIdx);

            log.debug("FoldMulChain: mul(mul({}, {}), {}) -> mul({}, {}) [product={}]",
                    xVarName, innerConst.getDouble(0), outerConst.getDouble(0),
                    xVarName, productValue, foldedName);

            // Rewire outer mul: inputs become [x, folded_constant]
            inputs.set(outerDynIdx, xVarName);
            inputs.set(outerConstIdx, foldedName);

            // Register x as input to outer op
            Variable xVariable = sd.getVariables().get(xVarName);
            if (xVariable != null && xVariable.getInputsForOp() != null
                    && !xVariable.getInputsForOp().contains(op.getName())) {
                xVariable.getInputsForOp().add(op.getName());
            }

            // Remove inner mul op and intermediate variable
            OptimizationUtils.removeOp(sd, helper, innerOpName);
            OptimizationUtils.removeVariable(sd, helper, innerResultVar);

            return true;
        }
    }
}

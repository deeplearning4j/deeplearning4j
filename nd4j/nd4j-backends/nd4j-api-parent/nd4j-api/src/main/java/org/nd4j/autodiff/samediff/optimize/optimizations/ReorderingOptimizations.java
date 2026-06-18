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
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.autodiff.samediff.optimize.OptimizationHelper;
import org.nd4j.autodiff.samediff.optimize.Optimizer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.shape.Transpose;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.AddOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MulOp;

import java.util.List;
import java.util.Set;

/**
 * Operator reordering and reassociation optimizations.
 *
 * <p>These passes restructure the graph to enable more constant folding and
 * eliminate redundant shape-manipulation ops.</p>
 *
 * <ul>
 *   <li><b>Associative reassociation</b>: {@code (a + c1) + c2 → a + (c1 + c2)}
 *       where c1, c2 are constants. The folded constant is pre-computed, eliminating
 *       one runtime add. Same for multiply.</li>
 *   <li><b>Transpose elimination</b>: {@code transpose(transpose(x)) → x} removes
 *       two redundant transpose ops. Common in ONNX-imported models.</li>
 * </ul>
 */
@Slf4j
public class ReorderingOptimizations extends BaseOptimizerSet {

    /**
     * Reassociate commutative/associative binary ops to enable constant folding.
     *
     * <p>Pattern: {@code op(op(a, const1), const2) → op(a, op(const1, const2))}</p>
     * <p>Since {@code op(const1, const2)} is all-constant, it gets folded by
     * {@link ConstantFunctionOptimizations} in the next iteration.</p>
     *
     * <p>Only applies to add and mul (both commutative and associative).</p>
     */
    public static class ReassociateConstants implements Optimizer {

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            DifferentialFunction df = op.getOp();
            if (df == null) return false;

            // Only commutative+associative ops
            boolean isAdd = df instanceof AddOp;
            boolean isMul = df instanceof MulOp;
            if (!isAdd && !isMul) return false;

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.size() != 2) return false;

            List<String> outputs = op.getOutputsOfOp();
            if (outputs == null || outputs.isEmpty()) return false;

            // Identify which input is constant
            String nonConstInput = null;
            String constInput = null;
            int constIdx = -1;

            for (int i = 0; i < 2; i++) {
                SDVariable v = sd.getVariable(inputs.get(i));
                if (v != null && v.isConstant()) {
                    constInput = inputs.get(i);
                    constIdx = i;
                    nonConstInput = inputs.get(1 - i);
                }
            }

            if (constInput == null || nonConstInput == null) return false;

            // Check if the non-constant input comes from the same type of op
            Variable nonConstVar = sd.getVariables().get(nonConstInput);
            if (nonConstVar == null) return false;
            String producerOpName = nonConstVar.getOutputOfOp();
            if (producerOpName == null) return false;
            SameDiffOp producerOp = sd.getOps().get(producerOpName);
            if (producerOp == null || producerOp.getOp() == null) return false;

            // Producer must be the same op type (add+add or mul+mul)
            boolean producerIsAdd = producerOp.getOp() instanceof AddOp;
            boolean producerIsMul = producerOp.getOp() instanceof MulOp;
            if (isAdd && !producerIsAdd) return false;
            if (isMul && !producerIsMul) return false;

            // The intermediate result must only be used by this outer op
            List<String> intermediateUsers = nonConstVar.getInputsForOp();
            if (intermediateUsers == null || intermediateUsers.size() != 1) return false;

            // Find the constant in the producer op
            List<String> producerInputs = producerOp.getInputsToOp();
            if (producerInputs == null || producerInputs.size() != 2) return false;

            String innerConst = null;
            String innerNonConst = null;
            for (int i = 0; i < 2; i++) {
                SDVariable v = sd.getVariable(producerInputs.get(i));
                if (v != null && v.isConstant()) {
                    innerConst = producerInputs.get(i);
                    innerNonConst = producerInputs.get(1 - i);
                }
            }

            if (innerConst == null || innerNonConst == null) return false;

            // Found: op(op(innerNonConst, innerConst), constInput) → op(innerNonConst, foldedConst)
            // Pre-compute foldedConst = op(innerConst, constInput)
            INDArray arr1 = getConstantArray(sd, constantArrays, innerConst);
            INDArray arr2 = getConstantArray(sd, constantArrays, constInput);
            if (arr1 == null || arr2 == null) return false;

            INDArray folded;
            try {
                if (isAdd) {
                    folded = arr1.add(arr2);
                } else {
                    folded = arr1.mul(arr2);
                }
            } catch (Exception e) {
                return false; // Shape incompatibility
            }

            String opName = isAdd ? "add" : "mul";
            String foldedName = "reassoc_" + opName + "_" + System.nanoTime();
            SDVariable foldedVar = sd.constant(foldedName, folded);

            log.debug("Reassociating: {}({}({}, {}), {}) -> {}({}, {})",
                    opName, opName, innerNonConst, innerConst, constInput,
                    opName, innerNonConst, foldedName);

            // Rewire the outer op: replace its inputs with (innerNonConst, foldedConst)
            inputs.set(0, innerNonConst);
            inputs.set(1, foldedName);

            // Update variable consumer tracking for innerNonConst
            Variable innerNonConstVar = sd.getVariables().get(innerNonConst);
            if (innerNonConstVar != null && innerNonConstVar.getInputsForOp() != null) {
                if (!innerNonConstVar.getInputsForOp().contains(op.getName())) {
                    innerNonConstVar.getInputsForOp().add(op.getName());
                }
            }

            // Register the outer op as a consumer of the folded constant
            Variable foldedVariable = sd.getVariables().get(foldedName);
            if (foldedVariable != null) {
                if (foldedVariable.getInputsForOp() == null) {
                    foldedVariable.setInputsForOp(new java.util.ArrayList<>());
                }
                foldedVariable.getInputsForOp().add(op.getName());
            }

            // Remove the old intermediate from outer op's consumer list
            // and remove the producer op entirely
            OptimizationUtils.removeOp(sd, helper, producerOpName);
            OptimizationUtils.removeVariable(sd, helper, nonConstInput);

            return true;
        }

        private INDArray getConstantArray(SameDiff sd, ArrayHolder constantArrays, String varName) {
            INDArray arr = constantArrays.getArray(varName);
            if (arr != null) return arr;
            SDVariable v = sd.getVariable(varName);
            return v != null ? v.getArr() : null;
        }
    }

    /**
     * Eliminate double transpose: {@code transpose(transpose(x)) → x}.
     *
     * <p>Two consecutive transposes without explicit permutation axes cancel each
     * other out. If both transposes are simple (no explicit permutation, i.e.,
     * reversing all axes), the result is the original tensor.</p>
     *
     * <p>With explicit permutations, transposes cancel when the composition of the
     * two permutations is the identity. This pass only handles the simple case
     * (no explicit permutation args) which covers the majority of ONNX imports.</p>
     */
    public static class EliminateDoubleTranspose implements Optimizer {

        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return Set.of(Transpose.class);
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof Transpose)) return false;

            Transpose outerTranspose = (Transpose) op.getOp();

            // Only handle simple transposes (no explicit permutation dims)
            // A transpose with no explicit dims just reverses all axes.
            if (hasExplicitPermutation(outerTranspose)) return false;

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.isEmpty()) return false;

            // The first input is the data tensor; skip if there's a permutation input
            String inputVar = inputs.get(0);
            Variable inputVariable = sd.getVariables().get(inputVar);
            if (inputVariable == null) return false;

            String producerOpName = inputVariable.getOutputOfOp();
            if (producerOpName == null) return false;

            SameDiffOp producerOp = sd.getOps().get(producerOpName);
            if (producerOp == null || !(producerOp.getOp() instanceof Transpose)) return false;

            Transpose innerTranspose = (Transpose) producerOp.getOp();
            if (hasExplicitPermutation(innerTranspose)) return false;

            // Both are simple transposes -- they cancel out
            // Check that the inner transpose output is only used by this outer transpose
            List<String> innerUsers = inputVariable.getInputsForOp();
            if (innerUsers == null || innerUsers.size() != 1) return false;

            // Get the original input (before both transposes)
            List<String> innerInputs = producerOp.getInputsToOp();
            if (innerInputs == null || innerInputs.isEmpty()) return false;
            String originalInput = innerInputs.get(0);

            List<String> outputs = op.getOutputsOfOp();
            if (outputs == null || outputs.isEmpty()) return false;
            String outerOutput = outputs.get(0);

            log.debug("Eliminating double transpose: transpose(transpose({})) -> {}",
                    originalInput, originalInput);

            // Replace all uses of the outer transpose output with the original input
            OptimizationUtils.replaceOpInputsWith(sd, helper, outerOutput, originalInput);

            // Update graph outputs
            List<String> graphOutputs = sd.outputs();
            if (graphOutputs != null) {
                for (int i = 0; i < graphOutputs.size(); i++) {
                    if (graphOutputs.get(i).equals(outerOutput)) {
                        graphOutputs.set(i, originalInput);
                    }
                }
            }

            // Remove both transpose ops
            OptimizationUtils.removeOp(sd, helper, op.getName());
            OptimizationUtils.removeVariable(sd, helper, outerOutput);

            // If inner transpose output now has no consumers, remove it too
            Variable innerAfter = sd.getVariables().get(inputVar);
            if (innerAfter != null) {
                List<String> remaining = innerAfter.getInputsForOp();
                if (remaining == null || remaining.isEmpty()) {
                    OptimizationUtils.removeOp(sd, helper, producerOpName);
                    OptimizationUtils.removeVariable(sd, helper, inputVar);
                }
            }

            return true;
        }

        private boolean hasExplicitPermutation(Transpose t) {
            // A transpose with explicit permutation has either:
            // 1. iArguments set (permutation as integer args), or
            // 2. A second input (dynamic permutation array)
            // A simple reverse-all-axes transpose has neither.
            long[] iArgs = t.iArgs();
            if (iArgs != null && iArgs.length > 0) {
                return true;
            }
            // Check for second input (permutation array)
            SDVariable[] args = t.args();
            return args != null && args.length > 1;
        }
    }
}

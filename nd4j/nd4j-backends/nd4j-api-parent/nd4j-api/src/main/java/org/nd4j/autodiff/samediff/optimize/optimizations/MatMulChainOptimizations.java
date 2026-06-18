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
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.autodiff.samediff.optimize.OptimizationHelper;
import org.nd4j.autodiff.samediff.optimize.Optimizer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce.Mmul;

import java.util.List;
import java.util.Set;

/**
 * MatMul chain optimizations.
 *
 * <ul>
 *   <li><b>FoldConstantMatMulChain</b>: {@code matmul(X, W1) * W2 → matmul(X, W1W2)}
 *       where W1 and W2 are both constants. Pre-computes {@code W1W2 = matmul(W1, W2)}
 *       at compile time, reducing two matmuls to one at runtime.</li>
 *   <li><b>TransposeMatMulFusion</b>: {@code matmul(transpose(A), B) → matmul(A, B, transposeA=true)}
 *       eliminates the explicit transpose op by using the matmul's transpose flag.</li>
 * </ul>
 */
@Slf4j
public class MatMulChainOptimizations extends BaseOptimizerSet {

    /**
     * matmul(matmul(X, W1), W2) → matmul(X, W1W2) where W1 and W2 are constants.
     *
     * Pre-computes the constant weight product at compile time.
     * Only applies when:
     * - The inner matmul has exactly one consumer (the outer matmul)
     * - W1 and W2 are both constants with compatible shapes
     * - Neither matmul uses transpose flags (for simplicity)
     */
    public static class FoldConstantMatMulChain implements Optimizer {
        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return Set.of(Mmul.class);
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof Mmul)) return false;

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.size() != 2) return false;
            List<String> outputs = op.getOutputsOfOp();
            if (outputs == null || outputs.isEmpty()) return false;

            String innerResultVar = inputs.get(0);  // First input to outer matmul
            String w2VarName = inputs.get(1);        // Second input (W2)

            // W2 must be a constant
            SDVariable w2Var = sd.getVariable(w2VarName);
            if (w2Var == null || w2Var.getVariableType() != VariableType.CONSTANT) return false;

            // Inner result must be produced by another Mmul
            Variable innerResultVariable = sd.getVariables().get(innerResultVar);
            if (innerResultVariable == null) return false;
            String innerOpName = innerResultVariable.getOutputOfOp();
            if (innerOpName == null) return false;

            SameDiffOp innerOp = sd.getOps().get(innerOpName);
            if (innerOp == null || !(innerOp.getOp() instanceof Mmul)) return false;

            // Inner result must only be consumed by the outer matmul
            List<String> innerUsers = innerResultVariable.getInputsForOp();
            if (innerUsers == null || innerUsers.size() != 1) return false;

            // Inner matmul inputs
            List<String> innerInputs = innerOp.getInputsToOp();
            if (innerInputs == null || innerInputs.size() != 2) return false;

            String xVarName = innerInputs.get(0);    // X (dynamic input)
            String w1VarName = innerInputs.get(1);   // W1 (must be constant)

            // W1 must be a constant
            SDVariable w1Var = sd.getVariable(w1VarName);
            if (w1Var == null || w1Var.getVariableType() != VariableType.CONSTANT) return false;

            // Skip if either matmul uses transpose flags
            Mmul outerMmul = (Mmul) op.getOp();
            Mmul innerMmul = (Mmul) innerOp.getOp();
            long[] outerIArgs = outerMmul.iArgs();
            long[] innerIArgs = innerMmul.iArgs();
            if (outerIArgs != null && outerIArgs.length > 0) {
                for (long arg : outerIArgs) {
                    if (arg != 0) return false;
                }
            }
            if (innerIArgs != null && innerIArgs.length > 0) {
                for (long arg : innerIArgs) {
                    if (arg != 0) return false;
                }
            }

            // Get the actual constant arrays
            INDArray w1Arr = constantArrays.getArray(w1VarName);
            if (w1Arr == null) w1Arr = sd.getArrForVarName(w1VarName);
            INDArray w2Arr = constantArrays.getArray(w2VarName);
            if (w2Arr == null) w2Arr = sd.getArrForVarName(w2VarName);

            if (w1Arr == null || w2Arr == null) return false;
            if (w1Arr.rank() != 2 || w2Arr.rank() != 2) return false;
            if (w1Arr.size(1) != w2Arr.size(0)) return false;

            // Pre-compute W1 @ W2
            INDArray w1w2 = w1Arr.mmul(w2Arr);
            String foldedName = "matmul_folded_" + System.identityHashCode(w1w2);
            sd.constant(foldedName, w1w2);

            // Register consumer tracking for the new constant
            Variable foldedVariable = sd.getVariables().get(foldedName);
            if (foldedVariable != null && foldedVariable.getInputsForOp() != null) {
                foldedVariable.getInputsForOp().add(op.getName());
            }

            log.debug("FoldConstantMatMulChain: matmul(matmul({}, {}), {}) -> matmul({}, {})",
                    xVarName, w1VarName, w2VarName, xVarName, foldedName);

            String outputVar = outputs.get(0);

            // Rewire outer matmul: inputs become [X, W1W2]
            inputs.set(0, xVarName);
            inputs.set(1, foldedName);

            // Register X as input to outer op
            Variable xVariable = sd.getVariables().get(xVarName);
            if (xVariable != null && xVariable.getInputsForOp() != null
                    && !xVariable.getInputsForOp().contains(op.getName())) {
                xVariable.getInputsForOp().add(op.getName());
            }

            // Remove inner matmul op and intermediate variable
            OptimizationUtils.removeOp(sd, helper, innerOpName);
            OptimizationUtils.removeVariable(sd, helper, innerResultVar);

            return true;
        }
    }

    /**
     * matmul(transpose(A), B) → matmul(A, B, transposeA=true).
     * Also handles matmul(A, transpose(B)) → matmul(A, B, transposeB=true).
     *
     * Eliminates the explicit transpose op by folding it into the matmul's flags.
     * Only applies when the transpose has no explicit permutation (simple 2D transpose)
     * and the transpose output is only consumed by this matmul.
     */
    public static class TransposeMatMulFusion implements Optimizer {
        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return Set.of(Mmul.class);
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof Mmul)) return false;

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.size() != 2) return false;

            Mmul mmul = (Mmul) op.getOp();
            long[] iArgs = mmul.iArgs();

            // Check current transpose flags — iArgs: [transposeA, transposeB, transposeResult]
            boolean currentTransA = (iArgs != null && iArgs.length > 0 && iArgs[0] != 0);
            boolean currentTransB = (iArgs != null && iArgs.length > 1 && iArgs[1] != 0);

            boolean applied = false;

            // Check input A for transpose
            if (!currentTransA) {
                applied = tryAbsorbTranspose(sd, helper, op, inputs, 0, mmul);
            }

            // Check input B for transpose (if A wasn't absorbed)
            if (!applied && !currentTransB) {
                applied = tryAbsorbTranspose(sd, helper, op, inputs, 1, mmul);
            }

            return applied;
        }

        private boolean tryAbsorbTranspose(SameDiff sd, OptimizationHelper helper,
                                           SameDiffOp op, List<String> inputs,
                                           int inputIdx, Mmul mmul) {
            String inputVar = inputs.get(inputIdx);

            Variable inputVariable = sd.getVariables().get(inputVar);
            if (inputVariable == null) return false;
            String producerOpName = inputVariable.getOutputOfOp();
            if (producerOpName == null) return false;

            SameDiffOp producerOp = sd.getOps().get(producerOpName);
            if (producerOp == null) return false;

            // Must be a Transpose op (not Permute with custom dims)
            DifferentialFunction producerFn = producerOp.getOp();
            if (!producerFn.opName().equals("transpose")) return false;

            // Must be a simple transpose (no explicit permutation)
            if (producerOp.getInputsToOp().size() > 1) return false;

            // Transpose output must only be consumed by this matmul
            List<String> transposeUsers = inputVariable.getInputsForOp();
            if (transposeUsers == null || transposeUsers.size() != 1) return false;

            List<String> transposeInputs = producerOp.getInputsToOp();
            if (transposeInputs == null || transposeInputs.isEmpty()) return false;
            String originalInput = transposeInputs.get(0);

            log.debug("TransposeMatMulFusion: absorbing transpose on input {} of matmul", inputIdx);

            // Update matmul iArgs to set the transpose flag
            long[] iArgs = mmul.iArgs();
            long[] newIArgs;
            if (iArgs == null || iArgs.length < 3) {
                newIArgs = new long[3];
                if (iArgs != null) {
                    System.arraycopy(iArgs, 0, newIArgs, 0, iArgs.length);
                }
            } else {
                newIArgs = iArgs.clone();
            }
            newIArgs[inputIdx] = 1;  // Set transposeA or transposeB

            // Replace the Mmul's iArguments (addIArgument appends, so we must clear first)
            mmul.clearIArguments();
            mmul.addIArgument(newIArgs);

            // Rewire: point matmul directly at the pre-transpose input
            inputs.set(inputIdx, originalInput);

            // Register original input as consumer of this op
            Variable originalVariable = sd.getVariables().get(originalInput);
            if (originalVariable != null && originalVariable.getInputsForOp() != null
                    && !originalVariable.getInputsForOp().contains(op.getName())) {
                originalVariable.getInputsForOp().add(op.getName());
            }

            // Remove transpose op and intermediate variable
            OptimizationUtils.removeOp(sd, helper, producerOpName);
            OptimizationUtils.removeVariable(sd, helper, inputVar);

            return true;
        }
    }
}

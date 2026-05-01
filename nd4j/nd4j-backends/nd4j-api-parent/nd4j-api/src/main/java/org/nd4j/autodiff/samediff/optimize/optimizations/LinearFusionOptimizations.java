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
import org.nd4j.linalg.api.ops.impl.reduce.Mmul;
import org.nd4j.linalg.api.ops.impl.reduce.TensorMmul;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.AddOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BiasAdd;
import org.nd4j.linalg.api.ops.impl.transforms.custom.XwPlusB;
import org.nd4j.linalg.api.ops.impl.shape.Reshape;

import java.util.List;

/**
 * Linear layer fusion optimizations for transformers/BERT.
 * Fuses matmul + add patterns into xw_plus_b for better performance.
 */
@Slf4j
public class LinearFusionOptimizations extends BaseOptimizerSet {

    /**
     * Fuses [matmul(x, w) -> add(bias)] into xw_plus_b(x, w, bias)
     * This is a common pattern in transformer linear layers.
     *
     * Requirements for fusion:
     * 1. Add operation where one input comes from matmul
     * 2. The other input to add is a 1D bias vector (or broadcastable)
     * 3. The matmul output is only used by this add operation
     */
    public static class FuseMatMulWithAdd implements Optimizer {
        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            // Check if this is an Add operation
            if (!(op.getOp() instanceof AddOp) && !(op.getOp() instanceof BiasAdd)) {
                return false;
            }

            List<String> addInputs = op.getInputsToOp();
            if (addInputs == null || addInputs.size() != 2) {
                return false;
            }

            // Find which input comes from matmul
            String matmulOutputVar = null;
            String biasVar = null;
            SameDiffOp matmulOp = null;

            for (int i = 0; i < 2; i++) {
                String inputVar = addInputs.get(i);
                // Use fast O(1) lookup via helper instead of PatriciaTrie O(k)
                Variable v = helper.getVariable(inputVar);
                if (v == null) continue;

                String producerOpName = v.getOutputOfOp();
                if (producerOpName == null) continue;

                SameDiffOp producerOp = sd.getOps().get(producerOpName);
                if (producerOp != null && producerOp.getOp() instanceof Mmul) {
                    matmulOutputVar = inputVar;
                    matmulOp = producerOp;
                    biasVar = addInputs.get(1 - i);
                    break;
                }
            }

            if (matmulOp == null || biasVar == null) {
                return false;
            }

            // Check that the matmul output is only used by this add
            // Use fast O(1) lookup via helper
            Variable matmulOutVariable = helper.getVariable(matmulOutputVar);
            if (matmulOutVariable == null) return false;

            List<String> matmulOutputUsers = matmulOutVariable.getInputsForOp();
            if (matmulOutputUsers == null || matmulOutputUsers.size() != 1) {
                // Matmul output is used by multiple ops, can't fuse
                return false;
            }

            // Check bias is 1D or compatible shape
            // Use fast O(1) lookup via helper
            Variable biasVariable = helper.getVariable(biasVar);
            if (biasVariable == null) return false;

            SDVariable biasSDVar = sd.getVariable(biasVar);
            if (biasSDVar == null) return false;

            // Bias must be a model parameter (CONSTANT or VARIABLE), not a computed value.
            // ARRAY variables are computed during execution (e.g., residual connections)
            // and must not be fused as bias.
            if (biasSDVar.getVariableType() == VariableType.ARRAY) {
                return false;
            }

            // Get bias shape - it should be 1D for xw_plus_b
            long[] biasShape = biasSDVar.getShape();
            if (biasShape != null && biasShape.length > 2) {
                // Bias has too many dimensions
                return false;
            }

            // Get matmul inputs
            List<String> matmulInputs = matmulOp.getInputsToOp();
            if (matmulInputs == null || matmulInputs.size() < 2) {
                return false;
            }

            String xVar = matmulInputs.get(0);
            String wVar = matmulInputs.get(1);

            // Get the add output variable name
            List<String> addOutputs = op.getOutputsOfOp();
            if (addOutputs == null || addOutputs.isEmpty()) {
                return false;
            }
            String addOutputVar = addOutputs.get(0);

            log.info("Fusing matmul({}, {}) + add({}) into xw_plus_b", xVar, wVar, biasVar);

            // Create xw_plus_b operation
            SDVariable xSDVar = sd.getVariable(xVar);
            SDVariable wSDVar = sd.getVariable(wVar);

            if (xSDVar == null || wSDVar == null) {
                return false;
            }

            // xw_plus_b relies on oneDNN inner_product which requires hardware BF16/FP16
            // support. Skip fusion for non-FP32 types — the unfused matmul+add path works
            // for all types via generic C++ MmulHelper.
            org.nd4j.linalg.api.buffer.DataType xDtype = xSDVar.dataType();
            if (xDtype != null && xDtype != org.nd4j.linalg.api.buffer.DataType.FLOAT
                    && xDtype != org.nd4j.linalg.api.buffer.DataType.DOUBLE) {
                return false;
            }

            try {
                // Extract transpose flags from the original Mmul
                Mmul mmul = (Mmul) matmulOp.getOp();
                long[] mmulIArgs = mmul.iArgs();
                boolean transposeA = mmulIArgs != null && mmulIArgs.length > 0 && mmulIArgs[0] != 0;
                boolean transposeB = mmulIArgs != null && mmulIArgs.length > 1 && mmulIArgs[1] != 0;
                boolean transposeResult = mmulIArgs != null && mmulIArgs.length > 2 && mmulIArgs[2] != 0;

                // Create the fused op preserving transpose flags
                SDVariable fusedOutput = new XwPlusB(sd, xSDVar, wSDVar, biasSDVar,
                        transposeA, transposeB, transposeResult).outputVariable();

                // Replace all uses of the add output with the fused output
                OptimizationUtils.replaceOpInputsWith(sd, helper, addOutputVar, fusedOutput.name());

                // If the add output was a registered graph output, update the output list
                // to point to the fused output variable. Otherwise outputSingle() returns
                // null because the orphaned variable has no producing op.
                List<String> graphOutputs = sd.outputs();
                if (graphOutputs != null) {
                    for (int idx = 0; idx < graphOutputs.size(); idx++) {
                        if (graphOutputs.get(idx).equals(addOutputVar)) {
                            graphOutputs.set(idx, fusedOutput.name());
                        }
                    }
                }

                // Remove the old add and matmul operations
                OptimizationUtils.removeOp(sd, helper, op.getName());
                OptimizationUtils.removeOp(sd, helper, matmulOp.getName());

                // Remove intermediate matmul output variable (no longer used)
                OptimizationUtils.removeVariable(sd, helper, matmulOutputVar);

                // Remove the old add output variable (replaced by fusedOutput)
                OptimizationUtils.removeVariable(sd, helper, addOutputVar);

                return true;
            } catch (Exception e) {
                log.warn("Failed to fuse matmul+add into xw_plus_b: {}", e.getMessage());
                return false;
            }
        }
    }

    /**
     * Fuses [tensormmul(x, w) -> add(bias)] into xw_plus_b(x, w, bias)
     * when tensormmul is used as a standard matrix multiplication.
     *
     * This pattern is common when ONNX models are imported where MatMul
     * is represented as TensorMmul with axes [[1], [0]] or similar.
     */
    public static class FuseTensorMmulWithAdd implements Optimizer {
        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            // Check if this is an Add operation
            if (!(op.getOp() instanceof AddOp) && !(op.getOp() instanceof BiasAdd)) {
                return false;
            }

            List<String> addInputs = op.getInputsToOp();
            if (addInputs == null || addInputs.size() != 2) {
                return false;
            }

            // Find which input comes from tensormmul
            String tensorMmulOutputVar = null;
            String biasVar = null;
            SameDiffOp tensorMmulOp = null;

            for (int i = 0; i < 2; i++) {
                String inputVar = addInputs.get(i);
                // Use fast O(1) lookup via helper instead of PatriciaTrie O(k)
                Variable v = helper.getVariable(inputVar);
                if (v == null) continue;

                String producerOpName = v.getOutputOfOp();
                if (producerOpName == null) continue;

                SameDiffOp producerOp = sd.getOps().get(producerOpName);
                if (producerOp != null && producerOp.getOp() instanceof TensorMmul) {
                    tensorMmulOutputVar = inputVar;
                    tensorMmulOp = producerOp;
                    biasVar = addInputs.get(1 - i);
                    break;
                }
            }

            if (tensorMmulOp == null || biasVar == null) {
                return false;
            }

            // Check that the tensormmul output is only used by this add
            // Use fast O(1) lookup via helper
            Variable tensorMmulOutVariable = helper.getVariable(tensorMmulOutputVar);
            if (tensorMmulOutVariable == null) return false;

            List<String> tensorMmulOutputUsers = tensorMmulOutVariable.getInputsForOp();
            if (tensorMmulOutputUsers == null || tensorMmulOutputUsers.size() != 1) {
                return false;
            }

            // Get tensormmul inputs
            List<String> tensorMmulInputs = tensorMmulOp.getInputsToOp();
            if (tensorMmulInputs == null || tensorMmulInputs.size() < 2) {
                return false;
            }

            String xVar = tensorMmulInputs.get(0);
            String wVar = tensorMmulInputs.get(1);

            // Check bias is compatible
            SDVariable biasSDVar = sd.getVariable(biasVar);
            if (biasSDVar == null) return false;

            // Bias must be a model parameter (CONSTANT or VARIABLE), not a computed value.
            if (biasSDVar.getVariableType() == VariableType.ARRAY) {
                return false;
            }

            long[] biasShape = biasSDVar.getShape();
            if (biasShape != null && biasShape.length > 2) {
                return false;
            }

            // Get the add output variable name
            List<String> addOutputs = op.getOutputsOfOp();
            if (addOutputs == null || addOutputs.isEmpty()) {
                return false;
            }
            String addOutputVar = addOutputs.get(0);

            SDVariable xSDVar = sd.getVariable(xVar);
            SDVariable wSDVar = sd.getVariable(wVar);

            if (xSDVar == null || wSDVar == null) {
                return false;
            }

            log.info("Fusing tensormmul({}, {}) + add({}) into xw_plus_b", xVar, wVar, biasVar);

            try {
                // Create the fused op
                SDVariable fusedOutput = new XwPlusB(sd, xSDVar, wSDVar, biasSDVar).outputVariable();

                // Replace all uses of the add output with the fused output
                OptimizationUtils.replaceOpInputsWith(sd, helper, addOutputVar, fusedOutput.name());

                // If the add output was a registered graph output, update the output list
                List<String> graphOutputs = sd.outputs();
                if (graphOutputs != null) {
                    for (int idx = 0; idx < graphOutputs.size(); idx++) {
                        if (graphOutputs.get(idx).equals(addOutputVar)) {
                            graphOutputs.set(idx, fusedOutput.name());
                        }
                    }
                }

                // Remove the old add and tensormmul operations
                OptimizationUtils.removeOp(sd, helper, op.getName());
                OptimizationUtils.removeOp(sd, helper, tensorMmulOp.getName());

                // Remove old variables
                OptimizationUtils.removeVariable(sd, helper, tensorMmulOutputVar);
                OptimizationUtils.removeVariable(sd, helper, addOutputVar);

                return true;
            } catch (Exception e) {
                log.warn("Failed to fuse tensormmul+add into xw_plus_b: {}", e.getMessage());
                return false;
            }
        }
    }

    /**
     * Fuses consecutive reshape operations into a single reshape.
     * Pattern: reshape(reshape(x, shape1), shape2) -> reshape(x, shape2)
     *
     * This is common after model import where multiple reshapes get chained.
     *
     * NOTE: DISABLED - This optimization has a bug where the output shape gets corrupted.
     * The issue is that when we update the reshape's input, the output shape computation
     * may get affected in unexpected ways. Needs further investigation.
     */
    public static class FuseConsecutiveReshapes implements Optimizer {
        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            // DISABLED: Output shapes get corrupted after fusion
            return false;
            /*
            // Check if this is a Reshape operation
            if (!(op.getOp() instanceof Reshape)) {
                return false;
            }

            List<String> reshapeInputs = op.getInputsToOp();
            if (reshapeInputs == null || reshapeInputs.isEmpty()) {
                return false;
            }

            // Check if the input comes from another reshape
            String inputVar = reshapeInputs.get(0);
            Variable v = sd.getVariables().get(inputVar);
            if (v == null) return false;

            String producerOpName = v.getOutputOfOp();
            if (producerOpName == null) return false;

            SameDiffOp producerOp = sd.getOps().get(producerOpName);
            if (producerOp == null || !(producerOp.getOp() instanceof Reshape)) {
                return false;
            }

            // Found consecutive reshapes!
            // Check that the intermediate reshape output is only used by this reshape
            List<String> intermediateUsers = v.getInputsForOp();
            if (intermediateUsers == null || intermediateUsers.size() != 1) {
                // Intermediate reshape is used elsewhere, can't fuse
                return false;
            }

            // Get the original input to the first reshape
            List<String> firstReshapeInputs = producerOp.getInputsToOp();
            if (firstReshapeInputs == null || firstReshapeInputs.isEmpty()) {
                return false;
            }
            String originalInput = firstReshapeInputs.get(0);

            log.info("Fusing consecutive reshapes: reshape(reshape({}, ...), ...) -> reshape({}, ...)",
                    originalInput, originalInput);

            try {
                // Update the current reshape to take the original input directly
                // by modifying its input list
                reshapeInputs.set(0, originalInput);

                // Update the variable's inputsForOp
                Variable originalVar = sd.getVariables().get(originalInput);
                if (originalVar != null) {
                    List<String> inputsForOp = originalVar.getInputsForOp();
                    if (inputsForOp != null && !inputsForOp.contains(op.getName())) {
                        inputsForOp.add(op.getName());
                    }
                }

                // Remove the intermediate reshape operation
                OptimizationUtils.removeOp(sd, producerOp.getName());

                // Remove the intermediate variable
                OptimizationUtils.removeVariable(sd, inputVar);

                return true;
            } catch (Exception e) {
                log.warn("Failed to fuse consecutive reshapes: {}", e.getMessage());
                return false;
            }
            */
        }
    }
}

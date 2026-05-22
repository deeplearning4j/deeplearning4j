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
import org.nd4j.autodiff.samediff.optimize.OptimizationHelper;
import org.nd4j.autodiff.samediff.optimize.Optimizer;
import org.nd4j.linalg.api.ops.impl.shape.Permute;
import org.nd4j.linalg.api.ops.impl.shape.Reshape;
import org.nd4j.linalg.api.ops.impl.transforms.same.Identity;

import java.util.List;
import java.util.Set;

@Slf4j
public class IdentityFunctionOptimizations extends BaseOptimizerSet {

    /**
     * Remove permute(0,1,2,...,rank-1) as this is a no-op.
     * Checks if the permute's iArguments form an identity permutation.
     */
    public static class RemoveIdentityPermute implements Optimizer {

        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return Set.of(Permute.class);
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op, ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof Permute)) return false;
            Permute permute = (Permute) op.getOp();

            long[] iArgs = permute.iArgs();
            if (iArgs == null || iArgs.length == 0) return false;

            // Check if permutation is identity: [0, 1, 2, ..., n-1]
            for (int i = 0; i < iArgs.length; i++) {
                if (iArgs[i] != i) return false;
            }

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.isEmpty()) return false;
            List<String> outputs = op.getOutputsOfOp();
            if (outputs == null || outputs.isEmpty()) return false;

            String inputName = inputs.get(0);
            String outputName = outputs.get(0);

            log.debug("Removing identity permute: permute({}, [0,1,...,{}]) -> {}", inputName, iArgs.length - 1, inputName);

            OptimizationUtils.replaceOpInputsWith(sd, helper, outputName, inputName);

            List<String> graphOutputs = sd.outputs();
            if (graphOutputs != null) {
                for (int i = 0; i < graphOutputs.size(); i++) {
                    if (graphOutputs.get(i).equals(outputName)) {
                        graphOutputs.set(i, inputName);
                    }
                }
            }

            OptimizationUtils.removeOp(sd, helper, op.getName());
            OptimizationUtils.removeVariable(sd, helper, outputName);
            return true;
        }
    }

    /**
     * Remove reshape(x) when the output shape equals the input shape.
     * This is a no-op reshape that adds unnecessary overhead.
     */
    public static class RemoveNoopReshape implements Optimizer {

        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return Set.of(Reshape.class);
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op, ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof Reshape)) return false;
            Reshape reshape = (Reshape) op.getOp();

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.isEmpty()) return false;
            List<String> outputs = op.getOutputsOfOp();
            if (outputs == null || outputs.isEmpty()) return false;

            String inputName = inputs.get(0);
            String outputName = outputs.get(0);

            // Get the target shape from iArguments.
            // Reshape stores iArgs as [ordering_flag, dim1, dim2, ...].
            // The ordering flag is negative (C_ORDER=-99, F_ORDER=-102).
            long[] iArgs = reshape.iArgs();
            if (iArgs == null || iArgs.length < 2) return false;

            // Extract the actual shape dimensions (skip the ordering flag)
            int shapeStart = (iArgs[0] < 0) ? 1 : 0;
            int shapeLen = iArgs.length - shapeStart;

            // Get the input variable's shape
            var inputVar = sd.getVariable(inputName);
            if (inputVar == null) return false;
            long[] inputShape = inputVar.getShape();
            if (inputShape == null) return false;

            // Compare shapes
            if (inputShape.length != shapeLen) return false;
            for (int i = 0; i < inputShape.length; i++) {
                // Skip dynamic dims (-1 means unknown)
                if (inputShape[i] < 0 || iArgs[shapeStart + i] < 0) return false;
                if (inputShape[i] != iArgs[shapeStart + i]) return false;
            }

            log.debug("Removing no-op reshape: reshape({}, {}) -> {}", inputName, java.util.Arrays.toString(iArgs), inputName);

            OptimizationUtils.replaceOpInputsWith(sd, helper, outputName, inputName);

            List<String> graphOutputs = sd.outputs();
            if (graphOutputs != null) {
                for (int i = 0; i < graphOutputs.size(); i++) {
                    if (graphOutputs.get(i).equals(outputName)) {
                        graphOutputs.set(i, inputName);
                    }
                }
            }

            OptimizationUtils.removeOp(sd, helper, op.getName());
            OptimizationUtils.removeVariable(sd, helper, outputName);
            return true;
        }
    }

    /**
     * Remove identity(x)
     */
    public static class RemoveIdentityOps implements Optimizer {
        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op, ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if(op.getOp() instanceof Identity){
                String inName = op.getInputsToOp().get(0);
                String outputName = op.getOutputsOfOp().get(0);
                OptimizationUtils.removeOp(sd, helper, op.getName());
                OptimizationUtils.replaceOpInputsWith(sd, helper, outputName, inName);
                OptimizationUtils.removeVariable(sd, helper, outputName);
                return true;
            }

            return false;
        }
    }
}

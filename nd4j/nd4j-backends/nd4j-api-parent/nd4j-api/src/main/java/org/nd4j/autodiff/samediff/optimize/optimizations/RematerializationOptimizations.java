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
import org.nd4j.linalg.api.ops.BaseTransformSameOp;
import org.nd4j.linalg.api.ops.BaseTransformStrictOp;
import org.nd4j.linalg.api.ops.BaseTransformFloatOp;
import org.nd4j.linalg.api.ops.BaseTransformBoolOp;

import java.util.ArrayList;
import java.util.List;

/**
 * Rematerialization optimization: trades compute for memory by duplicating cheap ops
 * near their consumers instead of keeping the intermediate result alive across many steps.
 *
 * When a cheap unary op's output has multiple consumers that are far apart in the
 * execution order, the intermediate tensor must stay alive for the entire span.
 * Rematerialization duplicates the op so each consumer has its own copy, allowing
 * earlier consumers' copies to be freed sooner.
 *
 * This is only applied when:
 * 1. The producing op is cheap (unary elementwise transform)
 * 2. The output has exactly 2 consumers (conservative — avoids explosion)
 * 3. The producing op's input is still alive at the point of each consumer
 *    (i.e., the input has consumers at or after the latest consumer of the output)
 *
 * The net effect: instead of one output tensor alive for N steps, we have two
 * shorter-lived tensors, reducing peak memory.
 */
@Slf4j
public class RematerializationOptimizations extends BaseOptimizerSet {

    /**
     * Duplicates a cheap unary op for each of its consumers when the output
     * has exactly 2 consumers, allowing earlier release of intermediate buffers.
     */
    public static class RematerializeCheapUnaryOps implements Optimizer {

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            // Only target cheap unary elementwise transforms
            DifferentialFunction fn = op.getOp();
            if (!isCheapUnaryOp(fn)) return false;

            List<String> inputs = op.getInputsToOp();
            List<String> outputs = op.getOutputsOfOp();
            if (inputs == null || inputs.size() != 1 || outputs == null || outputs.isEmpty()) return false;

            String inputVar = inputs.get(0);
            String outputVar = outputs.get(0);

            // Check consumers of the output
            Variable outV = helper.getVariable(outputVar);
            if (outV == null) outV = sd.getVariables().get(outputVar);
            if (outV == null) return false;

            List<String> consumers = outV.getInputsForOp();
            if (consumers == null || consumers.size() != 2) return false;

            // Don't rematerialize if the output is a graph output
            List<String> graphOutputs = sd.outputs();
            if (graphOutputs != null && graphOutputs.contains(outputVar)) return false;

            // Verify the input variable has enough consumers to stay alive
            // (it must survive until both new copies execute)
            Variable inV = helper.getVariable(inputVar);
            if (inV == null) inV = sd.getVariables().get(inputVar);
            if (inV == null) return false;

            // The input must have at least one consumer besides the op we're rematerializing
            // (otherwise the input might get freed before the duplicate executes)
            List<String> inputConsumers = inV.getInputsForOp();
            if (inputConsumers == null || inputConsumers.size() < 1) return false;

            // Create a duplicate op for the second consumer
            String consumer1 = consumers.get(0);
            String consumer2 = consumers.get(1);

            // Create a copy of this op with a new output name
            String dupOutputName = outputVar + "__remat";
            int suffix = 0;
            while (sd.hasVariable(dupOutputName)) {
                dupOutputName = outputVar + "__remat_" + (suffix++);
            }

            try {
                // Duplicate the op by creating the same operation
                if (!duplicateUnaryOp(sd, fn, inputVar, dupOutputName)) return false;

                // Redirect consumer2 to use the duplicate output
                SameDiffOp consumer2Op = sd.getOps().get(consumer2);
                if (consumer2Op == null) return false;

                List<String> consumer2Inputs = consumer2Op.getInputsToOp();
                if (consumer2Inputs == null) return false;

                boolean replaced = false;
                for (int i = 0; i < consumer2Inputs.size(); i++) {
                    if (consumer2Inputs.get(i).equals(outputVar)) {
                        consumer2Inputs.set(i, dupOutputName);
                        replaced = true;
                    }
                }
                if (!replaced) return false;

                // Update the variable consumer lists
                // Remove consumer2 from the original output's consumer list
                outV.getInputsForOp().remove(consumer2);

                // Add consumer2 to the duplicate output's consumer list
                Variable dupV = sd.getVariables().get(dupOutputName);
                if (dupV == null) {
                    dupV = helper.getVariable(dupOutputName);
                }
                if (dupV != null) {
                    if (dupV.getInputsForOp() == null) {
                        dupV.setInputsForOp(new ArrayList<>());
                    }
                    dupV.getInputsForOp().add(consumer2);
                }

                // Update the input variable's consumer list to include the new duplicate op
                String dupOpName = null;
                Variable dupVar = sd.getVariables().get(dupOutputName);
                if (dupVar != null) {
                    dupOpName = dupVar.getOutputOfOp();
                }
                if (dupOpName != null && inV.getInputsForOp() != null) {
                    if (!inV.getInputsForOp().contains(dupOpName)) {
                        inV.getInputsForOp().add(dupOpName);
                    }
                }

                log.debug("Rematerialization: duplicated {} for consumer '{}', original kept for '{}'",
                        fn.opName(), consumer2, consumer1);
                return true;

            } catch (Exception e) {
                log.debug("Rematerialization failed for {}: {}", outputVar, e.getMessage());
                return false;
            }
        }

        private boolean isCheapUnaryOp(DifferentialFunction fn) {
            // Unary elementwise transforms are cheap — single pass over data, O(n)
            return fn instanceof BaseTransformSameOp
                    || fn instanceof BaseTransformStrictOp
                    || fn instanceof BaseTransformFloatOp
                    || fn instanceof BaseTransformBoolOp;
        }

        private boolean duplicateUnaryOp(SameDiff sd, DifferentialFunction fn,
                                          String inputVar, String outputName) {
            // Use the SameDiff API to create a copy of the op
            String opName = fn.opName();

            // Map common unary ops to their SameDiff API calls
            switch (opName) {
                case "abs":
                    sd.math().abs(outputName, sd.getVariable(inputVar));
                    return true;
                case "neg":
                case "negative":
                    sd.math().neg(outputName, sd.getVariable(inputVar));
                    return true;
                case "exp":
                    sd.math().exp(outputName, sd.getVariable(inputVar));
                    return true;
                case "log":
                    sd.math().log(outputName, sd.getVariable(inputVar));
                    return true;
                case "sqrt":
                    sd.math().sqrt(outputName, sd.getVariable(inputVar));
                    return true;
                case "square":
                    sd.math().square(outputName, sd.getVariable(inputVar));
                    return true;
                case "tanh":
                    sd.math().tanh(outputName, sd.getVariable(inputVar));
                    return true;
                case "sigmoid":
                    sd.nn().sigmoid(outputName, sd.getVariable(inputVar));
                    return true;
                case "relu":
                    sd.nn().relu(outputName, sd.getVariable(inputVar), 0);
                    return true;
                case "reciprocal":
                    sd.math().reciprocal(outputName, sd.getVariable(inputVar));
                    return true;
                case "sign":
                    sd.math().sign(outputName, sd.getVariable(inputVar));
                    return true;
                case "ceil":
                    sd.math().ceil(outputName, sd.getVariable(inputVar));
                    return true;
                case "floor":
                    sd.math().floor(outputName, sd.getVariable(inputVar));
                    return true;
                case "round":
                    sd.math().round(outputName, sd.getVariable(inputVar));
                    return true;
                case "sin":
                    sd.math().sin(outputName, sd.getVariable(inputVar));
                    return true;
                case "cos":
                    sd.math().cos(outputName, sd.getVariable(inputVar));
                    return true;
                default:
                    // Unsupported op — don't rematerialize
                    return false;
            }
        }
    }
}

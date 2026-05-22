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
import org.nd4j.linalg.api.ops.impl.shape.Concat;
import org.nd4j.linalg.api.ops.impl.shape.ExpandDims;
import org.nd4j.linalg.api.ops.impl.shape.Permute;
import org.nd4j.linalg.api.ops.impl.shape.Reshape;
import org.nd4j.linalg.api.ops.impl.shape.Squeeze;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

@Slf4j
public class ShapeFunctionOptimizations extends BaseOptimizerSet {

    /**
     * Fuse [permute1 -> permute2 -> ... -> permuteN] into a single permute op,
     * as long as the intermediate permute outputs aren't needed for another op.
     *
     * For example: permute(permute(x, [1,0,2]), [2,1,0]) can be fused into permute(x, [2,0,1])
     * The combined permutation is computed as: combined[i] = first[second[i]]
     */
    public static class FuseChainedPermutes implements Optimizer {

        private static final Set<Class<? extends DifferentialFunction>> APPLICABLE_OPS = new HashSet<>();
        static {
            APPLICABLE_OPS.add(Permute.class);
        }

        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return APPLICABLE_OPS;
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op, ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            // Type check is now done by GraphOptimizer via getApplicableOpTypes()
            if(!(op.getOp() instanceof Permute))
                return false;

            Permute outerPermute = (Permute) op.getOp();

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.isEmpty())
                return false;

            String input = inputs.get(0);

            // Collect the chain of permute ops (from outer to inner)
            List<SameDiffOp> permuteChain = new ArrayList<>();
            permuteChain.add(op);

            String currInput = input;
            String originalInput = input; // The input to the innermost permute

            while(currInput != null){
                Variable v = helper.getVariable(currInput);
                if (v == null)
                    break;

                // Check if this variable is ONLY used by the next permute in chain
                if(v.getInputsForOp() == null || v.getInputsForOp().size() > 1)
                    break;

                // Check if this variable comes from a Permute op
                String producerOpName = v.getOutputOfOp();
                if (producerOpName == null)
                    break;

                SameDiffOp producerOp = sd.getOps().get(producerOpName);
                if (producerOp == null || !(producerOp.getOp() instanceof Permute))
                    break;

                // Found another permute in the chain
                permuteChain.add(producerOp);

                // Trace back to this permute's input
                List<String> producerInputs = producerOp.getInputsToOp();
                if (producerInputs == null || producerInputs.isEmpty())
                    break;

                originalInput = producerInputs.get(0);
                currInput = originalInput;
            }

            if(permuteChain.size() < 2){
                return false;
            }

            // Compute the combined permutation
            // Start with identity permutation and apply each permute from innermost to outermost
            // Reverse the chain so we go from innermost to outermost
            Collections.reverse(permuteChain);

            long[] combinedPerm = null;

            for (SameDiffOp permuteOp : permuteChain) {
                Permute permute = (Permute) permuteOp.getOp();
                // Get permute dimensions from iArguments
                long[] dims = permute.iArgs();
                if (dims == null || dims.length == 0) {
                    // Can't fuse if dimensions are not constant
                    return false;
                }

                if (combinedPerm == null) {
                    // First permute - just use its dimensions
                    combinedPerm = dims.clone();
                } else {
                    // Compose permutations: combined[i] = current[dims[i]]
                    if (dims.length != combinedPerm.length) {
                        // Mismatched ranks - can't fuse
                        return false;
                    }
                    long[] newCombined = new long[dims.length];
                    for (int i = 0; i < dims.length; i++) {
                        newCombined[i] = combinedPerm[(int) dims[i]];
                    }
                    combinedPerm = newCombined;
                }
            }

            // Check if the combined permutation is identity - if so, we can remove all permutes
            boolean isIdentity = true;
            for (int i = 0; i < combinedPerm.length; i++) {
                if (combinedPerm[i] != i) {
                    isIdentity = false;
                    break;
                }
            }

            try {
                // Get the output of the outermost permute (the op we started with)
                List<String> outerOutputs = op.getOutputsOfOp();
                if (outerOutputs == null || outerOutputs.isEmpty()) {
                    return false;
                }
                String outerOutputVar = outerOutputs.get(0);

                SDVariable inputVar = sd.getVariable(originalInput);
                if (inputVar == null) {
                    return false;
                }

                String replacementName;
                if (isIdentity) {
                    // Combined permutation is identity - just rewire to use original input
                    log.debug("Fusing {} chained permutes into identity (removing all)", permuteChain.size());
                    OptimizationUtils.replaceOpInputsWith(sd, helper, outerOutputVar, originalInput);
                    replacementName = originalInput;
                } else {
                    // Create a single fused permute
                    log.debug("Fusing {} chained permutes into single permute with dims {}",
                            permuteChain.size(), java.util.Arrays.toString(combinedPerm));

                    SDVariable fusedOutput = sd.permute(inputVar, combinedPerm);
                    OptimizationUtils.replaceOpInputsWith(sd, helper, outerOutputVar, fusedOutput.name());
                    replacementName = fusedOutput.name();
                }

                // Temporarily remove outerOutputVar from graph outputs so removeOp/
                // removeVariable guards don't refuse deletion, leaving ghost ops.
                List<String> graphOutputs = sd.outputs();
                boolean wasOutput = graphOutputs != null && graphOutputs.remove(outerOutputVar);

                // Remove all the old permute ops and intermediate variables
                // Go from outermost to innermost
                Collections.reverse(permuteChain); // Back to outer-to-inner order

                for (int i = 0; i < permuteChain.size(); i++) {
                    SameDiffOp permuteOp = permuteChain.get(i);
                    String opName = permuteOp.getName();

                    // Remove the op
                    OptimizationUtils.removeOp(sd, helper, opName);

                    // Remove the output variable (except for the outermost one which was replaced)
                    List<String> outputs = permuteOp.getOutputsOfOp();
                    if (outputs != null) {
                        for (String outVar : outputs) {
                            OptimizationUtils.removeVariable(sd, helper, outVar);
                        }
                    }
                }

                // If outerOutputVar was a graph output, rename the replacement to
                // preserve the original name for downstream lookups.
                if (wasOutput) {
                    if (!replacementName.equals(outerOutputVar)) {
                        sd.renameVariable(replacementName, outerOutputVar);
                    }
                    graphOutputs.add(outerOutputVar);
                }

                return true;
            } catch (Exception e) {
                log.warn("Failed to fuse chained permutes: {}", e.getMessage());
                return false;
            }
        }
    }

    /**
     * Fuse [reshape1 -> reshape2 -> ... -> reshapeN] into a single reshape op,
     * as long as the intermediate reshape ops aren't needed for another op.
     *
     * Multiple consecutive reshapes can be replaced with a single reshape to the final shape.
     */
    public static class FuseChainedReshapes implements Optimizer {

        private static final Set<Class<? extends DifferentialFunction>> APPLICABLE_OPS = new HashSet<>();
        static {
            APPLICABLE_OPS.add(Reshape.class);
        }

        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return APPLICABLE_OPS;
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op, ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof Reshape))
                return false;

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.isEmpty())
                return false;

            String input = inputs.get(0);

            // Collect the chain of reshape ops (from outer to inner)
            List<SameDiffOp> reshapeChain = new ArrayList<>();
            reshapeChain.add(op);

            String currInput = input;
            String originalInput = input;

            while (currInput != null) {
                Variable v = helper.getVariable(currInput);
                if (v == null)
                    break;

                // Check if this variable is ONLY used by the next reshape in chain
                if (v.getInputsForOp() == null || v.getInputsForOp().size() > 1)
                    break;

                String producerOpName = v.getOutputOfOp();
                if (producerOpName == null)
                    break;

                SameDiffOp producerOp = sd.getOps().get(producerOpName);
                if (producerOp == null || !(producerOp.getOp() instanceof Reshape))
                    break;

                reshapeChain.add(producerOp);

                List<String> producerInputs = producerOp.getInputsToOp();
                if (producerInputs == null || producerInputs.isEmpty())
                    break;

                originalInput = producerInputs.get(0);
                currInput = originalInput;
            }

            if (reshapeChain.size() < 2) {
                return false;
            }

            try {
                // Get the final shape from the outermost reshape
                // Shape is stored in iArguments starting at index 1 (index 0 is ordering)
                Reshape outerReshape = (Reshape) op.getOp();
                long[] iArgs = outerReshape.iArgs();
                if (iArgs == null || iArgs.length <= 1) {
                    // Shape is dynamic or not available, can't optimize statically
                    return false;
                }

                // Extract shape from iArgs (skip first element which is ordering)
                long[] finalShape = new long[iArgs.length - 1];
                for (int i = 0; i < finalShape.length; i++) {
                    finalShape[i] = iArgs[i + 1];
                }

                if (finalShape.length == 0) {
                    // Shape is dynamic, can't optimize statically
                    return false;
                }

                List<String> outerOutputs = op.getOutputsOfOp();
                if (outerOutputs == null || outerOutputs.isEmpty()) {
                    return false;
                }
                String outerOutputVar = outerOutputs.get(0);

                SDVariable inputVar = sd.getVariable(originalInput);
                if (inputVar == null) {
                    return false;
                }

                log.debug("Fusing {} chained reshapes into single reshape", reshapeChain.size());

                // Create a single fused reshape
                SDVariable fusedOutput = sd.reshape(inputVar, finalShape);
                OptimizationUtils.replaceOpInputsWith(sd, helper, outerOutputVar, fusedOutput.name());

                // Temporarily remove outerOutputVar from graph outputs so removeOp/
                // removeVariable guards don't refuse deletion, leaving ghost ops.
                List<String> graphOutputs = sd.outputs();
                boolean wasOutput = graphOutputs != null && graphOutputs.remove(outerOutputVar);

                // Remove all the old reshape ops and intermediate variables
                for (SameDiffOp reshapeOp : reshapeChain) {
                    OptimizationUtils.removeOp(sd, helper, reshapeOp.getName());

                    List<String> outputs = reshapeOp.getOutputsOfOp();
                    if (outputs != null) {
                        for (String outVar : outputs) {
                            OptimizationUtils.removeVariable(sd, helper, outVar);
                        }
                    }
                }

                // If outerOutputVar was a graph output, rename the fused variable to
                // preserve the original name for downstream lookups.
                if (wasOutput) {
                    sd.renameVariable(fusedOutput.name(), outerOutputVar);
                    graphOutputs.add(outerOutputVar);
                }

                return true;
            } catch (Exception e) {
                log.warn("Failed to fuse chained reshapes: {}", e.getMessage());
                return false;
            }
        }
    }

    /**
     * Fuse [concat(concat(concat(x,y,dim=D), z, dim=D), a, dim=D)] into a single concat op, concat(x,y,z,a, dim=D)
     * As long as the intermediate outputs aren't needed elsewhere.
     */
    public static class FuseChainedConcatOps implements Optimizer {

        private static final Set<Class<? extends DifferentialFunction>> APPLICABLE_OPS = new HashSet<>();
        static {
            APPLICABLE_OPS.add(Concat.class);
        }

        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return APPLICABLE_OPS;
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op, ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof Concat))
                return false;

            Concat outerConcat = (Concat) op.getOp();
            // Concat dimension is stored as the first iArgument
            long[] outerIArgs = outerConcat.iArgs();
            if (outerIArgs == null || outerIArgs.length == 0) {
                return false;
            }
            int concatDim = (int) outerIArgs[0];

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.size() < 2)
                return false;

            // Collect all inputs for the fused concat, expanding any intermediate concats
            List<String> allInputs = new ArrayList<>();
            List<SameDiffOp> opsToRemove = new ArrayList<>();
            boolean foundFusableConcat = false;

            for (String inputName : inputs) {
                Variable v = helper.getVariable(inputName);
                if (v == null) {
                    allInputs.add(inputName);
                    continue;
                }

                // Check if this input comes from another concat on the same dimension
                // and is only used by this concat
                if (v.getInputsForOp() != null && v.getInputsForOp().size() == 1) {
                    String producerOpName = v.getOutputOfOp();
                    if (producerOpName != null) {
                        SameDiffOp producerOp = sd.getOps().get(producerOpName);
                        if (producerOp != null && producerOp.getOp() instanceof Concat) {
                            Concat innerConcat = (Concat) producerOp.getOp();
                            // Get inner concat dimension from iArgs
                            long[] innerIArgs = innerConcat.iArgs();
                            if (innerIArgs != null && innerIArgs.length > 0) {
                                int innerConcatDim = (int) innerIArgs[0];
                                if (innerConcatDim == concatDim) {
                                    // Found a fusable concat - add its inputs instead
                                    List<String> innerInputs = producerOp.getInputsToOp();
                                    if (innerInputs != null) {
                                        allInputs.addAll(innerInputs);
                                        opsToRemove.add(producerOp);
                                        foundFusableConcat = true;
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                }

                allInputs.add(inputName);
            }

            if (!foundFusableConcat) {
                return false;
            }

            try {
                List<String> outerOutputs = op.getOutputsOfOp();
                if (outerOutputs == null || outerOutputs.isEmpty()) {
                    return false;
                }
                String outerOutputVar = outerOutputs.get(0);

                // Get SDVariables for all inputs
                SDVariable[] inputVars = new SDVariable[allInputs.size()];
                for (int i = 0; i < allInputs.size(); i++) {
                    inputVars[i] = sd.getVariable(allInputs.get(i));
                    if (inputVars[i] == null) {
                        return false;
                    }
                }

                log.debug("Fusing {} concat ops into single concat with {} inputs",
                        opsToRemove.size() + 1, allInputs.size());

                // Create a single fused concat
                SDVariable fusedOutput = sd.concat(concatDim, inputVars);
                OptimizationUtils.replaceOpInputsWith(sd, helper, outerOutputVar, fusedOutput.name());

                // Temporarily remove outerOutputVar from graph outputs so removeOp/
                // removeVariable guards don't refuse deletion, leaving ghost ops.
                List<String> graphOutputs = sd.outputs();
                boolean wasOutput = graphOutputs != null && graphOutputs.remove(outerOutputVar);

                // Remove the outer concat
                OptimizationUtils.removeOp(sd, helper, op.getName());
                List<String> outputs = op.getOutputsOfOp();
                if (outputs != null) {
                    for (String outVar : outputs) {
                        OptimizationUtils.removeVariable(sd, helper, outVar);
                    }
                }

                // Remove the inner concats
                for (SameDiffOp concatOp : opsToRemove) {
                    OptimizationUtils.removeOp(sd, helper, concatOp.getName());
                    outputs = concatOp.getOutputsOfOp();
                    if (outputs != null) {
                        for (String outVar : outputs) {
                            OptimizationUtils.removeVariable(sd, helper, outVar);
                        }
                    }
                }

                // If outerOutputVar was a graph output, rename the fused variable to
                // preserve the original name for downstream lookups.
                if (wasOutput) {
                    sd.renameVariable(fusedOutput.name(), outerOutputVar);
                    graphOutputs.add(outerOutputVar);
                }

                return true;
            } catch (Exception e) {
                log.warn("Failed to fuse chained concats: {}", e.getMessage());
                return false;
            }
        }
    }

    /**
     * Eliminate squeeze(expand_dims(x, axis), axis) → x.
     *
     * <p>Common in ONNX-imported models where unsqueeze/squeeze pairs are
     * inserted for broadcasting compatibility but cancel each other out.</p>
     */
    public static class EliminateSqueezeExpandDims implements Optimizer {

        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return Set.of(Squeeze.class);
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof Squeeze)) return false;
            Squeeze squeeze = (Squeeze) op.getOp();

            long[] squeezeArgs = squeeze.iArgs();
            if (squeezeArgs == null || squeezeArgs.length != 1) return false;
            long squeezeAxis = squeezeArgs[0];

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.isEmpty()) return false;
            List<String> outputs = op.getOutputsOfOp();
            if (outputs == null || outputs.isEmpty()) return false;

            String inputVar = inputs.get(0);
            String outputVar = outputs.get(0);

            // Check if the input was produced by expand_dims
            Variable inputVariable = sd.getVariables().get(inputVar);
            if (inputVariable == null) return false;
            String producerOpName = inputVariable.getOutputOfOp();
            if (producerOpName == null) return false;

            SameDiffOp producerOp = sd.getOps().get(producerOpName);
            if (producerOp == null || !(producerOp.getOp() instanceof ExpandDims)) return false;
            ExpandDims expand = (ExpandDims) producerOp.getOp();

            long[] expandArgs = expand.iArgs();
            if (expandArgs == null || expandArgs.length != 1) return false;
            long expandAxis = expandArgs[0];

            // The axes must match for cancellation
            if (squeezeAxis != expandAxis) return false;

            // The intermediate must only be consumed by this squeeze
            List<String> intermediateUsers = inputVariable.getInputsForOp();
            if (intermediateUsers == null || intermediateUsers.size() != 1) return false;

            // Get the original input (before expand_dims)
            List<String> expandInputs = producerOp.getInputsToOp();
            if (expandInputs == null || expandInputs.isEmpty()) return false;
            String originalInput = expandInputs.get(0);

            log.debug("Eliminating squeeze(expand_dims({}, {}), {}) -> {}", originalInput, expandAxis, squeezeAxis, originalInput);

            OptimizationUtils.replaceOpInputsWith(sd, helper, outputVar, originalInput);

            List<String> graphOutputs = sd.outputs();
            if (graphOutputs != null) {
                for (int i = 0; i < graphOutputs.size(); i++) {
                    if (graphOutputs.get(i).equals(outputVar)) {
                        graphOutputs.set(i, originalInput);
                    }
                }
            }

            OptimizationUtils.removeOp(sd, helper, op.getName());
            OptimizationUtils.removeVariable(sd, helper, outputVar);

            // If expand_dims now has no consumers, remove it too
            Variable intermediateAfter = sd.getVariables().get(inputVar);
            if (intermediateAfter != null) {
                List<String> remaining = intermediateAfter.getInputsForOp();
                if (remaining == null || remaining.isEmpty()) {
                    OptimizationUtils.removeOp(sd, helper, producerOpName);
                    OptimizationUtils.removeVariable(sd, helper, inputVar);
                }
            }

            return true;
        }
    }

    /**
     * Eliminate expand_dims(squeeze(x, axis), axis) → x.
     *
     * <p>The reverse of {@link EliminateSqueezeExpandDims}.</p>
     */
    public static class EliminateExpandDimsSqueeze implements Optimizer {

        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return Set.of(ExpandDims.class);
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof ExpandDims)) return false;
            ExpandDims expand = (ExpandDims) op.getOp();

            long[] expandArgs = expand.iArgs();
            if (expandArgs == null || expandArgs.length != 1) return false;
            long expandAxis = expandArgs[0];

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.isEmpty()) return false;
            List<String> outputs = op.getOutputsOfOp();
            if (outputs == null || outputs.isEmpty()) return false;

            String inputVar = inputs.get(0);
            String outputVar = outputs.get(0);

            Variable inputVariable = sd.getVariables().get(inputVar);
            if (inputVariable == null) return false;
            String producerOpName = inputVariable.getOutputOfOp();
            if (producerOpName == null) return false;

            SameDiffOp producerOp = sd.getOps().get(producerOpName);
            if (producerOp == null || !(producerOp.getOp() instanceof Squeeze)) return false;
            Squeeze squeeze = (Squeeze) producerOp.getOp();

            long[] squeezeArgs = squeeze.iArgs();
            if (squeezeArgs == null || squeezeArgs.length != 1) return false;
            long squeezeAxis = squeezeArgs[0];

            if (squeezeAxis != expandAxis) return false;

            List<String> intermediateUsers = inputVariable.getInputsForOp();
            if (intermediateUsers == null || intermediateUsers.size() != 1) return false;

            List<String> squeezeInputs = producerOp.getInputsToOp();
            if (squeezeInputs == null || squeezeInputs.isEmpty()) return false;
            String originalInput = squeezeInputs.get(0);

            log.debug("Eliminating expand_dims(squeeze({}, {}), {}) -> {}", originalInput, squeezeAxis, expandAxis, originalInput);

            OptimizationUtils.replaceOpInputsWith(sd, helper, outputVar, originalInput);

            List<String> graphOutputs = sd.outputs();
            if (graphOutputs != null) {
                for (int i = 0; i < graphOutputs.size(); i++) {
                    if (graphOutputs.get(i).equals(outputVar)) {
                        graphOutputs.set(i, originalInput);
                    }
                }
            }

            OptimizationUtils.removeOp(sd, helper, op.getName());
            OptimizationUtils.removeVariable(sd, helper, outputVar);

            Variable intermediateAfter = sd.getVariables().get(inputVar);
            if (intermediateAfter != null) {
                List<String> remaining = intermediateAfter.getInputsForOp();
                if (remaining == null || remaining.isEmpty()) {
                    OptimizationUtils.removeOp(sd, helper, producerOpName);
                    OptimizationUtils.removeVariable(sd, helper, inputVar);
                }
            }

            return true;
        }
    }

}

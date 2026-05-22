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
import org.nd4j.linalg.api.ops.impl.reduce.TensorMmul;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

/**
 * Horizontal fusion optimizations: batching multiple independent ops of the same type.
 *
 * Primary pattern: Parallel Matmul Fusion
 * When multiple matmuls share the same activation input but use different constant weight matrices,
 * fuse them into a single matmul with concatenated weights, then split the output.
 *
 * Example (Q/K/V projections in attention):
 *   Q = matmul(X, Wq)    [X: (B,D), Wq: (D,Hq)]
 *   K = matmul(X, Wk)    [X: (B,D), Wk: (D,Hk)]
 *   V = matmul(X, Wv)    [X: (B,D), Wv: (D,Hv)]
 * Becomes:
 *   QKV = matmul(X, concat(Wq, Wk, Wv, axis=1))  [X: (B,D), W: (D, Hq+Hk+Hv)]
 *   Q = slice(QKV, 0, Hq)
 *   K = slice(QKV, Hq, Hk)
 *   V = slice(QKV, Hq+Hk, Hv)
 */
@Slf4j
public class HorizontalFusionOptimizations extends BaseOptimizerSet {

    /**
     * Fuses parallel matmuls that share the same activation input.
     *
     * Requirements:
     * 1. Two or more Mmul ops share the exact same first input (activation)
     * 2. The second input (weight) to each matmul is a CONSTANT
     * 3. All weight matrices have compatible inner dimensions (same D)
     * 4. No transpose flags on the matmul (simple matmul(X, W) pattern)
     *
     * The optimizer anchors on the FIRST Mmul found for a given shared input,
     * scans the full op list for siblings, and fuses the group in one shot.
     */
    public static class FuseParallelMatmuls implements Optimizer {

        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return Collections.singleton(Mmul.class);
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof Mmul)) return false;

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.size() != 2) return false;

            String activationVar = inputs.get(0);
            String weightVar = inputs.get(1);

            // Weight must be a constant
            Variable wv = helper.getVariable(weightVar);
            if (wv == null) wv = sd.getVariables().get(weightVar);
            if (wv == null || wv.getVariable().getVariableType() != VariableType.CONSTANT) return false;

            // Check for transpose — skip if either input is transposed
            Mmul mmul = (Mmul) op.getOp();
            if (isTransposed(mmul)) return false;

            // Find all sibling Mmul ops sharing the same activation input
            Variable av = helper.getVariable(activationVar);
            if (av == null) av = sd.getVariables().get(activationVar);
            if (av == null || av.getInputsForOp() == null) return false;

            List<SameDiffOp> siblings = new ArrayList<>();
            List<String> siblingWeightVars = new ArrayList<>();
            List<String> siblingOutputVars = new ArrayList<>();

            for (String consumerOpName : av.getInputsForOp()) {
                SameDiffOp consumerOp = sd.getOps().get(consumerOpName);
                if (consumerOp == null) continue;
                if (!(consumerOp.getOp() instanceof Mmul)) continue;
                Mmul sibMmul = (Mmul) consumerOp.getOp();
                if (isTransposed(sibMmul)) continue;

                List<String> sibInputs = consumerOp.getInputsToOp();
                if (sibInputs == null || sibInputs.size() != 2) continue;
                if (!sibInputs.get(0).equals(activationVar)) continue;

                String sibWeight = sibInputs.get(1);
                Variable sibWv = helper.getVariable(sibWeight);
                if (sibWv == null) sibWv = sd.getVariables().get(sibWeight);
                if (sibWv == null || sibWv.getVariable().getVariableType() != VariableType.CONSTANT) continue;

                List<String> sibOutputs = consumerOp.getOutputsOfOp();
                if (sibOutputs == null || sibOutputs.isEmpty()) continue;

                siblings.add(consumerOp);
                siblingWeightVars.add(sibWeight);
                siblingOutputVars.add(sibOutputs.get(0));
            }

            // Need at least 2 matmuls to fuse
            if (siblings.size() < 2) return false;

            // Only fuse if the anchor op is the first in the group (avoid double-processing)
            if (!siblings.get(0).getName().equals(op.getName())) return false;

            // Validate all weights have compatible shapes: same rank and same inner dimension
            INDArray firstWeight = getConstantArray(sd, constantArrays, siblingWeightVars.get(0));
            if (firstWeight == null || firstWeight.rank() != 2) return false;
            long innerDim = firstWeight.shape()[0];

            long[] outputWidths = new long[siblings.size()];
            INDArray[] weightArrays = new INDArray[siblings.size()];
            weightArrays[0] = firstWeight;
            outputWidths[0] = firstWeight.shape()[1];

            for (int i = 1; i < siblings.size(); i++) {
                INDArray w = getConstantArray(sd, constantArrays, siblingWeightVars.get(i));
                if (w == null || w.rank() != 2 || w.shape()[0] != innerDim) return false;
                weightArrays[i] = w;
                outputWidths[i] = w.shape()[1];
            }

            // All checks passed — apply the fusion
            log.debug("Horizontal fusion: fusing {} parallel matmuls on shared input '{}'",
                    siblings.size(), activationVar);

            // 1. Concatenate weight matrices along axis 1
            INDArray fusedWeight = Nd4j.concat(1, weightArrays);
            String fusedWeightName = activationVar + "_hfused_weight";
            // Ensure unique name
            int suffix = 0;
            while (sd.hasVariable(fusedWeightName)) {
                fusedWeightName = activationVar + "_hfused_weight_" + (suffix++);
            }
            sd.constant(fusedWeightName, fusedWeight);

            // 2. Create single fused matmul
            String fusedMmulOutName = activationVar + "_hfused_mmul";
            suffix = 0;
            while (sd.hasVariable(fusedMmulOutName)) {
                fusedMmulOutName = activationVar + "_hfused_mmul_" + (suffix++);
            }
            sd.mmul(fusedMmulOutName, sd.getVariable(activationVar), sd.getVariable(fusedWeightName));

            // 3. Create slice ops with temporary names (properly registered in graph)
            List<String> sliceTempNames = new ArrayList<>();
            long offset = 0;
            for (int i = 0; i < siblings.size(); i++) {
                String origOutput = siblingOutputVars.get(i);
                long width = outputWidths[i];

                String tempSliceName = origOutput + "__hfslice";
                suffix = 0;
                while (sd.hasVariable(tempSliceName)) {
                    tempSliceName = origOutput + "__hfslice_" + (suffix++);
                }
                sliceTempNames.add(tempSliceName);

                // Slice along the last dimension of the fused matmul output.
                // The fused matmul output has the same leading dims as the activation,
                // with last dim = sum of all weight output dims.
                // We use begin/end arrays of length 1 to target ONLY the last dimension,
                // and set beginMask=0, endMask=0 (no mask bits) so the single explicit
                // dimension maps to the last axis via a negative stridedSlice index trick.
                //
                // Actually, the safest approach: resolve rank from activation shape or
                // weight shape, and create correctly-sized begin/end arrays.
                int outRank = resolveRankSafe(sd, activationVar, firstWeight);
                log.debug("Horizontal fusion: resolveRank('{}') = {}, weight shape = {}",
                        activationVar, outRank, java.util.Arrays.toString(firstWeight.shape()));

                long[] beginArr = new long[outRank];
                long[] endArr = new long[outRank];
                long[] strideArr = new long[outRank];
                Arrays.fill(strideArr, 1L);
                beginArr[outRank - 1] = offset;
                endArr[outRank - 1] = offset + width;
                // Mask all leading dims (not the last one) for full passthrough
                int leadingMask = (1 << (outRank - 1)) - 1;

                SDVariable fusedVar = sd.getVariable(fusedMmulOutName);
                sd.stridedSlice(tempSliceName, fusedVar, beginArr, endArr, strideArr,
                        leadingMask, leadingMask, 0, 0, 0);

                offset += width;
            }

            // 4. Replace consumers, remove old ops/vars, rename slices to original names
            for (int i = 0; i < siblings.size(); i++) {
                String origOutput = siblingOutputVars.get(i);
                String tempSliceName = sliceTempNames.get(i);

                // Redirect all consumers of the original matmul output to the slice
                OptimizationUtils.replaceOpInputsWith(sd, helper, origOutput, tempSliceName);

                // Temporarily remove from graph outputs so removeOp/removeVariable succeed
                List<String> graphOutputs = sd.outputs();
                boolean wasOutput = graphOutputs != null && graphOutputs.remove(origOutput);

                // Remove the original matmul op and its output variable
                OptimizationUtils.removeOp(sd, helper, siblings.get(i).getName());
                OptimizationUtils.removeVariable(sd, helper, origOutput);

                // Only remove weight constant if no other ops consume it
                String wName = siblingWeightVars.get(i);
                Variable wVar = helper.getVariable(wName);
                if (wVar == null) wVar = sd.getVariables().get(wName);
                if (wVar != null && (wVar.getInputsForOp() == null || wVar.getInputsForOp().isEmpty())) {
                    OptimizationUtils.removeVariable(sd, helper, wName);
                }

                // Rename slice to original output name so downstream name-based lookups work
                sd.renameVariable(tempSliceName, origOutput);

                // Restore in graph outputs with the original name
                if (wasOutput) {
                    graphOutputs.add(origOutput);
                }
            }

            return true;
        }

        private boolean isTransposed(Mmul mmul) {
            // Check transpose flags stored in iArgs: [transposeA, transposeB, transposeResult]
            if (mmul.numIArguments() > 0 && mmul.getIArgument(0) > 0) return true;  // transposeA
            if (mmul.numIArguments() > 1 && mmul.getIArgument(1) > 0) return true;  // transposeB
            if (mmul.numIArguments() > 2 && mmul.getIArgument(2) > 0) return true;  // transposeResult
            return false;
        }

        private INDArray getConstantArray(SameDiff sd, ArrayHolder constantArrays, String varName) {
            INDArray arr = constantArrays.getArray(varName);
            if (arr != null) return arr;
            // Fallback: check SameDiff constant arrays
            try {
                return sd.getArrForVarName(varName);
            } catch (Exception e) {
                return null;
            }
        }

        /**
         * Resolve the rank of the matmul output by checking:
         * 1. The activation variable's shape (from placeholder declarations)
         * 2. Tracing back through producer ops to find known shapes
         * 3. Using the weight matrix shape as fallback:
         *    - For matmul(X, W) where W is 2D [D, H], output rank = activation rank
         *    - Weight is always 2D, so if activation shape is unknown, we know the
         *      output has at least 2 dimensions. For transformers it's typically 3 (batch, seq, hidden).
         */
        private int resolveRank(SameDiff sd, String varName, INDArray weight) {
            // First try: direct shape lookup on the variable itself
            SDVariable directVar = sd.getVariable(varName);
            if (directVar != null) {
                long[] shape = directVar.getShape();
                if (shape != null && shape.length > 0) {
                    return shape.length;
                }
            }

            // Second try: BFS through producer ops to find ANY variable with known shape
            Set<String> visited = new HashSet<>();
            Deque<String> queue = new ArrayDeque<>();
            queue.add(varName);

            while (!queue.isEmpty()) {
                String current = queue.poll();
                if (!visited.add(current)) continue;

                SDVariable v = sd.getVariable(current);
                if (v == null) continue;

                long[] shape = v.getShape();
                if (shape != null && shape.length > 0) {
                    return shape.length;
                }

                // Trace back through producer ops
                Variable vMeta = sd.getVariables().get(current);
                if (vMeta != null && vMeta.getOutputOfOp() != null) {
                    String opKey = vMeta.getOutputOfOp();
                    // Handle ":0" suffix from FlatBuffers dup
                    if (opKey.contains(":")) {
                        opKey = opKey.substring(0, opKey.indexOf(':'));
                    }
                    SameDiffOp producerOp = sd.getOps().get(opKey);
                    if (producerOp == null) {
                        producerOp = sd.getOps().get(vMeta.getOutputOfOp());
                    }
                    if (producerOp != null && producerOp.getInputsToOp() != null) {
                        for (String input : producerOp.getInputsToOp()) {
                            queue.add(input);
                        }
                    }
                }
            }

            // Third try: infer from weight matrix.
            // matmul(X, W) where W=[D,H] → output has same rank as X.
            // Weight is always 2D, so output rank >= 2.
            // For transformers, activations are typically 3D [batch, seq, hidden].
            // Check if any consumer of the activation has known output shapes that hint at rank.
            if (weight != null && weight.rank() == 2) {
                // Count how many other ops use the activation — if any produce known shapes, use those
                Variable avMeta = sd.getVariables().get(varName);
                if (avMeta != null && avMeta.getInputsForOp() != null) {
                    for (String opName : avMeta.getInputsForOp()) {
                        SameDiffOp consOp = sd.getOps().get(opName);
                        if (consOp != null && consOp.getOutputsOfOp() != null) {
                            for (String outVar : consOp.getOutputsOfOp()) {
                                SDVariable ov = sd.getVariable(outVar);
                                if (ov != null) {
                                    long[] oShape = ov.getShape();
                                    if (oShape != null && oShape.length > 0) {
                                        return oShape.length;  // Sibling output has known rank
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Default: assume 2D (standard matmul).
            // matmul output is always at least rank 2.
            return 2;
        }

        /**
         * resolveRank wrapper that enforces a minimum rank of 2 for matmul outputs.
         * A matmul(X, W) where W is 2D always produces output with rank >= 2.
         */
        private int resolveRankSafe(SameDiff sd, String varName, INDArray weight) {
            int rank = resolveRank(sd, varName, weight);
            // matmul output can never be 1D — enforce minimum of 2
            return Math.max(rank, 2);
        }
    }
}

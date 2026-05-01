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

import org.nd4j.autodiff.samediff.ArrayHolder;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.autodiff.samediff.optimize.OptimizationHelper;
import org.nd4j.autodiff.samediff.optimize.Optimizer;
import org.nd4j.enums.WeightsFormat;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2D;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

public class CuDNNFunctionOptimizations extends BaseOptimizerSet {

    protected static final boolean isCudaBackend;

    static {
        String backend = Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend");
        isCudaBackend = "CUDA".equalsIgnoreCase(backend);
    }

    /**
     * https://docs.nvidia.com/deeplearning/sdk/dl-performance-guide/index.html#tensor-layout
     * For tensor cores: we want NHWC layout:
     * Section 7.3.1
     * "Layout choice has an effect on performance, as convolutions implemented for Tensor Cores require NHWC layout and are fastest when input tensors are laid out in NHWC."
     * "To maximize performance, we recommend using NHWC tensor layout."
     *
     * As for weights format: cuDNN docs are vague - but TF uses NCHW+OIHW or NHWC+OHWI
     */
    public static class CudnnConv2dNCHWtoNHWCConversion implements Optimizer {
        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op, ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if(!(op.getOp() instanceof Conv2D))
                return false;

            Conv2D c2d = (Conv2D)op.getOp();
            boolean activationsCorrect = c2d.getConfig().isNHWC();

            List<String> inputs = op.getInputsToOp();
            String wArgName = inputs.get(1);
            WeightsFormat wf = c2d.getConfig().getWeightsFormat();
            boolean weightsCorrect = (wf == WeightsFormat.OYXI);

            if(activationsCorrect && weightsCorrect)
                return false;   //Nothing to do here

            //Now, we need to do 2 things
            //(a) replace NCHW to NHWC for input
            //(b) set weight format to OHWI (OYXI)

            //Step 1 - replace activations
            // Insert permute ops to convert NCHW→NHWC before conv and NHWC→NCHW after.
            // Must create permute ops first, then rewire only the conv op's inputs —
            // NOT a global replaceOpInputsWith, which would rewire the permute's own
            // input and create a DAG cycle.
            if(!activationsCorrect) {
                String inArgName = inputs.get(0);
                SDVariable in = sd.getVariable(inArgName);

                // Create pre-conv permute: NCHW→NHWC
                String nhwcName = in.name() + "_cudnn_nchw_to_nhwc";
                SDVariable nhwc = in.permute(0, 2, 3, 1).rename(nhwcName);

                // Rewire only the conv op's input[0] from the original to the permuted variable
                op.getInputsToOp().set(0, nhwcName);
                // Update variable tracking: conv is now a consumer of nhwc, not of in
                Variable nhwcVar = sd.getVariables().get(nhwcName);
                if (nhwcVar != null) {
                    if (nhwcVar.getInputsForOp() == null) nhwcVar.setInputsForOp(new ArrayList<>());
                    if (!nhwcVar.getInputsForOp().contains(op.getName())) nhwcVar.getInputsForOp().add(op.getName());
                }
                Variable inVar = sd.getVariables().get(inArgName);
                if (inVar != null && inVar.getInputsForOp() != null) {
                    inVar.getInputsForOp().remove(op.getName());
                }

                // Create post-conv permute: NHWC→NCHW
                String convOutName = op.getOutputsOfOp().get(0);
                SDVariable outNhwc = sd.getVariable(convOutName);
                String nchwName = convOutName + "_cudnn_nhwc_to_nchw";
                SDVariable outNchw = outNhwc.permute(0, 3, 1, 2).rename(nchwName);

                // Rewire downstream consumers of the conv output to use the post-permute output.
                // Snapshot the consumer list first so we don't modify while iterating.
                Variable convOutVar = sd.getVariables().get(convOutName);
                if (convOutVar != null && convOutVar.getInputsForOp() != null) {
                    // Find the permute op we just created so we can exclude it
                    Variable nchwVar = sd.getVariables().get(nchwName);
                    String postPermuteOpName = nchwVar != null ? nchwVar.getOutputOfOp() : null;

                    List<String> consumers = new ArrayList<>(convOutVar.getInputsForOp());
                    for (String consumerOp : consumers) {
                        // Skip the post-permute op itself — it must keep convOut as input
                        if (consumerOp.equals(postPermuteOpName)) continue;
                        SameDiffOp consOp = sd.getOps().get(consumerOp);
                        if (consOp != null && consOp.getInputsToOp() != null) {
                            List<String> consInputs = consOp.getInputsToOp();
                            for (int i = 0; i < consInputs.size(); i++) {
                                if (consInputs.get(i).equals(convOutName)) {
                                    consInputs.set(i, nchwName);
                                }
                            }
                            // Update variable tracking
                            convOutVar.getInputsForOp().remove(consumerOp);
                            if (nchwVar != null) {
                                if (nchwVar.getInputsForOp() == null) nchwVar.setInputsForOp(new ArrayList<>());
                                if (!nchwVar.getInputsForOp().contains(consumerOp)) nchwVar.getInputsForOp().add(consumerOp);
                            }
                        }
                    }
                }

                c2d.getConfig().isNHWC(true);
            }

            //Step 2 - convert weights to OYXI [oC, kH, kW, iC] for cuDNN
            if(!weightsCorrect) {
                SDVariable w = sd.getVariable(wArgName);
                String newWname = w.name() + "_cudnn_to_oyxi";

                // Create the permute op first
                SDVariable permutedW;
                if (wf == WeightsFormat.YXIO) {
                    permutedW = w.permute(3, 0, 1, 2).rename(newWname);
                } else if (wf == WeightsFormat.OIYX) {
                    permutedW = w.permute(0, 2, 3, 1).rename(newWname);
                } else {
                    permutedW = null;
                }

                if (permutedW != null) {
                    // Rewire only the conv op's weight input
                    int wIdx = op.getInputsToOp().indexOf(wArgName);
                    if (wIdx >= 0) {
                        op.getInputsToOp().set(wIdx, newWname);
                        Variable newWVar = sd.getVariables().get(newWname);
                        if (newWVar != null) {
                            if (newWVar.getInputsForOp() == null) newWVar.setInputsForOp(new ArrayList<>());
                            if (!newWVar.getInputsForOp().contains(op.getName())) newWVar.getInputsForOp().add(op.getName());
                        }
                        Variable wVar = sd.getVariables().get(wArgName);
                        if (wVar != null && wVar.getInputsForOp() != null) {
                            wVar.getInputsForOp().remove(op.getName());
                        }
                    }
                }

                c2d.getConfig().setWeightsFormat(WeightsFormat.OYXI);
            }

            // Force iArguments to reflect the updated config
            c2d.refreshArgs();

            return true;
        }
    }

    /*
    TODO: Also do pooling2d, batchnorm, etc
     */

}

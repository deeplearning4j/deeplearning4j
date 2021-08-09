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
import org.nd4j.autodiff.samediff.optimize.OptimizationHelper;
import org.nd4j.autodiff.samediff.optimize.Optimizer;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2D;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

public class CuDNNFunctionOptimizations extends BaseOptimizerSet {

    protected static final boolean isCudaBackend;

    static {
        String backend = Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend");
//        isCudaBackend = "CUDA".equalsIgnoreCase(backend);
        isCudaBackend = true;   //For testing only
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
            boolean weightsCorrect = false;
            boolean activationsCorrect = c2d.getConfig().isNHWC();

            if(activationsCorrect && weightsCorrect)
                return false;   //Nothing to do here

            //Now, we need to do 2 things
            //(a) replace NCHW to NHWC for input
            //(b) set weight format to OHWI (OYXI)

            List<String> inputs = op.getInputsToOp();
            String wArgName = inputs.get(1);

            //Step 1 - replace activations
            if(!activationsCorrect) {
                String inArgName = inputs.get(0);
                SDVariable in = sd.getVariable(inArgName);
                //Replace [in -> Conv2d(NCHW) -> out] with [in -> permute -> Conv2d(NHWC) -> permute -> out]
                String newName = in.name() + "_cudnn_nchw_to_nhwc";
                OptimizationUtils.replaceOpInputsWith(sd, in.name(), newName);
                SDVariable nhwc = in.permute(0, 2, 3, 1).rename(newName);              //NCHW to NHWC

                SDVariable outNhwc = sd.getVariable(op.getOutputsOfOp().get(0));
                String newName2 = outNhwc.name() + "_cudnn_nhwc_to_nchw";
                SDVariable outNchw = outNhwc.permute(0, 3, 1, 2).rename(newName2); //NHWC to NCHW

                OptimizationUtils.replaceOpInputsWith(sd, outNhwc.name(), outNchw.name());

                c2d.getConfig().isNHWC(true);
            }

            //Step 2 - replace YXIO weights (default) with OYXI weights
            //We'll just add a permute here, and let other optimizer steps fix the (variable -> permute -> op ==> permutedVariable -> op) part
            if(!weightsCorrect) {
                SDVariable w = sd.getVariable(wArgName);
                String newWname = w.name() + "_cudnn_yxio_to_oyxi";
                OptimizationUtils.replaceOpInputsWith(sd, w.name(), newWname);
                SDVariable wPermuted = w.permute(3, 0, 1, 2).rename(newWname);


                //TODO once config supports weight layout, set it here
            }


            return true;
        }
    }

    /*
    TODO: Also do pooling2d, batchnorm, etc
     */

}

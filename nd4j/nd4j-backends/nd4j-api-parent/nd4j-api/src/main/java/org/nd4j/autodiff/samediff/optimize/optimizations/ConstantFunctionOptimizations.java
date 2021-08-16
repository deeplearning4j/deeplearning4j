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

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.ArrayHolder;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.optimize.OptimizationHelper;
import org.nd4j.autodiff.samediff.optimize.Optimizer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

/**
 * This set of optimizations looks for functions that are applied to constants, and "pre executes" them, so they don't have
 * to be calculated (returning the same value) on each run.
 *
 * @author Alex Black
 */
public class ConstantFunctionOptimizations extends BaseOptimizerSet {

    public static final String CONSTANT_FN_FOLDING_MAX_SIZE = "optimizer.constants.function.max.output.size";
    public static final long CONSTANT_FN_FOLDING_MAX_SIZE_DEFAULT = 4 * 1024 * 1024;    //4MB

    public static class FoldConstantFunctions implements Optimizer {
        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op, ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            //TODO This function needs to check for non-deterministic ops - i.e., random ops - and not apply the optimization to these

            List<String> in = op.getInputsToOp();
            if (in == null || in.isEmpty())
                return false;
            for (String s : in) {
                if (!sd.getVariable(s).isConstant())
                    return false;
            }

            long maxSizeToApply = Long.parseLong(helper.getProperties().getProperty(CONSTANT_FN_FOLDING_MAX_SIZE, String.valueOf(CONSTANT_FN_FOLDING_MAX_SIZE_DEFAULT)));
            //Apply the optimization:
            DifferentialFunction df = op.getOp();
            df.clearArrays();
            for (int i = 0; i < in.size(); i++) {
                String s = in.get(i);
                INDArray arr = sd.getVariable(s).getArr();
                if (df instanceof CustomOp) {
                    ((CustomOp) df).addInputArgument(arr);
                } else {
                    if (i == 0)
                        ((Op) df).setX(arr);
                    else
                        ((Op) df).setY(arr);
                }
            }

            INDArray[] outputs;
            if (df instanceof CustomOp) {
                CustomOp o = (CustomOp) df;
                Nd4j.exec(o);
                outputs = new INDArray[o.numOutputArguments()];
                for (int j = 0; j < outputs.length; j++) {
                    outputs[j] = o.getOutputArgument(j);
                }
            } else {
                Op o = (Op) df;
                Nd4j.exec(o);
                outputs = new INDArray[]{o.z()};
            }
            long sizeCount = 0;
            for (INDArray i : outputs) {
                if (!i.dataType().isNumerical())
                    continue;
                sizeCount += i.length() * i.dataType().width();
            }

            if (sizeCount > maxSizeToApply)
                return false;

            //Convert outputs to constants
            List<String> outputNames = op.getOutputsOfOp();
            for(int i=0; i<outputNames.size(); i++ ){
                String n = outputNames.get(i);
                sd.getVariable(n).setVariableType(VariableType.CONSTANT);
                constantArrays.setArray(n, outputs[i]);
                sd.getVariables().get(n).setOutputOfOp(null);
            }

            //Remove the op
            OptimizationUtils.removeOp(sd, df.getOwnName());

            return true;
        }
    }
}

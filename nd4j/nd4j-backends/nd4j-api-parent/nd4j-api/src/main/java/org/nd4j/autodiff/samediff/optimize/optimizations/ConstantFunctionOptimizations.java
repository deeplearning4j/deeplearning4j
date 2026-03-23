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
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseOp;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.RandomOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

/**
 * This set of optimizations looks for functions that are applied to constants, and "pre executes" them, so they don't have
 * to be calculated (returning the same value) on each run.
 *
 * @author Alex Black
 */
@Slf4j
public class ConstantFunctionOptimizations extends BaseOptimizerSet {

    public static final String CONSTANT_FN_FOLDING_MAX_SIZE = "optimizer.constants.function.max.output.size";
    public static final long CONSTANT_FN_FOLDING_MAX_SIZE_DEFAULT = 4 * 1024 * 1024;    //4MB

    public static class FoldConstantFunctions implements Optimizer {
        private static final String RANDOM_OPS_PACKAGE = "org.nd4j.linalg.api.ops.random";

        private static boolean isNonDeterministicOp(DifferentialFunction df) {
            // Traditional random ops implement RandomOp
            if (df instanceof RandomOp)
                return true;
            // Modern custom random ops (RandomNormal, RandomBernoulli, etc.) extend DynamicCustomOp
            // but live under the random package. Range is deterministic despite being in this package.
            String className = df.getClass().getName();
            return className.startsWith(RANDOM_OPS_PACKAGE) && !className.endsWith(".Range");
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op, ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            DifferentialFunction df = op.getOp();
            if (df == null || isNonDeterministicOp(df))
                return false;

            List<String> in = op.getInputsToOp();
            if (in == null || in.isEmpty())
                return false;
            for (String s : in) {
                var v = sd.getVariable(s);
                if (v == null || !v.isConstant())
                    return false;
            }

            long maxSizeToApply = Long.parseLong(helper.getProperties().getProperty(CONSTANT_FN_FOLDING_MAX_SIZE, String.valueOf(CONSTANT_FN_FOLDING_MAX_SIZE_DEFAULT)));

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
            try {
                if (df instanceof CustomOp) {
                    CustomOp o = (CustomOp) df;
                    OpContext ctx = Nd4j.getExecutioner().buildContext();
                    try {
                        // Transfer all arguments (inputs, iArgs, tArgs, bArgs, dArgs) to the context
                        ctx.setArgsFrom(o);

                        // Pre-allocate output arrays from calculated shapes
                        List<DataBuffer> outputShapes = o.calculateOutputShape(ctx);
                        if (outputShapes != null) {
                            for (int j = 0; j < outputShapes.size(); j++) {
                                INDArray out = Nd4j.createFromDescriptor(outputShapes.get(j));
                                ctx.setOutputArray(j, out);
                            }
                        }

                        // Execute with context
                        Nd4j.getExecutioner().exec(o, ctx);

                        // Retrieve outputs from the context (not the op)
                        int numOut = ctx.numOutputArguments();
                        if (numOut == 0) {
                            return false;
                        }
                        outputs = new INDArray[numOut];
                        for (int j = 0; j < numOut; j++) {
                            outputs[j] = ctx.getOutputArray(j);
                        }
                    } finally {
                        try { ctx.close(); } catch (Exception ignored) {}
                    }
                } else {
                    Op o = (Op) df;
                    OpContext ctx = Nd4j.getExecutioner().buildContext();
                    try {
                        // Set input arrays on the context
                        if (o.y() != null) {
                            ctx.setInputArrays(o.x(), o.y());
                        } else {
                            ctx.setInputArrays(o.x());
                        }

                        // Calculate output shape and allocate output array
                        List<DataBuffer> outputShape = ((BaseOp) o).calculateOutputShape(ctx);
                        if (outputShape != null && !outputShape.isEmpty()) {
                            DataBuffer shapeDesc = outputShape.get(0);
                            INDArray z = Nd4j.createFromDescriptor(shapeDesc);
                            ctx.setOutputArray(0, z);
                        }

                        // Execute with context - this properly handles ScalarOps
                        Nd4j.getExecutioner().exec(o, ctx);

                        outputs = new INDArray[]{ctx.getOutputArray(0)};
                    } finally {
                        try { ctx.close(); } catch (Exception ignored) {}
                    }
                }
            } catch (Exception e) {
                log.debug("Constant folding failed for op {} ({}): {}",
                        df.getOwnName(), df.opName(), e.getMessage());
                return false;
            }
            //Validate outputs before proceeding
            List<String> outputNames = op.getOutputsOfOp();
            if (outputs.length < outputNames.size()) {
                log.debug("Constant folding skipped for op {}: expected {} outputs but got {}",
                        df.getOwnName(), outputNames.size(), outputs.length);
                return false;
            }

            long sizeCount = 0;
            for (INDArray i : outputs) {
                if (i == null)
                    return false;
                if (!i.dataType().isNumerical())
                    continue;
                sizeCount += i.length() * i.dataType().width();
            }

            if (sizeCount > maxSizeToApply)
                return false;

            //Convert outputs to constants
            for(int i=0; i<outputNames.size(); i++ ){
                String n = outputNames.get(i);
                sd.getVariable(n).setVariableType(VariableType.CONSTANT);
                constantArrays.setArray(n, outputs[i]);
                // Use fast O(1) lookup via helper instead of PatriciaTrie O(k)
                helper.getVariable(n).setOutputOfOp(null);
            }

            //Remove the op with fast lookup
            OptimizationUtils.removeOp(sd, helper, df.getOwnName());

            return true;
        }
    }
}

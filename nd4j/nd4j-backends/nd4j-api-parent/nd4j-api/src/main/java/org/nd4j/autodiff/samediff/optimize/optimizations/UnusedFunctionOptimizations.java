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
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.autodiff.samediff.optimize.OptimizationHelper;
import org.nd4j.autodiff.samediff.optimize.Optimizer;
import org.nd4j.common.function.Supplier;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

@Slf4j
public class UnusedFunctionOptimizations extends BaseOptimizerSet {

    public static class RemoveUnusedConstants implements Optimizer {
        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op, ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            //TODO check this once _per graph_ not per op
            List<Variable> variables = new ArrayList<>(sd.getVariables().values());
            boolean anyRemoved = false;
            for(Variable v : variables){
                if(v.getVariable().getVariableType() == VariableType.CONSTANT){
                    List<String> inputFor = v.getInputsForOp();
                    if(inputFor == null || inputFor.isEmpty()){
                        //This constant isn't used...

                        //TODO let's put these on disk instead of keeping them in memory...
                        final INDArray arr = v.getVariable().getArr();
                        helper.arrayRecoveryFunction(v.getName(), new Supplier<INDArray>() {
                            @Override
                            public INDArray get() {
                                return arr;
                            }
                        });

                        sd.getVariables().remove(v.getName());
                        log.info("Removed unused constant: {}", v.getName());
                        anyRemoved = true;
                    }
                }
            }
            return anyRemoved;
        }
    }

}

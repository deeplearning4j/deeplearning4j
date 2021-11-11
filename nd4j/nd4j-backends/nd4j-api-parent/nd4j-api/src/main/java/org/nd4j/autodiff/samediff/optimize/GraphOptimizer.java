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

package org.nd4j.autodiff.samediff.optimize;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.ArrayHolder;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.optimize.debug.OptimizationDebugger;
import org.nd4j.autodiff.samediff.optimize.optimizations.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

/**
 *
 * @author Alex Black
 */
@Slf4j
public class GraphOptimizer {

    public static List<OptimizerSet> defaultOptimizations() {
        return Arrays.<OptimizerSet>asList(
                new UnusedFunctionOptimizations(),
                new ConstantFunctionOptimizations(),
                new IdentityFunctionOptimizations(),
                new ShapeFunctionOptimizations(),
                new UnusedFunctionOptimizations(),
                new CuDNNFunctionOptimizations()
        );
    }

    public static SameDiff optimize(SameDiff graph, String... requiredOutputs){
        return optimize(graph, Arrays.asList(requiredOutputs));
    }

    public static SameDiff optimize(SameDiff graph, List<String> requiredOutputs){
        return optimize(graph, requiredOutputs, defaultOptimizations());
    }

    public static SameDiff optimize(SameDiff graph, List<String> requiredOutputs, List<OptimizerSet> optimizations) {
        return optimize(graph, requiredOutputs, optimizations, null);
    }

    public static SameDiff optimize(SameDiff graph, List<String> requiredOutputs, List<OptimizerSet> optimizations, OptimizationDebugger debugger){
        //TODO Use required outputs - strip unnecessary graph components

        SameDiff sd = graph.dup();

        ArrayHolder cArr = sd.getConstantArrays();
        ArrayHolder vArr = sd.getVariablesArrays();

        OptimizationHelper h = new OptimizationHelper(graph, new OptimizationConfig());    //TODO defaults for config

        for( int i=0; i<3; i++ ) {  //Run multiple times - one run isn't enough, as some more optimizations may need to be applied to the output of earlier optimizations
            for (OptimizerSet s : optimizations) {
                List<Optimizer> l = s.getOptimizers();
                for(Optimizer o : l ){
                    Collection<SameDiffOp> startingOps = new ArrayList<>(sd.getOps().values()); //Create list to avoid concurrent modification exception
                    for(SameDiffOp op : startingOps) {
                        //Because ops might disappear from previous optimization steps, we need to check if the previous op
                        // still exists when iterating...
                        if(!sd.getOps().containsKey(op.getName()))
                            continue;

                        if(debugger != null)
                            debugger.beforeOptimizationCheck(sd, op, o);

                        boolean applied = o.checkAndApply(sd, h, op, cArr, vArr);
                        if(applied) {
                            log.info("Operation was applied: {}", o);
                        }

                        if(debugger != null)
                            debugger.afterOptimizationsCheck(sd, op, o, applied);
                    }
                }
            }
        }

        int constBefore = 0;
        int constAfter = 0;
        int varBefore = 0;
        int varAfter = 0;
        int arrBefore = 0;
        int arrAfter = 0;

        for(SDVariable v : graph.variables()){
            switch(v.getVariableType()){
                case VARIABLE:
                    varBefore++;
                    break;
                case CONSTANT:
                    constBefore++;
                    break;
                case ARRAY:
                    arrBefore++;
                    break;
                case PLACEHOLDER:
                    break;
            }
        }

        for(SDVariable v : sd.variables()){
            switch(v.getVariableType()){
                case VARIABLE:
                    varAfter++;
                    break;
                case CONSTANT:
                    constAfter++;
                    break;
                case ARRAY:
                    arrAfter++;
                    break;
                case PLACEHOLDER:
                    break;
            }
        }


        log.info("Total variables: {} before, {} after", graph.getVariables().size(), sd.getVariables().size());
        log.info("Constant variables: {} before, {} after", constBefore, constAfter);
        log.info("Array type variables: {} before, {} after", arrBefore, arrAfter);
        log.info("Variable type variables: {} before, {} after", varBefore, varAfter);
        log.info("Ops: {} before, {} after", graph.getOps().size(), sd.getOps().size());

        return sd;
    }

}

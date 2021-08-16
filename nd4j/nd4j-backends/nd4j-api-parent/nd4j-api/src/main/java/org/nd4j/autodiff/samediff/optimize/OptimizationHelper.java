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

import lombok.Getter;
import org.nd4j.autodiff.samediff.ArrayHolder;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.array.OptimizedGraphArrayHolder;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.function.Supplier;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Properties;

public class OptimizationHelper {

    private final SameDiff originalGraph;
    @Getter
    private final Properties properties;
    private boolean setConstantHolder = false;
    private boolean setVariableHolder = false;

    public OptimizationHelper(SameDiff originalGraph, Properties properties){
        this.originalGraph = originalGraph;
        this.properties = properties;
    }

    public OptimizationHelper arrayRecoveryFunction(String arrayName, Supplier<INDArray> fn){
        SDVariable v = originalGraph.getVariable(arrayName);
        Preconditions.checkState(v.getVariableType() == VariableType.VARIABLE || v.getVariableType() == VariableType.CONSTANT,
                "Can only set an array recovery function for a variable or a constant");

        if(v.getVariableType() == VariableType.VARIABLE){
            ArrayHolder h = originalGraph.getVariablesArrays();
            if(!setVariableHolder){
                originalGraph.setVariablesArrays(new OptimizedGraphArrayHolder(h));
                h = originalGraph.getVariablesArrays();
                setVariableHolder = true;
            }
            ((OptimizedGraphArrayHolder)h).setFunction(arrayName, fn);
        } else {
            ArrayHolder h = originalGraph.getConstantArrays();
            if(!setConstantHolder){
                originalGraph.setConstantArrays(new OptimizedGraphArrayHolder(h));
                h = originalGraph.getConstantArrays();
                setConstantHolder = true;
            }
            ((OptimizedGraphArrayHolder)h).setFunction(arrayName, fn);
        }

        return this;
    }

}

/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.autodiff.samediff.internal;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.listeners.Listener;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Infer datatypes for all variables
 */
public class DataTypesSession extends AbstractSession<DataType, DataTypesSession.DataTypeCalc> {


    public DataTypesSession(SameDiff sameDiff) {
        super(sameDiff);
    }

    @Override
    public DataType getConstantOrVariable(String variableName) {
        //Variables and constants should always have datatype available
        DataType dt = sameDiff.getVariable(variableName).dataType();
        Preconditions.checkNotNull(dt, "No datatype available for variable %s", variableName);
        return dt;
    }

    @Override
    public DataTypeCalc getAndParameterizeOp(String opName, FrameIter frameIter, Set<VarId> inputs, Set<VarId> allIterInputs, Set<String> constAndPhInputs, Map<String, DataType> placeholderValues) {
        DifferentialFunction df = sameDiff.getFunctionById(opName);
        List<DataType> inputDataTypes = new ArrayList<>();
        for(SDVariable v : df.args()){
            DataType dt = v.dataType();
            if(dt != null){
                inputDataTypes.add(dt);
            } else {
                String s = v.getVarName();
                for(VarId vid : inputs){
                    if(vid.getVariable().equals(s)){
                        DataType dt2 = nodeOutputs.get(vid);
                        Preconditions.checkNotNull(dt2, "No datatype for %s", vid);
                        inputDataTypes.add(dt2);
                    }
                }
            }
        }
        return new DataTypeCalc(df, inputDataTypes);
    }

    @Override
    public DataType[] getOutputs(DataTypeCalc op, FrameIter outputFrameIter, Set<VarId> inputs, Set<VarId> allIterInputs,
                                 Set<String> constAndPhInputs, List<Listener> listeners, boolean training) {
        List<DataType> outTypes = op.getFn().calculateOutputDataTypes(op.getInputTypes());
        return outTypes.toArray(new DataType[outTypes.size()]);
    }

    @AllArgsConstructor
    @Data
    protected static class DataTypeCalc {
        protected final DifferentialFunction fn;
        protected final List<DataType> inputTypes;
    }
}

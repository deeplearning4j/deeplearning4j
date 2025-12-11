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

package org.nd4j.autodiff.samediff.internal;

import lombok.*;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;

import java.util.List;

@AllArgsConstructor
@NoArgsConstructor
@Data   //TODO immutable?
@Builder
@EqualsAndHashCode(exclude = {"gradient", "variableIndex"})
public class Variable {
    protected String name;
    protected SDVariable variable;
    protected List<String> inputsForOp;
    protected List<String> controlDepsForOp;    //if a op control dependency (x -> opY) exists, then "opY" will be in this list
    protected List<String> controlDepsForVar;   //if a variable control dependency (x -> varY) exists, then "varY" will be in this list
    protected String outputOfOp;        //Null for placeholders/constants. For array type SDVariables, the name of the op it's an output of
    protected List<String> controlDeps;     //Control dependencies: name of ops that must be available before this variable is considered available for execution
    protected SDVariable gradient;      //Variable corresponding to the gradient of this variable
    protected int variableIndex = -1;


    /**
     * Returns true if this variable's array is ready/available for use
     */
    public boolean isArrayReady() {
        SameDiff sameDiff = variable.getSameDiff();
        if (variable.isConstant() || variable.getVariableType() == VariableType.VARIABLE) {
            return sameDiff.arrayAlreadyExistsForVarName(name);
        }
        if (variable.isPlaceHolder()) {
            return sameDiff.arrayAlreadyExistsForVarName(name); // User provided
        }
        if (variable.getVariableType() == VariableType.ARRAY) {
            return sameDiff.arrayAlreadyExistsForVarName(name) ||
                    (sameDiff.isEagerMode() && sameDiff.getEagerArrays().hasArray(name));
        }
        return false;
    }


    /**
     * Returns true if this variable's array can be computed (all inputs are ready)
     */
    public boolean canComputeArray() {
        if (isArrayReady()) return true;
        if (variable.getVariableType() != VariableType.ARRAY) return false;


        SameDiff sameDiff = variable.getSameDiff();
        // Check if all inputs are ready
        if (outputOfOp != null) {
            SameDiffOp op = sameDiff.getOps().get(outputOfOp);
            if (op != null && op.getInputsToOp() != null) {
                for (String input : op.getInputsToOp()) {
                    Variable inputVar = sameDiff.getVariables().get(input);
                    if (inputVar == null || !inputVar.isArrayReady()) {
                        return false;
                    }
                }
                return true;
            }
        }
        return false;
    }

    public List<String> getInputsForOp() {
        return inputsForOp;
    }

    public void setInputsForOp(List<String> inputsForOp) {
        this.inputsForOp = inputsForOp;
    }

    public String getOutputOfOp() {
        return outputOfOp;
    }

    public void setOutputOfOp(String outputOfOp) {
        this.outputOfOp = outputOfOp;
    }
}

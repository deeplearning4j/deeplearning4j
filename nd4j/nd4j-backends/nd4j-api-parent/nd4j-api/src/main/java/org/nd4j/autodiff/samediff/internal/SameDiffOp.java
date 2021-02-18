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

import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.autodiff.functions.DifferentialFunction;

import java.util.List;

@Data
@NoArgsConstructor
public class SameDiffOp {
    protected String name;
    protected DifferentialFunction op;          //Actual op (note: should be mutable: i.e., cloneable, no arrays set)
    protected List<String> inputsToOp;          //Name of SDVariables as input
    protected List<String> outputsOfOp;         //Name of SDVariables as output
    protected List<String> controlDeps;         //Name of SDVariables as control dependencies (not data inputs, but need to be available before exec)
    protected List<String> varControlDeps;      //Variables (constants, placeholders, etc) that are control dependencies for this op
    protected List<String> controlDepFor;       //Name of the variables that this op is a control dependency for

    @Builder
    public SameDiffOp(String name, DifferentialFunction op, List<String> inputsToOp, List<String> outputsOfOp, List<String> controlDeps, List<String> varControlDeps, List<String> controlDepFor) {
        this.name = name;
        this.op = op;
        this.inputsToOp = inputsToOp;
        this.outputsOfOp = outputsOfOp;
        this.controlDeps = controlDeps;
        this.varControlDeps = varControlDeps;
        this.controlDepFor = controlDepFor;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public DifferentialFunction getOp() {
        return op;
    }

    public void setOp(DifferentialFunction op) {
        this.op = op;
    }

    public List<String> getInputsToOp() {
        return inputsToOp;
    }

    public void setInputsToOp(List<String> inputsToOp) {
        this.inputsToOp = inputsToOp;
    }

    public List<String> getOutputsOfOp() {
        return outputsOfOp;
    }

    public void setOutputsOfOp(List<String> outputsOfOp) {
        this.outputsOfOp = outputsOfOp;
    }

    public List<String> getControlDeps() {
        return controlDeps;
    }

    public void setControlDeps(List<String> controlDeps) {
        this.controlDeps = controlDeps;
    }

    public List<String> getVarControlDeps() {
        return varControlDeps;
    }

    public void setVarControlDeps(List<String> varControlDeps) {
        this.varControlDeps = varControlDeps;
    }

    public List<String> getControlDepFor() {
        return controlDepFor;
    }

    public void setControlDepFor(List<String> controlDepFor) {
        this.controlDepFor = controlDepFor;
    }
}

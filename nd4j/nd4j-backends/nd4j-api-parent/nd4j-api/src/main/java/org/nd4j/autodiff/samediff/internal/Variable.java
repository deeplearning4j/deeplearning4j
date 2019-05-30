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
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;

import java.util.List;

@AllArgsConstructor
@NoArgsConstructor
@Data   //TODO immutable?
@Builder
public class Variable {
    protected String name;
    protected SDVariable variable;
    protected Object shapeInfo;         //TODO decide type, or if even to include (Variable class should ideally be immutable)
    protected List<String> inputsForOp;
    protected List<String> controlDepsForOp;    //if a op control dependency (x -> opY) exists, then "opY" will be in this list
    protected List<String> controlDepsForVar;   //if a variable control dependency (x -> varY) exists, then "varY" will be in this list
    protected String outputOfOp;        //Null for placeholders/constants. For array type SDVariables, the name of the op it's an output of
    protected List<String> controlDeps;     //Control dependencies: name of variables that must be available before this variable is considered available for execution
    protected int outputOfOpIdx;        //Index of the output for the op (say, variable is output number 2 of op "outputOfOp")
    protected SDVariable gradient;      //Variable corresponding to the gradient of this variable
    protected int variableIndex = -1;
}

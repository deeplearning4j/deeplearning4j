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
import org.nd4j.autodiff.functions.DifferentialFunction;

import java.util.List;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class SameDiffOp {
    protected String name;
    protected DifferentialFunction op;	//Actual op (note: should be mutable: i.e., cloneable, no arrays set)
    protected List<String> inputsToOp;		//Name of SDVariables as input
    protected List<String> outputsOfOp;	    //Name of SDVariables as output
    protected List<String> controlDeps;	    //Name of SDVariables as control dependencies (not data inputs, but need to be available before exec)
    protected List<String> varControlDeps;  //Variables (constants, placeholders, etc) that are control dependencies for this op
    protected List<String> controlDepFor;    //Name of the variables that this op is a control dependency for
}

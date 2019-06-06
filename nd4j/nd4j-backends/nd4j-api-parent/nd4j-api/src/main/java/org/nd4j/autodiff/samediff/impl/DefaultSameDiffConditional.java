/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.nd4j.autodiff.samediff.impl;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SameDiffConditional;
import org.nd4j.autodiff.samediff.SameDiffFunctionDefinition;

import java.util.ArrayList;

public class DefaultSameDiffConditional implements SameDiffConditional {

    @Override
    public SDVariable eval(SameDiff context, SameDiffFunctionDefinition body, SDVariable[] inputVars) {
        context.defineFunction("eval", body, inputVars);
        context.invokeFunctionOn("eval", context);
//        return new ArrayList<>(context.getFunctionInstancesById().values()).get(context.getFunctionInstancesById().size() - 1).outputVariables()[0];
        throw new UnsupportedOperationException("Not yet reimplemented");
    }
}

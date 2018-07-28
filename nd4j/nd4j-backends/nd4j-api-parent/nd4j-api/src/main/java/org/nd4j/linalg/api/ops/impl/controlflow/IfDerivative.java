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

package org.nd4j.linalg.api.ops.impl.controlflow;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SameDiffConditional;
import org.nd4j.autodiff.samediff.SameDiffFunctionDefinition;

import java.util.List;

@NoArgsConstructor
public class IfDerivative extends If {

    private If ifDelegate;

    public IfDerivative(If ifBlock) {
        super(ifBlock);
        this.ifDelegate = ifBlock;
    }

    @Override
    public Boolean getTrueBodyExecuted() {
        return ifDelegate.trueBodyExecuted;
    }


    @Override
    public SameDiffFunctionDefinition getFalseBody() {
        return ifDelegate.falseBody;
    }

    @Override
    public SameDiff getFalseBodyExecution() {
        return ifDelegate.falseBodyExecution;
    }

    @Override
    public String getBlockName() {
        return ifDelegate.blockName;
    }

    @Override
    public String getFalseBodyName() {
        return ifDelegate.falseBodyName;
    }

    @Override
    public SameDiff getLoopBodyExecution() {
        return ifDelegate.loopBodyExecution;
    }

    @Override
    public SameDiffConditional getPredicate() {
        return ifDelegate.getPredicate();
    }

    @Override
    public SameDiff getPredicateExecution() {
        return ifDelegate.predicateExecution;
    }

    @Override
    public List<long[]> calculateOutputShape() {
        return super.calculateOutputShape();
    }

    @Override
    public String opName() {
        return "if_bp";
    }

    @Override
    public List<SDVariable> diff(List<SDVariable> i_v1) {
        throw new UnsupportedOperationException("Unable to take the derivative of the derivative for if");
    }
}

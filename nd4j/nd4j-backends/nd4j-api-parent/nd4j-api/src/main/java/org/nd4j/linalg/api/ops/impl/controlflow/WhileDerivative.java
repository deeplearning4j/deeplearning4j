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
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ops.Op;

/**
 * While loop derivative
 * @author Adam Gibson
 */
@NoArgsConstructor
public class WhileDerivative extends While {
    private While delegate;

    public WhileDerivative(While delegate) {
        super(delegate);
        this.delegate = delegate;
    }



    @Override
    public SameDiffFunctionDefinition getTrueBody() {
        return delegate.trueBody;
    }

    @Override
    public String getTrueBodyName() {
        return delegate.getTrueBodyName();
    }

    @Override
    public SameDiffConditional getPredicate() {
        return delegate.getPredicate();
    }

    @Override
    public SameDiff getPredicateExecution() {
        return delegate.getPredicateExecution();
    }

    @Override
    public SDVariable[] getInputVars() {
        return delegate.getInputVars();
    }

    @Override
    public String getBlockName() {
        return delegate.getBlockName();
    }

    @Override
    public SameDiff getLoopBodyExecution() {
        return delegate.getLoopBodyExecution();
    }

    @Override
    public int getNumLooped() {
        return delegate.getNumLooped();
    }

    @Override
    public String opName() {
        return "while_bp";
    }

    @Override
    public Op.Type opType() {
        return Op.Type.CONDITIONAL;
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow name for while backprop");
    }
}

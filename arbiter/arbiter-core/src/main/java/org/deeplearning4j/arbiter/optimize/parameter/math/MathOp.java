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

package org.deeplearning4j.arbiter.optimize.parameter.math;

import org.deeplearning4j.arbiter.optimize.api.AbstractParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;

import java.util.List;

/**
 * A simple parameter space that implements scalar mathematical operations on another parameter space. This allows you
 * to do things like Y = X * 2, where X is a parameter space. For example, a layer size hyperparameter could be set
 * using this to 2x the size of the previous layer
 *
 * @param <T> Type of the parameter space
 * @author Alex Black
 */
public class MathOp<T extends Number> extends AbstractParameterSpace<T> {

    private ParameterSpace<T> parameterSpace;
    private Op op;
    private T scalar;

    public MathOp(ParameterSpace<T> parameterSpace, Op op, T scalar){
        this.parameterSpace = parameterSpace;
        this.op = op;
        this.scalar = scalar;
    }

    @Override
    public T getValue(double[] parameterValues) {
        T u = parameterSpace.getValue(parameterValues);
        return op.doOp(u, scalar);
    }

    @Override
    public int numParameters() {
        return parameterSpace.numParameters();
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        return parameterSpace.collectLeaves();
    }

    @Override
    public boolean isLeaf() {
        return false;
    }

    @Override
    public void setIndices(int... indices) {
        parameterSpace.setIndices(indices);
    }
}

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

package org.nd4j.linalg.activations.impl;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

/**
 /** Parametrized Rectified Linear Unit (PReLU)
 *
 * f(x) = alpha * x for x < 0, f(x) = x for x >= 0
 *
 * alpha has the same shape as x and is a learned parameter.
 *
 * @author Max Pumperla
 */
@EqualsAndHashCode
@Getter
public class ActivationPReLU extends BaseActivationFunction {

    private INDArray alpha;
    private long[] sharedAxes = null;

    public ActivationPReLU(INDArray alpha) {
        this.alpha = alpha;
    }

    public ActivationPReLU(INDArray alpha, long[] sharedAxes) {
        this.alpha = alpha;
        this.sharedAxes = sharedAxes;
    }

    @Override
    public INDArray getActivation(INDArray in, boolean training) {
        DynamicCustomOp.DynamicCustomOpsBuilder prelu = DynamicCustomOp.builder("prelu")
                .addOutputs(in).addInputs(in, alpha);
        if (sharedAxes != null) {
            for (long axis: sharedAxes) {
                prelu.addIntegerArguments(axis);
            }
        }
        Nd4j.getExecutioner().exec(prelu.build());
        return in;
    }

    @Override
    public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
        assertShape(in, epsilon);
        INDArray dLdalpha = Nd4j.create(alpha.shape());
        DynamicCustomOp.DynamicCustomOpsBuilder preluBp = DynamicCustomOp.builder("prelu_bp")
                .addInputs(in, alpha, epsilon).addOutputs(in, alpha);

        if (sharedAxes != null) {
            for (long axis: sharedAxes) {
                preluBp.addIntegerArguments(axis);
            }
        }
        Nd4j.getExecutioner().exec(preluBp.build());
        return new Pair<>(in, dLdalpha);
    }

    @Override
    public String toString() {
        return "prelu";
    }
}

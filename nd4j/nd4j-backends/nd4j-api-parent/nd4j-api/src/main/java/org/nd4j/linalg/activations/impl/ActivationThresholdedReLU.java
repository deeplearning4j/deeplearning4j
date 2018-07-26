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
import org.nd4j.linalg.api.ops.impl.transforms.LeakyReLU;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.LeakyReLUDerivative;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

/**
 * Thresholded RELU
 *
 * f(x) = x for x > theta, f(x) = 0 otherwise. theta defaults to 1.0
 *
 * @author Max Pumperla
 */
@EqualsAndHashCode
@Getter
public class ActivationThresholdedReLU extends BaseActivationFunction {

    public static final double DEFAULT_THETA = 1.0;
    private double theta = DEFAULT_THETA;

    public ActivationThresholdedReLU() {
        this(DEFAULT_THETA);
    }

    public ActivationThresholdedReLU(double theta) {
        this.theta = theta;
    }

    @Override
    public INDArray getActivation(INDArray in, boolean training) {
        INDArray activation = Nd4j.create(in.shape());
        DynamicCustomOp threshRelu = DynamicCustomOp.builder("thresholdedrelu")
                .addOutputs(activation).addInputs(in)
                .addFloatingPointArguments(theta).build();
        Nd4j.getExecutioner().exec(threshRelu);
        return activation;
    }

    @Override
    public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
        assertShape(in, epsilon);
        INDArray dLdz = Nd4j.create(in.shape());
        DynamicCustomOp threshReluBp = DynamicCustomOp.builder("thresholdedrelu_bp")
                .addInputs(in, epsilon).addOutputs(dLdz).addFloatingPointArguments(theta).build();
        Nd4j.getExecutioner().exec(threshReluBp);
        return new Pair<>(dLdz, null);
    }

    @Override
    public String toString() {
        return "thresholdedrelu(theta=" + theta + ")";
    }
}

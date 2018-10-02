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
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.floating.SoftSign;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.SoftSignDerivative;
import org.nd4j.linalg.factory.Nd4j;

/**
 * f_i(x) = x_i / (1+|x_i|)
 */
@EqualsAndHashCode
@Getter
public class ActivationSoftSign extends BaseActivationFunction {

    @Override
    public INDArray getActivation(INDArray in, boolean training) {
        Nd4j.getExecutioner().execAndReturn(new SoftSign(in));
        return in;
    }

    @Override
    public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
        assertShape(in, epsilon);
        INDArray dLdz = Nd4j.getExecutioner().execAndReturn(new SoftSignDerivative(in));
        dLdz.muli(epsilon);
        return new Pair<>(dLdz, null);
    }

    @Override
    public String toString() {
        return "softsign";
    }
}

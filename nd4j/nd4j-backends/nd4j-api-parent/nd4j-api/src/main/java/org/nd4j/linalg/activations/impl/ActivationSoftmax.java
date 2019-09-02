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
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.SoftmaxBp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

/**
 * f_i(x) = exp(x_i - shift) / sum_j exp(x_j - shift)
 * where shift = max_i(x_i)
 */
@EqualsAndHashCode
@Getter
public class ActivationSoftmax extends BaseActivationFunction {

    @Override
    public INDArray getActivation(INDArray in, boolean training) {
        Nd4j.getExecutioner().execAndReturn((CustomOp) new SoftMax(in, in));
        return in;
    }

    @Override
    public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
        assertShape(in, epsilon);

        Nd4j.getExecutioner().execAndReturn(new SoftmaxBp(in, epsilon, in, -1));

        return new Pair<>(in, null);
    }

    @Override
    public String toString() {
        return "softmax";
    }

}

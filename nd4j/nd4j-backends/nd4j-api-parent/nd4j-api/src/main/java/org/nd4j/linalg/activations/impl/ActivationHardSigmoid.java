/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.activations.impl;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.HardSigmoidBp;
import org.nd4j.linalg.api.ops.impl.transforms.strict.HardSigmoid;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;

/**
 * f(x) = min(1, max(0, 0.2*x + 0.5))
 */
@EqualsAndHashCode(callSuper = false)
@Getter
public class ActivationHardSigmoid extends BaseActivationFunction {

    @Override
    public INDArray getActivation(INDArray in, boolean training) {
        Nd4j.getExecutioner().execAndReturn(new HardSigmoid(in));
        return in;
    }

    @Override
    public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
        assertShape(in, epsilon);
        Nd4j.getExecutioner().execAndReturn(new HardSigmoidBp(in, epsilon, in));

        return new Pair<>(in, null);
    }

    @Override
    public String toString() {
        return "hardsigmoid";
    }
}

/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
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
import org.nd4j.linalg.api.ops.impl.transforms.gradient.RationalTanhBp;
import org.nd4j.linalg.api.ops.impl.transforms.strict.RationalTanh;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;

/**
 * Rational tanh approximation
 * From https://arxiv.org/pdf/1508.01292v3
 *
 * f(x) = 1.7159 * tanh(2x/3)
 * where tanh is approximated as follows,
 * tanh(y) ~ sgn(y) * { 1 - 1/(1+|y|+y^2+1.41645*y^4)}
 *
 * Underlying implementation is in native code
 */
@EqualsAndHashCode(callSuper = false)
@Getter
public class ActivationRationalTanh extends BaseActivationFunction {

    @Override
    public INDArray getActivation(INDArray in, boolean training) {
        Nd4j.getExecutioner().execAndReturn(new RationalTanh(in));
        return in;
    }

    @Override
    public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
        assertShape(in, epsilon);

        Nd4j.getExecutioner().execAndReturn(new RationalTanhBp(in, epsilon, in));

        return new Pair<>(in, null);
    }

    @Override
    public String toString() {
        return "rationaltanh";
    }
}

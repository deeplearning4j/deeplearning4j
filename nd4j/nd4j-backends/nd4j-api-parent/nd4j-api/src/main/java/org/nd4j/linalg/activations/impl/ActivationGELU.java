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
import org.nd4j.linalg.api.ops.impl.transforms.strict.GELU;
import org.nd4j.linalg.api.ops.impl.transforms.strict.GELUDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.strict.PreciseGELU;
import org.nd4j.linalg.api.ops.impl.transforms.strict.PreciseGELUDerivative;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;

/**
 * GELU activation function - Gaussian Error Linear Units
 *
 * @see GELU
 */
@EqualsAndHashCode(callSuper = false)
@Getter
public class ActivationGELU extends BaseActivationFunction {

    private boolean precise;

    public ActivationGELU(boolean precise){
        this.precise = precise;
    }

    public ActivationGELU(){
        this(false);
    }

    @Override
    public INDArray getActivation(INDArray in, boolean training) {
        if (precise)
            Nd4j.getExecutioner().execAndReturn(new PreciseGELU(in, in));
        else
            Nd4j.getExecutioner().execAndReturn(new GELU(in, in));
        return in;
    }

    @Override
    public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
        assertShape(in, epsilon);
        INDArray dLdz;
        if (precise)
            dLdz = Nd4j.getExecutioner().exec(new PreciseGELUDerivative(in, in));
        else
            dLdz = Nd4j.getExecutioner().exec(new GELUDerivative(in, in));

        dLdz.muli(epsilon);
        return new Pair<>(dLdz, null);
    }

    @Override
    public String toString() {
        return "gelu(precise="+precise+")";
    }

}

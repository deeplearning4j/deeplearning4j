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
import org.nd4j.linalg.api.ops.impl.transforms.floating.ELU;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.ELUDerivative;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

/**
 *  f(x) = alpha * (exp(x) - 1.0); x < 0
 *       = x ; x>= 0
 *
 *  alpha defaults to 1, if not specified
 */
@EqualsAndHashCode
@Getter
public class ActivationELU extends BaseActivationFunction {
    public static final double DEFAULT_ALPHA = 1.0;

    private double alpha = DEFAULT_ALPHA;

    public ActivationELU() {
        this(DEFAULT_ALPHA);
    }

    public ActivationELU(double alpha) {
        this.alpha = alpha;
    }

    /*
             = alpha * (exp(x) - 1.0); x < 0
       f(x)
             = x ; x >= 0
     */
    @Override
    public INDArray getActivation(INDArray in, boolean training) {
        // no support in ELU native to override alpha
        if (this.alpha != 1.00) {
            INDArray alphaMultiple = Nd4j.getExecutioner().execAndReturn(new ELU(in.dup()));
            alphaMultiple.muli(alpha);
            BooleanIndexing.replaceWhere(in, alphaMultiple, Conditions.lessThan(0));
        } else {
            Nd4j.getExecutioner().execAndReturn(new ELU(in));
        }
        return in;
    }

    /*
             = alpha * exp(x) ; x < 0
       f'(x)
             = 1 ; x >= 0
     */
    @Override
    public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
        assertShape(in, epsilon);
        // no support in ELU native to override alpha
        if (alpha != 1.00) {
            INDArray dLdz = Nd4j.getExecutioner().execAndReturn(new ELUDerivative(in.dup()));
            dLdz.muli(alpha);
            BooleanIndexing.replaceWhere(dLdz, 1, Conditions.equals(alpha));

            dLdz.muli(epsilon);
            return new Pair<>(dLdz, null);
        }

        else {
            INDArray dLdz = Nd4j.getExecutioner().execAndReturn(new ELUDerivative(in));
            dLdz.muli(epsilon);
            return new Pair<>(dLdz, null);
        }
    }

    @Override
    public String toString() {
        return "elu(alpha=" + alpha + ")";
    }
}

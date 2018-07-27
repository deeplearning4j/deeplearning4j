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

package org.nd4j.weightinit.impl;

import lombok.Builder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.weightinit.BaseWeightInitScheme;
import org.nd4j.weightinit.WeightInit;

import java.util.Arrays;

/**
 * Initialize the weight to one.
 * @author Adam Gibson
 */
public class IdentityInitScheme extends BaseWeightInitScheme {

    @Builder
    public IdentityInitScheme(char order) {
        super(order);
    }

    @Override
    public INDArray doCreate(long[] shape, INDArray paramsView) {
        if(shape.length != 2 || shape[0] != shape[1]){
            throw new IllegalStateException("Cannot use IDENTITY init with parameters of shape "
                    + Arrays.toString(shape) + ": weights must be a square matrix for identity");
        }
        if(order() == Nd4j.order()){
            return Nd4j.eye(shape[0]);
        } else {
            return  Nd4j.createUninitialized(shape, order()).assign(Nd4j.eye(shape[0]));
        }
    }


    @Override
    public WeightInit type() {
        return WeightInit.IDENTITY;
    }
}

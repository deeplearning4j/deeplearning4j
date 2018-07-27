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
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.weightinit.BaseWeightInitScheme;
import org.nd4j.weightinit.WeightInit;

/**
 * Initialize the weights based on a given {@link Distribution}
 * @author Adam Gibson
 */
public class DistributionInitScheme  extends BaseWeightInitScheme {
    private Distribution distribution;

    @Builder
    public DistributionInitScheme(char order, Distribution distribution) {
        super(order);
        this.distribution = distribution;
    }

    @Override
    public INDArray doCreate(long[] shape, INDArray paramsView) {
        return distribution.sample(shape);
    }


    @Override
    public WeightInit type() {
        return WeightInit.DISTRIBUTION;
    }
}

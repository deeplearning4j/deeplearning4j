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
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.weightinit.BaseWeightInitScheme;
import org.nd4j.weightinit.WeightInit;

/**
 * Initialize the weight to:
 * range = = 3.0 / Math.sqrt(fanIn)
 * U(-range,range)
 * @author Adam Gibson
 */
public class VarScalingNormalUniformFanInInitScheme extends BaseWeightInitScheme {

    private double fanIn;

    @Builder
    public VarScalingNormalUniformFanInInitScheme(char order, double fanIn) {
        super(order);
        this.fanIn = fanIn;
    }

    @Override
    public INDArray doCreate(long[] shape, INDArray paramsView) {
        double scalingFanIn = 3.0 / Math.sqrt(fanIn);
        return Nd4j.rand(shape, Nd4j.getDistributions().createUniform(-scalingFanIn, scalingFanIn));
    }


    @Override
    public WeightInit type() {
        return WeightInit.VAR_SCALING_UNIFORM_FAN_IN;
    }
}

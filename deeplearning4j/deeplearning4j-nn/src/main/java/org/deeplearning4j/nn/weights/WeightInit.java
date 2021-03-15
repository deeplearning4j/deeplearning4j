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

package org.deeplearning4j.nn.weights;

import org.deeplearning4j.nn.conf.distribution.Distribution;

public enum WeightInit {
    DISTRIBUTION, ZERO, ONES, SIGMOID_UNIFORM, NORMAL, LECUN_NORMAL, UNIFORM, XAVIER, XAVIER_UNIFORM, XAVIER_FAN_IN, XAVIER_LEGACY, RELU,
    RELU_UNIFORM, IDENTITY, LECUN_UNIFORM, VAR_SCALING_NORMAL_FAN_IN, VAR_SCALING_NORMAL_FAN_OUT, VAR_SCALING_NORMAL_FAN_AVG,
    VAR_SCALING_UNIFORM_FAN_IN, VAR_SCALING_UNIFORM_FAN_OUT, VAR_SCALING_UNIFORM_FAN_AVG;


    /**
     * Create an instance of the weight initialization function
     *
     * @return a new {@link IWeightInit} instance
     */
    public IWeightInit getWeightInitFunction() {
        return getWeightInitFunction(null);
    }

    /**
     * Create an instance of the weight initialization function
     *
     * @param distribution Distribution of the weights (Only used in case DISTRIBUTION)
     * @return a new {@link IWeightInit} instance
     */
    public IWeightInit getWeightInitFunction(Distribution distribution) {
        switch (this) {
            case ZERO:
                return new WeightInitConstant(0.0);
            case ONES:
                return new WeightInitConstant(1.0);
            case DISTRIBUTION:
                return new WeightInitDistribution(distribution);
            case SIGMOID_UNIFORM:
                return new WeightInitSigmoidUniform();
            case LECUN_NORMAL: //Fall through: these 3 are equivalent
            case XAVIER_FAN_IN:
            case NORMAL:
                return new WeightInitNormal();
            case UNIFORM:
                return new WeightInitUniform();
            case XAVIER:
                return new WeightInitXavier();
            case XAVIER_UNIFORM:
                return new WeightInitXavierUniform();
            case XAVIER_LEGACY:
                return new WeightInitXavierLegacy();
            case RELU:
                return new WeightInitRelu();
            case RELU_UNIFORM:
                return new WeightInitReluUniform();
            case IDENTITY:
                return new WeightInitIdentity();
            case LECUN_UNIFORM:
                return new WeightInitLecunUniform();
            case VAR_SCALING_NORMAL_FAN_IN:
                return new WeightInitVarScalingNormalFanIn();
            case VAR_SCALING_NORMAL_FAN_OUT:
                return new WeightInitVarScalingNormalFanOut();
            case VAR_SCALING_NORMAL_FAN_AVG:
                return new WeightInitVarScalingNormalFanAvg();
            case VAR_SCALING_UNIFORM_FAN_IN:
                return new WeightInitVarScalingUniformFanIn();
            case VAR_SCALING_UNIFORM_FAN_OUT:
                return new WeightInitVarScalingUniformFanOut();
            case VAR_SCALING_UNIFORM_FAN_AVG:
                return new WeightInitVarScalingUniformFanAvg();

            default:
                throw new UnsupportedOperationException("Unknown or not supported weight initialization function: " + this);
        }
    }
}

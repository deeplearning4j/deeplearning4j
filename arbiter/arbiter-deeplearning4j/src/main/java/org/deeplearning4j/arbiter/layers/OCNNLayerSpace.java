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

package org.deeplearning4j.arbiter.layers;

import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.arbiter.util.LeafUtils;
import org.deeplearning4j.nn.conf.ocnn.OCNNOutputLayer;


@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor(access = AccessLevel.PRIVATE) //For Jackson JSON/YAML deserialization
public class OCNNLayerSpace extends BaseOutputLayerSpace<OCNNOutputLayer> {


    protected ParameterSpace<Double> nuSpace;
    protected ParameterSpace<Double> initialRValue;
    protected ParameterSpace<Integer> hiddenLayerSize;
    protected ParameterSpace<Integer> windowSize;
    protected ParameterSpace<Boolean> configureR;

    private OCNNLayerSpace(Builder builder) {
        super(builder);

        this.numParameters = LeafUtils.countUniqueParameters(collectLeaves());
        this.nuSpace = builder.nuSpace;
        this.initialRValue = builder.initialRValue;
        this.hiddenLayerSize = builder.hiddenLayerSize;
        this.configureR = builder.configureR;
    }


    @Override
    public OCNNOutputLayer getValue(double[] parameterValues) {
        OCNNOutputLayer.Builder o = new OCNNOutputLayer.Builder();
        setLayerOptionsBuilder(o, parameterValues);
        return o.build();
    }

    protected void setLayerOptionsBuilder(OCNNOutputLayer.Builder builder, double[] values) {
        super.setLayerOptionsBuilder(builder, values);
        builder.nu(nuSpace.getValue(values));
        builder.hiddenLayerSize(hiddenLayerSize.getValue(values));
        builder.initialRValue(initialRValue.getValue(values));
        builder.configureR(configureR.getValue(values));
        builder.windowSize(windowSize.getValue(values));
    }


    public static class Builder extends BaseOutputLayerSpace.Builder<Builder> {
        protected ParameterSpace<Double> nuSpace;
        protected ParameterSpace<Double> initialRValue;
        protected ParameterSpace<Integer> hiddenLayerSize;
        protected ParameterSpace<Integer> windowSize;
        protected ParameterSpace<Boolean> configureR;

        public Builder nu(ParameterSpace<Double> nuSpace) {
            this.nuSpace = nuSpace;
            return this;
        }

        /**
         * Use hiddenLayerSize instead
         * @param numHiddenSpace
         * @return
         */
        @Deprecated
        public Builder numHidden(ParameterSpace<Integer> numHiddenSpace) {
            return hiddenLayerSize(numHiddenSpace);
        }

        /**
         * Use hiddenLayerSize instead
         * @param numHidden
         * @return
         */
        @Deprecated
        public Builder numHidden(int numHidden) {
            return hiddenLayerSize(numHidden);
        }

        public Builder hiddenLayerSize(ParameterSpace<Integer> hiddenLayerSize) {
            this.hiddenLayerSize = hiddenLayerSize;
            return this;
        }

        public Builder hiddenLayerSize(int hiddenLayerSize) {
            this.hiddenLayerSize = new FixedValue<>(hiddenLayerSize);
            return this;
        }

        public Builder nu(double nu) {
            this.nuSpace = new FixedValue<>(nu);
            return this;
        }

        public Builder initialRValue(double initialRValue) {
            this.initialRValue = new FixedValue<>(initialRValue);
            return this;
        }

        public Builder initialRValue(ParameterSpace<Double> initialRValue) {
            this.initialRValue = initialRValue;
            return this;
        }

        public Builder windowSize(int windowSize) {
            this.windowSize = new FixedValue<>(windowSize);
            return this;
        }

        public Builder windowSize(ParameterSpace<Integer> windowSize) {
            this.windowSize = windowSize;
            return this;
        }

        public Builder configureR(boolean configureR) {
            this.configureR = new FixedValue<>(configureR);
            return this;
        }

        public Builder configureR(ParameterSpace<Boolean> configureR) {
            this.configureR = configureR;
            return this;
        }


        @Override
        @SuppressWarnings("unchecked")
        public OCNNLayerSpace build() {
            return new OCNNLayerSpace(this);
        }
    }
}

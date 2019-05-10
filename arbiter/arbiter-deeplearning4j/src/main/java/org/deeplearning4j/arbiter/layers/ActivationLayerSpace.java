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
import org.deeplearning4j.arbiter.adapter.ActivationParameterSpaceAdapter;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.arbiter.util.LeafUtils;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;

/**
 * Layer space for {@link ActivationLayer}
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor(access = AccessLevel.PRIVATE) //For Jackson JSON/YAML deserialization
public class ActivationLayerSpace extends LayerSpace<ActivationLayer> {

    private ParameterSpace<IActivation> activationFunction;

    protected ActivationLayerSpace(Builder builder) {
        super(builder);
        this.activationFunction = builder.activationFunction;
        this.numParameters = LeafUtils.countUniqueParameters(collectLeaves());
    }


    @Override
    public ActivationLayer getValue(double[] parameterValues) {
        ActivationLayer.Builder b = new ActivationLayer.Builder();
        super.setLayerOptionsBuilder(b, parameterValues);
        b.activation(activationFunction.getValue(parameterValues));
        return b.build();
    }

    public static class Builder extends LayerSpace.Builder<Builder> {

        private ParameterSpace<IActivation> activationFunction;

        public Builder activation(Activation activation) {
            return activation(new FixedValue<>(activation));
        }

        public Builder activation(IActivation iActivation) {
            return activationFn(new FixedValue<>(iActivation));
        }

        public Builder activation(ParameterSpace<Activation> activationFunction) {
            return activationFn(new ActivationParameterSpaceAdapter(activationFunction));
        }

        public Builder activationFn(ParameterSpace<IActivation> activationFunction) {
            this.activationFunction = activationFunction;
            return this;
        }

        @SuppressWarnings("unchecked")
        public ActivationLayerSpace build() {
            return new ActivationLayerSpace(this);
        }
    }

    @Override
    public String toString() {
        return toString(", ");
    }

    @Override
    public String toString(String delim) {
        return "ActivationLayerSpace(" + super.toString(delim) + ")";
    }
}

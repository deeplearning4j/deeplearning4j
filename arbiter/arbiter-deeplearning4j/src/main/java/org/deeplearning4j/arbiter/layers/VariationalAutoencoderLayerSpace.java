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
import org.deeplearning4j.nn.conf.layers.variational.LossFunctionWrapper;
import org.deeplearning4j.nn.conf.layers.variational.ReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Layer space for {@link VariationalAutoencoder}
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor(access = AccessLevel.PRIVATE) //For Jackson JSON/YAML deserialization
public class VariationalAutoencoderLayerSpace extends BasePretrainNetworkLayerSpace<VariationalAutoencoder> {

    private ParameterSpace<int[]> encoderLayerSizes;
    private ParameterSpace<int[]> decoderLayerSizes;
    private ParameterSpace<ReconstructionDistribution> outputDistribution;
    private ParameterSpace<IActivation> pzxActivationFn;
    private ParameterSpace<Integer> numSamples;

    protected VariationalAutoencoderLayerSpace(Builder builder) {
        super(builder);

        this.encoderLayerSizes = builder.encoderLayerSizes;
        this.decoderLayerSizes = builder.decoderLayerSizes;
        this.outputDistribution = builder.outputDistribution;
        this.pzxActivationFn = builder.pzxActivationFn;
        this.numSamples = builder.numSamples;

        this.numParameters = LeafUtils.countUniqueParameters(collectLeaves());
    }

    @Override
    public VariationalAutoencoder getValue(double[] parameterValues) {
        VariationalAutoencoder.Builder b = new VariationalAutoencoder.Builder();
        setLayerOptionsBuilder(b, parameterValues);
        return b.build();
    }

    protected void setLayerOptionsBuilder(VariationalAutoencoder.Builder builder, double[] values) {
        super.setLayerOptionsBuilder(builder, values);
        if (encoderLayerSizes != null)
            builder.encoderLayerSizes(encoderLayerSizes.getValue(values));
        if (decoderLayerSizes != null)
            builder.decoderLayerSizes(decoderLayerSizes.getValue(values));
        if (outputDistribution != null)
            builder.reconstructionDistribution(outputDistribution.getValue(values));
        if (pzxActivationFn != null)
            builder.pzxActivationFn(pzxActivationFn.getValue(values));
        if (numSamples != null)
            builder.numSamples(numSamples.getValue(values));
    }

    @Override
    public String toString() {
        return toString(", ");
    }

    @Override
    public String toString(String delim) {
        StringBuilder sb = new StringBuilder("VariationalAutoencoderLayerSpace(");
        if (encoderLayerSizes != null)
            sb.append("encoderLayerSizes: ").append(encoderLayerSizes).append(delim);
        if (decoderLayerSizes != null)
            sb.append("decoderLayerSizes: ").append(decoderLayerSizes).append(delim);
        if (outputDistribution != null)
            sb.append("reconstructionDistribution: ").append(outputDistribution).append(delim);
        if (pzxActivationFn != null)
            sb.append("pzxActivationFn: ").append(pzxActivationFn).append(delim);
        if (numSamples != null)
            sb.append("numSamples: ").append(numSamples).append(delim);
        sb.append(super.toString(delim)).append(")");
        return sb.toString();
    }

    public static class Builder extends BasePretrainNetworkLayerSpace.Builder<Builder> {

        private ParameterSpace<int[]> encoderLayerSizes;
        private ParameterSpace<int[]> decoderLayerSizes;
        private ParameterSpace<ReconstructionDistribution> outputDistribution;
        private ParameterSpace<IActivation> pzxActivationFn;
        private ParameterSpace<Integer> numSamples;


        public Builder encoderLayerSizes(int... encoderLayerSizes) {
            return encoderLayerSizes(new FixedValue<>(encoderLayerSizes));
        }

        public Builder encoderLayerSizes(ParameterSpace<int[]> encoderLayerSizes) {
            this.encoderLayerSizes = encoderLayerSizes;
            return this;
        }

        public Builder decoderLayerSizes(int... decoderLayerSizes) {
            return decoderLayerSizes(new FixedValue<>(decoderLayerSizes));
        }

        public Builder decoderLayerSizes(ParameterSpace<int[]> decoderLayerSizes) {
            this.decoderLayerSizes = decoderLayerSizes;
            return this;
        }

        public Builder reconstructionDistribution(ReconstructionDistribution distribution) {
            return reconstructionDistribution(new FixedValue<>(distribution));
        }

        public Builder reconstructionDistribution(ParameterSpace<ReconstructionDistribution> distribution) {
            this.outputDistribution = distribution;
            return this;
        }

        public Builder lossFunction(IActivation outputActivationFn, LossFunctions.LossFunction lossFunction) {
            return lossFunction(outputActivationFn, lossFunction.getILossFunction());
        }

        public Builder lossFunction(Activation outputActivationFn, LossFunctions.LossFunction lossFunction) {
            return lossFunction(outputActivationFn.getActivationFunction(), lossFunction.getILossFunction());
        }

        public Builder lossFunction(IActivation outputActivationFn, ILossFunction lossFunction) {
            return reconstructionDistribution(new LossFunctionWrapper(outputActivationFn, lossFunction));
        }

        public Builder pzxActivationFn(IActivation activationFunction) {
            return pzxActivationFn(new FixedValue<>(activationFunction));
        }

        public Builder pzxActivationFn(ParameterSpace<IActivation> activationFunction) {
            this.pzxActivationFn = activationFunction;
            return this;
        }

        public Builder pzxActivationFunction(Activation activation) {
            return pzxActivationFn(activation.getActivationFunction());
        }

        public Builder numSamples(int numSamples) {
            return numSamples(new FixedValue<>(numSamples));
        }

        public Builder numSamples(ParameterSpace<Integer> numSamples) {
            this.numSamples = numSamples;
            return this;
        }


        @Override
        public <E extends LayerSpace> E build() {
            return (E) new VariationalAutoencoderLayerSpace(this);
        }

    }
}

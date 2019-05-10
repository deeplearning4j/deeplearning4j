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

import com.google.common.base.Preconditions;
import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.arbiter.adapter.ActivationParameterSpaceAdapter;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.arbiter.optimize.parameter.discrete.DiscreteParameterSpace;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.weightnoise.IWeightNoise;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.shade.jackson.annotation.JsonInclude;

import java.util.Map;

/**
 * BaseLayerSpace contains the common Layer hyperparameters; should match {@link BaseLayer} in terms of features
 *
 * @author Alex Black
 */
@JsonInclude(JsonInclude.Include.NON_NULL)

@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor(access = AccessLevel.PROTECTED) //For Jackson JSON/YAML deserialization
public abstract class BaseLayerSpace<L extends BaseLayer> extends LayerSpace<L> {
    protected ParameterSpace<IActivation> activationFunction;
    protected ParameterSpace<WeightInit> weightInit;
    protected ParameterSpace<Double> biasInit;
    protected ParameterSpace<Distribution> dist;
    protected ParameterSpace<Double> l1;
    protected ParameterSpace<Double> l2;
    protected ParameterSpace<Double> l1Bias;
    protected ParameterSpace<Double> l2Bias;
    protected ParameterSpace<IUpdater> updater;
    protected ParameterSpace<IUpdater> biasUpdater;
    protected ParameterSpace<IWeightNoise> weightNoise;
    protected ParameterSpace<GradientNormalization> gradientNormalization;
    protected ParameterSpace<Double> gradientNormalizationThreshold;
    protected int numParameters;

    @SuppressWarnings("unchecked")
    protected BaseLayerSpace(Builder builder) {
        super(builder);
        this.activationFunction = builder.activationFunction;
        this.weightInit = builder.weightInit;
        this.biasInit = builder.biasInit;
        this.dist = builder.dist;
        this.l1 = builder.l1;
        this.l2 = builder.l2;
        this.l1Bias = builder.l1Bias;
        this.l2Bias = builder.l2Bias;
        this.updater = builder.updater;
        this.biasUpdater = builder.biasUpdater;
        this.weightNoise = builder.weightNoise;
        this.gradientNormalization = builder.gradientNormalization;
        this.gradientNormalizationThreshold = builder.gradientNormalizationThreshold;
    }

    @Override
    public int numParameters() {
        return numParameters;
    }

    @Override
    public boolean isLeaf() {
        return false;
    }

    @Override
    public void setIndices(int... indices) {
        throw new UnsupportedOperationException("Cannot set indices for non-leaf parameter space");
    }


    protected void setLayerOptionsBuilder(BaseLayer.Builder builder, double[] values) {
        super.setLayerOptionsBuilder(builder, values);
        if (activationFunction != null)
            builder.activation(activationFunction.getValue(values));
        if (biasInit != null)
            builder.biasInit(biasInit.getValue(values));
        if (weightInit != null)
            builder.weightInit(weightInit.getValue(values));
        if (dist != null)
            builder.dist(dist.getValue(values));
        if (l1 != null)
            builder.l1(l1.getValue(values));
        if (l2 != null)
            builder.l2(l2.getValue(values));
        if (l1Bias != null)
            builder.l1Bias(l1Bias.getValue(values));
        if (l2Bias != null)
            builder.l2Bias(l2Bias.getValue(values));
        if (updater != null)
            builder.updater(updater.getValue(values));
        if (biasUpdater != null)
            builder.biasUpdater(biasUpdater.getValue(values));
        if (weightNoise != null)
            builder.weightNoise(weightNoise.getValue(values));
        if (gradientNormalization != null)
            builder.gradientNormalization(gradientNormalization.getValue(values));
        if (gradientNormalizationThreshold != null)
            builder.gradientNormalizationThreshold(gradientNormalizationThreshold.getValue(values));
    }


    @Override
    public String toString() {
        return toString(", ");
    }

    protected String toString(String delim) {
        StringBuilder sb = new StringBuilder();

        for (Map.Entry<String, ParameterSpace> e : getNestedSpaces().entrySet()) {
            sb.append(e.getKey()).append(": ").append(e.getValue()).append("\n");
        }
        return sb.toString();
    }

    @SuppressWarnings("unchecked")
    public abstract static class Builder<T> extends LayerSpace.Builder<T> {
        protected ParameterSpace<IActivation> activationFunction;
        protected ParameterSpace<WeightInit> weightInit;
        protected ParameterSpace<Double> biasInit;
        protected ParameterSpace<Distribution> dist;
        protected ParameterSpace<Double> l1;
        protected ParameterSpace<Double> l2;
        protected ParameterSpace<Double> l1Bias;
        protected ParameterSpace<Double> l2Bias;
        protected ParameterSpace<IUpdater> updater;
        protected ParameterSpace<IUpdater> biasUpdater;
        protected ParameterSpace<IWeightNoise> weightNoise;
        protected ParameterSpace<GradientNormalization> gradientNormalization;
        protected ParameterSpace<Double> gradientNormalizationThreshold;

        public T activation(Activation... activations){
            Preconditions.checkArgument(activations.length > 0, "Activations length must be 1 or more");
            if(activations.length == 1){
                return activation(activations[0]);
            }
            return activation(new DiscreteParameterSpace<>(activations));
        }

        public T activation(Activation activation) {
            return activation(new FixedValue<>(activation));
        }

        public T activation(IActivation iActivation) {
            return activationFn(new FixedValue<>(iActivation));
        }

        public T activation(ParameterSpace<Activation> activationFunction) {
            return activationFn(new ActivationParameterSpaceAdapter(activationFunction));
        }

        public T activationFn(ParameterSpace<IActivation> activationFunction) {
            this.activationFunction = activationFunction;
            return (T) this;
        }

        public T weightInit(WeightInit weightInit) {
            return (T) weightInit(new FixedValue<WeightInit>(weightInit));
        }

        public T weightInit(ParameterSpace<WeightInit> weightInit) {
            this.weightInit = weightInit;
            return (T) this;
        }

        public T weightInit(Distribution distribution){
            weightInit(WeightInit.DISTRIBUTION);
            return dist(distribution);
        }

        public T biasInit(double biasInit){
            return biasInit(new FixedValue<>(biasInit));
        }

        public T biasInit(ParameterSpace<Double> biasInit){
            this.biasInit = biasInit;
            return (T) this;
        }

        public T dist(Distribution dist) {
            return dist(new FixedValue<>(dist));
        }

        public T dist(ParameterSpace<Distribution> dist) {
            this.dist = dist;
            return (T) this;
        }

        public T l1(double l1) {
            return l1(new FixedValue<Double>(l1));
        }

        public T l1(ParameterSpace<Double> l1) {
            this.l1 = l1;
            return (T) this;
        }

        public T l2(double l2) {
            return l2(new FixedValue<Double>(l2));
        }

        public T l2(ParameterSpace<Double> l2) {
            this.l2 = l2;
            return (T) this;
        }

        public T l1Bias(double l1Bias) {
            return l1Bias(new FixedValue<Double>(l1Bias));
        }

        public T l1Bias(ParameterSpace<Double> l1Bias) {
            this.l1Bias = l1Bias;
            return (T) this;
        }

        public T l2Bias(double l2Bias) {
            return l2Bias(new FixedValue<>(l2Bias));
        }

        public T l2Bias(ParameterSpace<Double> l2Bias) {
            this.l2Bias = l2Bias;
            return (T) this;
        }

        public T updater(IUpdater updater) {
            return updater(new FixedValue<>(updater));
        }

        public T updater(ParameterSpace<IUpdater> updater) {
            this.updater = updater;
            return (T) this;
        }

        public T biasUpdater(IUpdater biasUpdater) {
            return biasUpdater(new FixedValue<>(biasUpdater));
        }

        public T biasUpdater(ParameterSpace<IUpdater> biasUpdater) {
            this.biasUpdater = biasUpdater;
            return (T) this;
        }

        public T gradientNormalization(GradientNormalization gradientNormalization) {
            return gradientNormalization(new FixedValue<GradientNormalization>(gradientNormalization));
        }

        public T gradientNormalization(ParameterSpace<GradientNormalization> gradientNormalization) {
            this.gradientNormalization = gradientNormalization;
            return (T) this;
        }

        public T gradientNormalizationThreshold(double threshold) {
            return gradientNormalizationThreshold(new FixedValue<>(threshold));
        }

        public T gradientNormalizationThreshold(ParameterSpace<Double> gradientNormalizationThreshold) {
            this.gradientNormalizationThreshold = gradientNormalizationThreshold;
            return (T) this;
        }
    }

}

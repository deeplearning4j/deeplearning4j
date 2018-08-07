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
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;

import java.util.Arrays;
import java.util.List;

/**
 * LayerSpace for batch normalization layers
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor(access = AccessLevel.PRIVATE) //For Jackson JSON/YAML deserialization
public class BatchNormalizationSpace extends FeedForwardLayerSpace<BatchNormalization> {

    protected ParameterSpace<Double> decay;
    protected ParameterSpace<Double> eps;
    protected ParameterSpace<Boolean> isMinibatch;
    protected ParameterSpace<Boolean> lockGammaBeta;
    protected ParameterSpace<Double> gamma;
    protected ParameterSpace<Double> beta;
    protected ParameterSpace<List<LayerConstraint>> constrainBeta;
    protected ParameterSpace<List<LayerConstraint>> constrainGamma;

    private BatchNormalizationSpace(Builder builder) {
        super(builder);
        this.decay = builder.decay;
        this.eps = builder.eps;
        this.isMinibatch = builder.isMinibatch;
        this.lockGammaBeta = builder.lockGammaBeta;
        this.gamma = builder.gamma;
        this.beta = builder.beta;
        this.constrainBeta = builder.betaConstraints;
        this.constrainGamma = builder.gammaConstraints;

        this.numParameters = LeafUtils.countUniqueParameters(collectLeaves());
    }

    @Override
    public BatchNormalization getValue(double[] parameterValues) {
        BatchNormalization.Builder b = new BatchNormalization.Builder();
        setLayerOptionsBuilder(b, parameterValues);
        return b.build();
    }

    protected void setLayerOptionsBuilder(BatchNormalization.Builder builder, double[] values) {
        super.setLayerOptionsBuilder(builder, values);
        if (decay != null)
            builder.decay(decay.getValue(values));
        if (eps != null)
            builder.eps(eps.getValue(values));
        if (isMinibatch != null)
            builder.minibatch(isMinibatch.getValue(values));
        if (lockGammaBeta != null)
            builder.lockGammaBeta(lockGammaBeta.getValue(values));
        if (gamma != null)
            builder.gamma(gamma.getValue(values));
        if (beta != null)
            builder.beta(beta.getValue(values));
        if (constrainBeta != null){
            List<LayerConstraint> c = constrainBeta.getValue(values);
            if(c != null){
                builder.constrainBeta(c.toArray(new LayerConstraint[c.size()]));
            }
        }
        if (constrainGamma != null){
            List<LayerConstraint> c = constrainGamma.getValue(values);
            if(c != null){
                builder.constrainGamma(c.toArray(new LayerConstraint[c.size()]));
            }
        }
    }

    @Override
    public String toString() {
        return toString(", ");
    }

    @Override
    public String toString(String delim) {
        StringBuilder sb = new StringBuilder();
        sb.append("BatchNormalizationSpace(").append(super.toString(delim));
        if (decay != null)
            sb.append("decay: ").append(decay).append(delim);
        if (eps != null)
            sb.append("eps: ").append(eps).append(delim);
        if (isMinibatch != null)
            sb.append("isMinibatch: ").append(isMinibatch).append(delim);
        if (lockGammaBeta != null)
            sb.append("lockGammaBeta: ").append(lockGammaBeta).append(delim);
        if (gamma != null)
            sb.append("gamma: ").append(gamma).append(delim);
        if (beta != null)
            sb.append("beta: ").append(beta).append(delim);
        sb.append(")");
        return sb.toString();
    }

    public static class Builder extends FeedForwardLayerSpace.Builder<Builder> {

        protected ParameterSpace<Double> decay;
        protected ParameterSpace<Double> eps;
        protected ParameterSpace<Boolean> isMinibatch;
        protected ParameterSpace<Boolean> lockGammaBeta;
        protected ParameterSpace<Double> gamma;
        protected ParameterSpace<Double> beta;
        protected ParameterSpace<List<LayerConstraint>> betaConstraints;
        protected ParameterSpace<List<LayerConstraint>> gammaConstraints;

        public Builder minibatch(boolean minibatch) {
            return minibatch(new FixedValue<>(minibatch));
        }

        public Builder minibatch(ParameterSpace<Boolean> minibatch) {
            this.isMinibatch = minibatch;
            return this;
        }

        public Builder gamma(double gamma) {
            return gamma(new FixedValue<>(gamma));
        }

        public Builder gamma(ParameterSpace<Double> gamma) {
            this.gamma = gamma;
            return this;
        }

        public Builder beta(double beta) {
            return beta(new FixedValue<>(beta));
        }

        public Builder beta(ParameterSpace<Double> beta) {
            this.beta = beta;
            return this;
        }

        public Builder eps(double eps) {
            return eps(new FixedValue<>(eps));
        }

        public Builder eps(ParameterSpace<Double> eps) {
            this.eps = eps;
            return this;
        }

        public Builder decay(double decay) {
            return decay(new FixedValue<Double>(decay));
        }

        public Builder decay(ParameterSpace<Double> decay) {
            this.decay = decay;
            return this;
        }

        public Builder lockGammaBeta(boolean lockGammaBeta) {
            return lockGammaBeta(new FixedValue<>(lockGammaBeta));
        }

        public Builder lockGammaBeta(ParameterSpace<Boolean> lockGammaBeta) {
            this.lockGammaBeta = lockGammaBeta;
            return this;
        }

        public Builder constrainBeta(LayerConstraint... constraints) {
            return constrainBeta(new FixedValue<>(Arrays.asList(constraints)));
        }

        public Builder constrainBeta(ParameterSpace<List<LayerConstraint>> constraints) {
            this.betaConstraints = constraints;
            return this;
        }

        public Builder constrainGamma(LayerConstraint... constraints) {
            return constrainGamma(new FixedValue<>(Arrays.asList(constraints)));
        }

        public Builder constrainGamma(ParameterSpace<List<LayerConstraint>> constraints) {
            this.gammaConstraints = constraints;
            return this;
        }


        @Override
        public BatchNormalizationSpace build() {
            return new BatchNormalizationSpace(this);
        }
    }
}

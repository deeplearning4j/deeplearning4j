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

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;

import java.util.Arrays;
import java.util.List;

@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor //For Jackson JSON/YAML deserialization
public abstract class FeedForwardLayerSpace<L extends FeedForwardLayer> extends BaseLayerSpace<L> {
    protected ParameterSpace<Integer> nIn;
    protected ParameterSpace<Integer> nOut;
    protected ParameterSpace<List<LayerConstraint>> constrainWeights;
    protected ParameterSpace<List<LayerConstraint>> constrainBias;
    protected ParameterSpace<List<LayerConstraint>> constrainAll;


    protected FeedForwardLayerSpace(Builder builder) {
        super(builder);
        nIn = builder.nIn;
        nOut = builder.nOut;
        constrainWeights = builder.constrainWeights;
        constrainBias = builder.constrainBias;
        constrainAll = builder.constrainAll;
    }

    protected void setLayerOptionsBuilder(FeedForwardLayer.Builder builder, double[] values) {
        super.setLayerOptionsBuilder(builder, values);
        if (nIn != null)
            builder.nIn(nIn.getValue(values));
        if (nOut != null)
            builder.nOut(nOut.getValue(values));
        if (constrainWeights != null){
            List<LayerConstraint> c = constrainWeights.getValue(values);
            if(c != null){
                builder.constrainWeights(c.toArray(new LayerConstraint[c.size()]));
            }
        }
        if (constrainBias != null){
            List<LayerConstraint> c = constrainBias.getValue(values);
            if(c != null){
                builder.constrainBias(c.toArray(new LayerConstraint[c.size()]));
            }
        }
        if (constrainAll != null){
            List<LayerConstraint> c = constrainAll.getValue(values);
            if(c != null){
                builder.constrainAllParameters(c.toArray(new LayerConstraint[c.size()]));
            }
        }

    }


    public abstract static class Builder<T> extends BaseLayerSpace.Builder<T> {

        protected ParameterSpace<Integer> nIn;
        protected ParameterSpace<Integer> nOut;
        protected ParameterSpace<List<LayerConstraint>> constrainWeights;
        protected ParameterSpace<List<LayerConstraint>> constrainBias;
        protected ParameterSpace<List<LayerConstraint>> constrainAll;

        public T nIn(int nIn) {
            return nIn(new FixedValue<>(nIn));
        }

        public T nIn(ParameterSpace<Integer> nIn) {
            this.nIn = nIn;
            return (T) this;
        }

        public T nOut(int nOut) {
            return nOut(new FixedValue<>(nOut));
        }

        public T nOut(ParameterSpace<Integer> nOut) {
            this.nOut = nOut;
            return (T) this;
        }

        public T constrainWeights(LayerConstraint... constraints){
            return constrainWeights(new FixedValue<List<LayerConstraint>>(Arrays.asList(constraints)));
        }

        public T constrainWeights(ParameterSpace<List<LayerConstraint>> constraints){
            this.constrainWeights = constraints;
            return (T) this;
        }

        public T constrainBias(LayerConstraint... constraints){
            return constrainBias(new FixedValue<List<LayerConstraint>>(Arrays.asList(constraints)));
        }

        public T constrainBias(ParameterSpace<List<LayerConstraint>> constraints){
            this.constrainBias = constraints;
            return (T) this;
        }

        public T constrainAllParams(LayerConstraint... constraints){
            return constrainAllParams(new FixedValue<List<LayerConstraint>>(Arrays.asList(constraints)));
        }

        public T constrainAllParams(ParameterSpace<List<LayerConstraint>> constraints){
            this.constrainAll = constraints;
            return (T) this;
        }
    }

    @Override
    public String toString() {
        return toString(", ");
    }

    @Override
    protected String toString(String delim) {
        StringBuilder sb = new StringBuilder();
        if (nIn != null)
            sb.append("nIn: ").append(nIn).append(delim);
        if (nOut != null)
            sb.append("nOut: ").append(nOut).append(delim);
        if (constrainWeights != null)
            sb.append("constrainWeights: ").append(constrainWeights).append(delim);
        if (constrainBias != null)
            sb.append("constrainBias: ").append(constrainBias).append(delim);
        if (constrainAll != null)
            sb.append("constrainAllParams: ").append(constrainAll).append(delim);
        sb.append(super.toString(delim));
        return sb.toString();
    }

}

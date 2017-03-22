/*-
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */
package org.deeplearning4j.arbiter.layers;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.List;

@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor //For Jackson JSON/YAML deserialization
public abstract class FeedForwardLayerSpace<L extends FeedForwardLayer> extends LayerSpace<L> {
    protected ParameterSpace<Integer> nIn;
    protected ParameterSpace<Integer> nOut;


    protected FeedForwardLayerSpace(Builder builder) {
        super(builder);
        nIn = builder.nIn;
        nOut = builder.nOut;
    }

    protected void setLayerOptionsBuilder(FeedForwardLayer.Builder builder, double[] values){
        super.setLayerOptionsBuilder(builder, values);
        if(nIn != null) builder.nIn(nIn.getValue(values));
        if(nOut != null) builder.nOut(nOut.getValue(values));
    }

    @Override
    public List<ParameterSpace> collectLeaves(){
        List<ParameterSpace> list = super.collectLeaves();
        if(nIn != null) list.addAll(nIn.collectLeaves());
        if(nOut != null) list.addAll(nOut.collectLeaves());
        return list;
    }


    public abstract static class Builder<T> extends LayerSpace.Builder<T> {

        protected ParameterSpace<Integer> nIn;
        protected ParameterSpace<Integer> nOut;

        public T nIn(int nIn){
            return nIn(new FixedValue<>(nIn));
        }

        public T nIn(ParameterSpace<Integer> nIn){
            this.nIn = nIn;
            return (T)this;
        }

        public T nOut(int nOut){
            return nOut(new FixedValue<>(nOut));
        }

        public T nOut(ParameterSpace<Integer> nOut){
            this.nOut = nOut;
            return (T)this;
        }
    }

    @Override
    public String toString(){
        return toString(", ");
    }

    @Override
    protected String toString(String delim){
        StringBuilder sb = new StringBuilder();
        if(nIn != null) sb.append("nIn: ").append(nIn).append(delim);
        if(nOut != null) sb.append("nOut: ").append(nOut).append(delim);
        sb.append(super.toString(delim));
        return sb.toString();
    }

}

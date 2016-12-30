/*
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

import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.nn.conf.layers.BaseOutputLayer;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.util.List;

/**
 * @param <L>    Type of the (concrete) output layer
 */
public abstract class BaseOutputLayerSpace<L extends BaseOutputLayer> extends FeedForwardLayerSpace<L>{

    protected ParameterSpace<LossFunction> lossFunction;
    protected ParameterSpace<ILossFunction> iLossFunction;

    protected BaseOutputLayerSpace(Builder builder){
        super(builder);
        this.lossFunction = builder.lossFunction;
        this.iLossFunction = builder.iLossFunction;
    }

    protected void setLayerOptionsBuilder(BaseOutputLayer.Builder builder, double[] values){
        super.setLayerOptionsBuilder(builder,values);
        if(lossFunction != null) builder.lossFunction(lossFunction.getValue(values));
        if(iLossFunction != null) builder.lossFunction(iLossFunction.getValue(values));
    }

    @Override
    public List<ParameterSpace> collectLeaves(){
        List<ParameterSpace> list = super.collectLeaves();
        if(lossFunction != null) list.addAll(lossFunction.collectLeaves());
        if(iLossFunction != null) list.addAll(iLossFunction.collectLeaves());
        return list;
    }

    @SuppressWarnings("unchecked")
    public static abstract class Builder<T> extends FeedForwardLayerSpace.Builder<T>{

        protected ParameterSpace<LossFunction> lossFunction;
        protected ParameterSpace<ILossFunction> iLossFunction;


        public T lossFunction(LossFunction lossFunction){
            return lossFunction(new FixedValue<>(lossFunction));
        }

        public T lossFunction(ParameterSpace<LossFunction> lossFunction){
            this.lossFunction = lossFunction;
            return (T)this;
        }

        public T iLossFunction(ILossFunction lossFunction) {
            return iLossFunction(new FixedValue<>(lossFunction));
        }

        public T iLossFunction(ParameterSpace<ILossFunction> lossFunction){
            this.iLossFunction = lossFunction;
            return (T)this;
        }
    }

}

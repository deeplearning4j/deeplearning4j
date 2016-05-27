/*
 *
 *  * Copyright 2015 Skymind,Inc.
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

package org.deeplearning4j.nn.gradient;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

/**
 * Default gradient implementation. Basically lookup table
 * for ndarrays
 *
 * @author Adam Gibson
 */

public class DefaultGradient implements Gradient {
    public static final char DEFAULT_FLATTENING_ORDER = 'f';
    private Map<String,INDArray> gradients = new LinkedHashMap<>();
    private Map<String,Character> flatteningOrders;
    private INDArray flattenedGradient;

    public DefaultGradient(){ }

    public DefaultGradient(INDArray flattenedGradient){
        this.flattenedGradient = flattenedGradient;
    }

    @Override
    public Map<String, INDArray> gradientForVariable() {
        return gradients;
    }

    @Override
    public INDArray gradient(List<String> order) {
        List<INDArray> toFlatten = new ArrayList<>();
        if(flatteningOrders == null) {
            for (String s : order) {
                if (!gradients.containsKey(s)) continue;
                toFlatten.add(gradients.get(s));
            }
        } else {
            for(String s : order){
                if (!gradients.containsKey(s)) continue;
                if(flatteningOrders != null && flatteningOrders.containsKey(s) && flatteningOrders.get(s) != DEFAULT_FLATTENING_ORDER){
                    //Arrays with non-default order get flattened to row vector first, then everything is flattened to f order
                    //TODO revisit this, and make more efficient
                    toFlatten.add(Nd4j.toFlattened(flatteningOrders.get(s),gradients.get(s)));
                } else {
                    toFlatten.add(gradients.get(s));
                }
            }
        }
        return Nd4j.toFlattened(DEFAULT_FLATTENING_ORDER, toFlatten);
    }

    @Override
    public INDArray gradient() {
        if(flattenedGradient != null) return flattenedGradient;

        if(flatteningOrders != null){
            //Arrays with non-default order get flattened to row vector first, then everything is flattened to f order
            //TODO revisit this, and make more efficient
            List<INDArray> toFlatten = new ArrayList<>();
            for(Map.Entry<String,INDArray> entry : gradients.entrySet()){
                if(flatteningOrders.containsKey(entry.getKey()) && flatteningOrders.get(entry.getKey()) != DEFAULT_FLATTENING_ORDER){
                    //Specific flattening order for this array, that isn't the default
                    toFlatten.add(Nd4j.toFlattened(flatteningOrders.get(entry.getKey()),entry.getValue()));
                } else {
                    //default flattening order for this array
                    toFlatten.add(entry.getValue());
                }
            }
            flattenedGradient = Nd4j.toFlattened(DEFAULT_FLATTENING_ORDER, toFlatten);
        } else {
            //Standard case: flatten all to f order
            flattenedGradient = Nd4j.toFlattened(DEFAULT_FLATTENING_ORDER, gradients.values());
        }
        return flattenedGradient;
    }

    @Override
    public void clear() {
        gradients.clear();
    }

    @Override
    public INDArray getGradientFor(String variable) {
        return gradients.get(variable);
    }

    @Override
    public INDArray setGradientFor(String variable, INDArray newGradient) {
        return gradients.put(variable, newGradient);
    }

    @Override
    public INDArray setGradientFor(String variable, INDArray gradient, Character flatteningOrder) {
        INDArray last = setGradientFor(variable,gradient);

        if(flatteningOrder != null){
            if(flatteningOrders == null) flatteningOrders = new LinkedHashMap<>();
            flatteningOrders.put(variable,flatteningOrder);
        }
        return last;
    }

    @Override
    public Character flatteningOrderForVariable(String variable) {
        if(flatteningOrders == null) return null;
        return flatteningOrders.get(variable);
    }


    @Override
    public String toString() {
        return "DefaultGradient{" +
                "gradients=" + gradients +
                (flatteningOrders != null ? flatteningOrders : "") +
                '}';
    }
}

package org.nd4j.linalg.activations;

import org.nd4j.linalg.activations.impl.*;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by susaneraly on 12/8/16.
 */
public enum Activation {
    TANH,
    SIGMOID,
    IDENTITY,
    LEAKYRELU,
    RELU,
    RRELU,
    SOFTMAX,
    SOFTSIGN;

    private static final Map<String,Activation> nameMap = initNameMap();

    private static Map<String,Activation> initNameMap(){
        Map<String,Activation> map = new HashMap<>();
        for(Activation a : values()){
            map.put(a.name().toLowerCase(), a);
        }
        return map;
    }

    public IActivation getActivationFunction() {
        switch(this) {
            case TANH:
                return new ActivationTanH();
            case SIGMOID:
                return new ActivationSigmoid();
            case IDENTITY:
                return new ActivationIdentity();
            case LEAKYRELU:
                return new ActivationLReLU();
            case RELU:
                return new ActivationReLU();
            case SOFTMAX:
                return new ActivationSoftmax();
            case RRELU:
                return new ActivationRReLU();
            case SOFTSIGN:
                return new ActivationSoftSign();
            default:
                throw new UnsupportedOperationException("Unknown or not supported activation function: " + this);
        }
    }

    public static Activation fromString(String name){
        Activation a = nameMap.get(name);
        if(a == null){
            throw new RuntimeException("Unknown activation function: " + name);
        }
        return a;
    }

}

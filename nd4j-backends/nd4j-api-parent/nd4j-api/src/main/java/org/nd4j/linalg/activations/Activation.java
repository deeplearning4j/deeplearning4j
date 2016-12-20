package org.nd4j.linalg.activations;

import org.nd4j.linalg.activations.impl.*;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by susaneraly on 12/8/16.
 */
public enum Activation {
    CUBE,
    ELU,
    HARDSIGMOID,
    HARDTANH,
    IDENTITY,
    LEAKYRELU,
    RELU,
    RRELU,
    SIGMOID,
    SOFTMAX,
    SOFTPLUS,
    SOFTSIGN,
    TANH;

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
            case CUBE:
                return new ActivationCube();
            case ELU:
                return new ActivationELU();
            case HARDSIGMOID:
                return new ActivationHardSigmoid();
            case HARDTANH:
                return new ActivationHardTanH();
            case IDENTITY:
                return new ActivationIdentity();
            case LEAKYRELU:
                return new ActivationLReLU();
            case RELU:
                return new ActivationReLU();
            case RRELU:
                return new ActivationRReLU();
            case SIGMOID:
                return new ActivationSigmoid();
            case SOFTMAX:
                return new ActivationSoftmax();
            case SOFTPLUS:
                return new ActivationSoftPlus();
            case SOFTSIGN:
                return new ActivationSoftSign();
            case TANH:
                return new ActivationTanH();
            default:
                throw new UnsupportedOperationException("Unknown or not supported activation function: " + this);
        }
    }

    public static Activation fromString(String name){
        Activation a = nameMap.get(name.toLowerCase());
        if(a == null){
            throw new RuntimeException("Unknown activation function: " + name);
        }
        return a;
    }

}

package org.deeplearning4j.rl4j.network;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;

public class NeuralNetOutput {
    private final HashMap<String, INDArray> outputs = new HashMap<String, INDArray>();

    public void put(String key, INDArray output) {
        outputs.put(key, output);
    }

    public INDArray get(String key) {
        INDArray result = outputs.get(key);
        if(result == null) {
            throw new IllegalArgumentException(String.format("There is no element with key '%s' in the neural net output.", key));
        }
        return result;
    }
}

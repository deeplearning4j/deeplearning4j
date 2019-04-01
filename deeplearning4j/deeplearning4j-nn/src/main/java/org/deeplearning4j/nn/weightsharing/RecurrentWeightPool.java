package org.deeplearning4j.nn.weightsharing;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.nd4j.linalg.api.ndarray.INDArray;

public class RecurrentWeightPool extends WeightPool {
    public Map<String, INDArray> stateMap = new ConcurrentHashMap<>();
    public Map<String, INDArray> tBpttStateMap = new ConcurrentHashMap<>();
}

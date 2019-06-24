package org.deeplearning4j.rl4j.observation;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface PoolContentConverter {
    INDArray convert(INDArray[] poolContent);
}

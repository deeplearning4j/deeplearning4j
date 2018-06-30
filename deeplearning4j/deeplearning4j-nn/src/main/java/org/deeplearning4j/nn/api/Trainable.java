package org.deeplearning4j.nn.api;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

public interface Trainable {

    TrainingConfig getConfig();

    int numParams();

    INDArray params();

    Map<String,INDArray> paramTable(boolean backpropOnly);

    INDArray getGradientsViewArray();

}

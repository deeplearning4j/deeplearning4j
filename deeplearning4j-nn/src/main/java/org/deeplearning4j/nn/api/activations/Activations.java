package org.deeplearning4j.nn.api.activations;

import org.deeplearning4j.nn.api.MaskState;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface Activations {

    int size();

    INDArray get(int idx);

    INDArray getMask(int idx);

    MaskState getMaskState(int idx);

    void set(int idx, INDArray activations);

    void setMask(int idx, INDArray mask);

    void setMaskState(int idx, MaskState maskState);

    void clear();


    INDArray[] getAsArray();

    INDArray[] getMaskAsArray();

    MaskState[] getMaskStateAsArray();

}

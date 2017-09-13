package org.deeplearning4j.nn.api.gradients;

import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface Gradients {

    int size();

    INDArray getActivationGrad(int idx);

    void setActivationGrad(int idx, INDArray activationGradient);

    Gradient getParameterGradients();

    void setParameterGradients(Gradient gradient);

    void clear();

    INDArray[] getActivationGradAsArray();
}

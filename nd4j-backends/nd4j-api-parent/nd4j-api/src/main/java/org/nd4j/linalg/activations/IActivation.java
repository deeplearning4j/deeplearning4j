package org.nd4j.linalg.activations;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.activations.impl.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonAutoDetect;
import org.nd4j.shade.jackson.annotation.JsonSubTypes;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;

/**
 * Interface for implementing custom activation functions
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
@JsonSubTypes(value = {@JsonSubTypes.Type(value = ActivationCube.class, name = "Cube"),
                @JsonSubTypes.Type(value = ActivationELU.class, name = "ELU"),
                @JsonSubTypes.Type(value = ActivationHardSigmoid.class, name = "HardSigmoid"),
                @JsonSubTypes.Type(value = ActivationHardTanH.class, name = "HardTanh"),
                @JsonSubTypes.Type(value = ActivationIdentity.class, name = "Identity"),
                @JsonSubTypes.Type(value = ActivationLReLU.class, name = "LReLU"),
                @JsonSubTypes.Type(value = ActivationRationalTanh.class, name = "RationalTanh"),
                @JsonSubTypes.Type(value = ActivationReLU.class, name = "ReLU"),
                @JsonSubTypes.Type(value = ActivationRReLU.class, name = "RReLU"),
                @JsonSubTypes.Type(value = ActivationSigmoid.class, name = "Sigmoid"),
                @JsonSubTypes.Type(value = ActivationSoftmax.class, name = "Softmax"),
                @JsonSubTypes.Type(value = ActivationSoftPlus.class, name = "SoftPlus"),
                @JsonSubTypes.Type(value = ActivationSoftSign.class, name = "SoftSign"),
                @JsonSubTypes.Type(value = ActivationTanH.class, name = "TanH")})
@JsonAutoDetect(fieldVisibility = JsonAutoDetect.Visibility.ANY, getterVisibility = JsonAutoDetect.Visibility.NONE,
                setterVisibility = JsonAutoDetect.Visibility.NONE)
public interface IActivation extends Serializable {

    /**
     * Carry out activation function on the input array (usually known as 'preOut' or 'z')
     * Implementations must overwrite "in", transform in place and return "in"
     * Can support separate behaviour during test
     * @param in
     * @param training
     * @return transformed activation
     */
    INDArray getActivation(INDArray in, boolean training);

    /**
     * Backpropagate the errors through the activation function, given input z and epsilon dL/da.<br>
     * Returns 2 INDArrays:<br>
     * (a) The gradient dL/dz, calculated from dL/da, and<br>
     * (b) The parameter gradients dL/dw, where w is the weights in the activation function. For activation functions
     *     with no gradients, this will be null.
     *
     * @param in      Input, before applying the activation function (z, or 'preOut')
     * @param epsilon Gradient to be backpropagated: dL/da, where L is the loss function
     * @return        dL/dz and dL/dw, for weights w (null if activatino function has no weights)
     */
    Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon);


    int numParams(int inputSize);

    void setParametersViewArray(INDArray viewArray, boolean initialize);

    INDArray getParametersViewArray();

    void setGradientViewArray(INDArray viewArray);

    INDArray getGradientViewArray();

}

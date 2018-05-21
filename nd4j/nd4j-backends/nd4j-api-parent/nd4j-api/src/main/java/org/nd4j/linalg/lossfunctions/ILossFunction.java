package org.nd4j.linalg.lossfunctions;


import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.serde.json.LegacyILossFunctionDeserializerHelper;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;

/**
 * Interface for loss functions
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class",
        defaultImpl = LegacyILossFunctionDeserializerHelper.class)
public interface ILossFunction extends Serializable {

    /**
     * Compute the score (loss function value) for the given inputs.
     *  @param labels       Label/expected preOutput
     * @param preOutput    Output of the model (neural network)
     * @param activationFn Activation function that should be applied to preOutput
     * @param mask         Mask array; may be null
     * @param average      Whether the score should be averaged (divided by number of rows in labels/preOutput) or not   @return Loss function value
     */
    double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average);

    /**
     * Compute the score (loss function value) for each example individually.
     * For input [numExamples,nOut] returns scores as a column vector: [numExamples,1]
     *  @param labels       Labels/expected output
     * @param preOutput    Output of the model (neural network)
     * @param activationFn Activation function that should be applied to preOutput
     * @param mask         @return Loss function value for each example; column vector
     */
    INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask);

    /**
     * Compute the gradient of the loss function with respect to the inputs: dL/dOutput
     *
     * @param labels       Label/expected output
     * @param preOutput    Output of the model (neural network), before the activation function is applied
     * @param activationFn Activation function that should be applied to preOutput
     * @param mask         Mask array; may be null
     * @return Gradient dL/dPreOut
     */
    INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask);

    /**
     * Compute both the score (loss function value) and gradient. This is equivalent to calling {@link #computeScore(INDArray, INDArray, IActivation, INDArray, boolean)}
     * and {@link #computeGradient(INDArray, INDArray, IActivation, INDArray)} individually
     *
     * @param labels       Label/expected output
     * @param preOutput    Output of the model (neural network)
     * @param activationFn Activation function that should be applied to preOutput
     * @param mask         Mask array; may be null
     * @param average      Whether the score should be averaged (divided by number of rows in labels/output) or not
     * @return The score (loss function value) and gradient
     */
    //TODO: do we want to use the apache commons pair here?
    Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn,
                    INDArray mask, boolean average);

    /**
     * The opName of this function
     * @return
     */
    String name();

}

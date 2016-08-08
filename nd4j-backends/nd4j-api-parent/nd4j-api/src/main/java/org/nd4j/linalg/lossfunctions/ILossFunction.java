package org.nd4j.linalg.lossfunctions;

import org.apache.commons.lang3.tuple.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Interface for loss functions
 */
public interface ILossFunction {

    /**
     * Compute the score (loss function value) for the given inputs.
     *
     * @param labels  Label/expected output
     * @param output  Output of the model (neural network)
     * @param mask    Mask array; may be null
     * @param average Whether the score should be averaged (divided by number of rows in labels/output) or not
     * @return Loss function value
     */
    double computeScore(INDArray labels, INDArray output, INDArray mask, boolean average);

    /**
     * Compute the gradient of the loss function with respect to the inputs: dL/dOutput
     *
     * @param labels Label/expected output
     * @param output Output of the model (neural network)
     * @param mask   Mask array; may be null
     * @return Gradient dL/dOutput
     */
    INDArray computeGradient(INDArray labels, INDArray output, INDArray mask);

    /**
     * Compute both the score (loss function value) and gradient. This is equivalent to calling {@link #computeScore(INDArray, INDArray, INDArray, boolean)}
     * and {@link #computeGradient(INDArray, INDArray, INDArray)} individually
     *
     * @param labels  Label/expected output
     * @param output  Output of the model (neural network)
     * @param mask    Mask array; may be null
     * @param average Whether the score should be averaged (divided by number of rows in labels/output) or not
     * @return The score (loss function value) and gradient
     */
    //TODO: do we want to use the apache commons pair here?
    Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray output, INDArray mask, boolean average);

}

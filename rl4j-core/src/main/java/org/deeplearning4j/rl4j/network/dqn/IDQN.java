package org.deeplearning4j.rl4j.network.dqn;

import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.OutputStream;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/12/16.
 *
 * This neural net quantify the value of each action given a state
 *
 */
public interface IDQN extends NeuralNet {

    void fit(INDArray input, INDArray labels);

    void fit(INDArray input, INDArray[] labels);

    INDArray output(INDArray batch);

    INDArray[] outputAll(INDArray batch);

    IDQN clone();

    Gradient[] gradient(INDArray input, INDArray label);

    Gradient[] gradient(INDArray input, INDArray[] label);

    void applyGradient(Gradient[] gradient, int batchSize);

    double getLatestScore();
}

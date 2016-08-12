package org.deeplearning4j.rl4j.network;

import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.OutputStream;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/5/16.
 *
 * Factorisation between ActorCritic and DQN.
 * Useful for AsyncLearning and Thread code.
 */
public interface NeuralNet {


    INDArray[] outputAll(INDArray batch);

    NeuralNet clone();

    Gradient gradient(INDArray input, INDArray[] labels);

    void fit(INDArray input, INDArray[] labels);

    void applyGradient(Gradient gradient);

    double getLatestScore();

    void save(OutputStream os);

    void save(String filename);

}

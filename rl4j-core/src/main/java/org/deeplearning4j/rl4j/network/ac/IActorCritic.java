package org.deeplearning4j.rl4j.network.ac;

import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.io.OutputStream;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/5/16.
 *
 * An actor critic has one of its input act as an actor and the
 * other one as a critic.
 * The first output quantify the advantage provided by getting to one state
 * while the other choose among a set of action which is the best one.
 */
public interface IActorCritic<NN extends IActorCritic> extends NeuralNet<NN> {

    boolean isRecurrent();

    void reset();

    void fit(INDArray input, INDArray[] labels);

    //FIRST SHOULD BE VALUE AND SECOND IS SOFTMAX POLICY. DONT MESS THIS UP OR ELSE ASYNC THREAD IS BROKEN (maxQ) !
    INDArray[] outputAll(INDArray batch);

    NN clone();

    void copy(NN from);

    Gradient[] gradient(INDArray input, INDArray[] labels);

    void applyGradient(Gradient[] gradient, int batchSize);

    void save(OutputStream streamValue, OutputStream streamPolicy) throws IOException;

    void save(String pathValue, String pathPolicy) throws IOException;

    double getLatestScore();

}

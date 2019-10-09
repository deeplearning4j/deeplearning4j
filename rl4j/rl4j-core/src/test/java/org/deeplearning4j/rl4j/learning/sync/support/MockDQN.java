package org.deeplearning4j.rl4j.learning.sync.support;

import lombok.Setter;
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.io.OutputStream;

public class MockDQN implements IDQN {

    private final double mult;

    public MockDQN() {
        this(1.0);
    }

    public MockDQN(double mult) {
        this.mult = mult;
    }

    @Override
    public NeuralNetwork[] getNeuralNetworks() {
        return new NeuralNetwork[0];
    }

    @Override
    public boolean isRecurrent() {
        return false;
    }

    @Override
    public void reset() {

    }

    @Override
    public void fit(INDArray input, INDArray labels) {

    }

    @Override
    public void fit(INDArray input, INDArray[] labels) {

    }

    @Override
    public INDArray output(INDArray batch) {
        if(mult != 1.0) {
            return batch.dup().muli(mult);
        }

        return batch;
    }

    @Override
    public INDArray[] outputAll(INDArray batch) {
        return new INDArray[0];
    }

    @Override
    public IDQN clone() {
        return null;
    }

    @Override
    public void copy(NeuralNet from) {

    }

    @Override
    public void copy(IDQN from) {

    }

    @Override
    public Gradient[] gradient(INDArray input, INDArray label) {
        return new Gradient[0];
    }

    @Override
    public Gradient[] gradient(INDArray input, INDArray[] label) {
        return new Gradient[0];
    }

    @Override
    public void applyGradient(Gradient[] gradient, int batchSize) {

    }

    @Override
    public double getLatestScore() {
        return 0;
    }

    @Override
    public void save(OutputStream os) throws IOException {

    }

    @Override
    public void save(String filename) throws IOException {

    }
}

package org.deeplearning4j.rl4j.support;

import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.io.OutputStream;

public class TestNeuralNet implements NeuralNet {
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
    public INDArray[] outputAll(INDArray batch) {
        return new INDArray[0];
    }

    @Override
    public NeuralNet clone() {
        return null;
    }

    @Override
    public void copy(NeuralNet from) {

    }

    @Override
    public Gradient[] gradient(INDArray input, INDArray[] labels) {
        return new Gradient[0];
    }

    @Override
    public void fit(INDArray input, INDArray[] labels) {

    }

    @Override
    public void applyGradient(Gradient[] gradients, int batchSize) {

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

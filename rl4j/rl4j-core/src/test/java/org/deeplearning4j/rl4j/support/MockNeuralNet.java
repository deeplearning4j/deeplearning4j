package org.deeplearning4j.rl4j.support;

import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.observation.Observation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

public class MockNeuralNet implements NeuralNet {

    public int resetCallCount = 0;
    public int copyCallCount = 0;
    public List<INDArray> outputAllInputs = new ArrayList<INDArray>();

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
        ++resetCallCount;
    }

    @Override
    public INDArray[] outputAll(INDArray batch) {
        outputAllInputs.add(batch);
        return new INDArray[] { Nd4j.create(new double[] { outputAllInputs.size() }) };
    }

    @Override
    public void fit(FeaturesLabels featuresLabels) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Gradients computeGradients(FeaturesLabels updateLabels) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void applyGradients(Gradients gradients) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void copy(ITrainableNeuralNet from) {
        ++copyCallCount;
    }

    @Override
    public NeuralNet clone() {
        return this;
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

    @Override
    public INDArray output(Observation observation) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray output(INDArray batch) {
        throw new UnsupportedOperationException();
    }
}
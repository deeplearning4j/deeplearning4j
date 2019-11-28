package org.deeplearning4j.rl4j.support;

import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.observation.Observation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

public class MockDQN implements IDQN {

    public boolean hasBeenReset = false;
    public final List<INDArray> outputParams = new ArrayList<>();
    public final List<Pair<INDArray, INDArray>> fitParams = new ArrayList<>();

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
        hasBeenReset = true;
    }

    @Override
    public void fit(INDArray input, INDArray labels) {
        fitParams.add(new Pair<>(input, labels));
    }

    @Override
    public void fit(INDArray input, INDArray[] labels) {

    }

    @Override
    public INDArray output(INDArray batch){
        outputParams.add(batch);
        return batch;
    }

    @Override
    public INDArray output(Observation observation) {
        return this.output(observation.getData());
    }

    @Override
    public INDArray[] outputAll(INDArray batch) {
        return new INDArray[0];
    }

    @Override
    public IDQN clone() {
        MockDQN clone = new MockDQN();
        clone.hasBeenReset = hasBeenReset;

        return clone;
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

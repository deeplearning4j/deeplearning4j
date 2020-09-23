package org.deeplearning4j.rl4j.support;

import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.network.CommonOutputNames;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.deeplearning4j.rl4j.network.NeuralNetOutput;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.observation.Observation;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

public class MockDQN implements IDQN {

    public boolean hasBeenReset = false;
    public final List<INDArray> outputParams = new ArrayList<>();
    public final List<Pair<INDArray, INDArray>> fitParams = new ArrayList<>();
    public final List<Pair<INDArray, INDArray>> gradientParams = new ArrayList<>();
    public final List<INDArray> outputAllParams = new ArrayList<>();

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
    public NeuralNetOutput output(INDArray batch){
        outputParams.add(batch);

        NeuralNetOutput result = new NeuralNetOutput();
        result.put(CommonOutputNames.QValues, batch);
        return result;
    }

    @Override
    public NeuralNetOutput output(Observation observation) {
        return this.output(observation.getData());
    }

    @Override
    public INDArray[] outputAll(INDArray batch) {
        outputAllParams.add(batch);
        return new INDArray[] { batch.mul(-1.0) };
    }

    @Override
    public void fit(FeaturesLabels featuresLabels) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Gradients computeGradients(FeaturesLabels featuresLabels) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void applyGradients(Gradients gradients) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void copyFrom(ITrainableNeuralNet from) {
        throw new UnsupportedOperationException();
    }

    @Override
    public IDQN clone() {
        MockDQN clone = new MockDQN();
        clone.hasBeenReset = hasBeenReset;

        return clone;
    }

    @Override
    public Gradient[] gradient(INDArray input, INDArray label) {
        gradientParams.add(new Pair<INDArray, INDArray>(input, label));
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
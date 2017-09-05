package org.deeplearning4j.rl4j.network.dqn;

import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.io.OutputStream;
import java.util.Collection;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/25/16.
 */
public class DQN<NN extends DQN> implements IDQN<NN> {

    final protected MultiLayerNetwork mln;

    int i = 0;

    public DQN(MultiLayerNetwork mln) {
        this.mln = mln;
    }

    public NeuralNetwork[] getNeuralNetworks() {
        return new NeuralNetwork[] { mln };
    }

    public static DQN load(String path) throws IOException {
        return new DQN(ModelSerializer.restoreMultiLayerNetwork(path));
    }

    public boolean isRecurrent() {
        return false;
    }

    public void reset() {
        // no recurrent layer
    }

    public void fit(INDArray input, INDArray labels) {
        mln.fit(input, labels);
    }

    public void fit(INDArray input, INDArray[] labels) {
        fit(input, labels[0]);
    }

    public INDArray output(INDArray batch) {
        return mln.output(batch);
    }

    public INDArray[] outputAll(INDArray batch) {
        return new INDArray[] {output(batch)};
    }

    public NN clone() {
        NN nn = (NN)new DQN(mln.clone());
        nn.mln.setListeners(mln.getListeners());
        return nn;
    }

    public void copy(NN from) {
        mln.setParams(from.mln.params());
    }

    public Gradient[] gradient(INDArray input, INDArray labels) {
        mln.setInput(input);
        mln.setLabels(labels);
        mln.computeGradientAndScore();
        Collection<IterationListener> iterationListeners = mln.getListeners();
        if (iterationListeners != null && iterationListeners.size() > 0) {
            for (IterationListener l : iterationListeners) {
                if (l instanceof TrainingListener) {
                    ((TrainingListener) l).onGradientCalculation(mln);
                }
            }
        }
        //System.out.println("SCORE: " + mln.score());
        return new Gradient[] {mln.gradient()};
    }

    public Gradient[] gradient(INDArray input, INDArray[] labels) {
        return gradient(input, labels[0]);
    }

    public void applyGradient(Gradient[] gradient, int batchSize) {
        MultiLayerConfiguration mlnConf = mln.getLayerWiseConfigurations();
        int iterationCount = mlnConf.getIterationCount();
        mln.getUpdater().update(mln, gradient[0], iterationCount, batchSize);
        mln.params().subi(gradient[0].gradient());
        Collection<IterationListener> iterationListeners = mln.getListeners();
        if (iterationListeners != null && iterationListeners.size() > 0) {
            for (IterationListener listener : iterationListeners) {
                listener.iterationDone(mln, iterationCount);
            }
        }
        mlnConf.setIterationCount(iterationCount + 1);
    }

    public double getLatestScore() {
        return mln.score();
    }

    public void save(OutputStream stream) throws IOException {
        ModelSerializer.writeModel(mln, stream, true);
    }

    public void save(String path) throws IOException {
        ModelSerializer.writeModel(mln, path, true);
    }
}

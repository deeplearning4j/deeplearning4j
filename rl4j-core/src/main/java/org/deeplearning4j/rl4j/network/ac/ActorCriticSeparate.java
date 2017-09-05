package org.deeplearning4j.rl4j.network.ac;

import lombok.Getter;
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.recurrent.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.io.OutputStream;
import java.util.Collection;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/23/16.
 */
public class ActorCriticSeparate<NN extends ActorCriticSeparate> implements IActorCritic<NN> {

    final protected MultiLayerNetwork valueNet;
    final protected MultiLayerNetwork policyNet;
    @Getter
    final protected boolean recurrent;

    public ActorCriticSeparate(MultiLayerNetwork valueNet, MultiLayerNetwork policyNet) {
        this.valueNet = valueNet;
        this.policyNet = policyNet;
        this.recurrent = valueNet.getOutputLayer() instanceof RnnOutputLayer;
    }

    public NeuralNetwork[] getNeuralNetworks() {
        return new NeuralNetwork[] { valueNet, policyNet };
    }

    public static ActorCriticSeparate load(String pathValue, String pathPolicy) throws IOException {
        return new ActorCriticSeparate(ModelSerializer.restoreMultiLayerNetwork(pathValue),
                                       ModelSerializer.restoreMultiLayerNetwork(pathPolicy));
    }

    public void reset() {
        if (recurrent) {
            valueNet.rnnClearPreviousState();
            policyNet.rnnClearPreviousState();
        }
    }

    public void fit(INDArray input, INDArray[] labels) {

        valueNet.fit(input, labels[0]);
        policyNet.fit(input, labels[1]);

    }


    public INDArray[] outputAll(INDArray batch) {
        if (recurrent) {
            return new INDArray[] {valueNet.rnnTimeStep(batch), policyNet.rnnTimeStep(batch)};
        } else {
            return new INDArray[] {valueNet.output(batch), policyNet.output(batch)};
        }
    }

    public NN clone() {
        NN nn = (NN)new ActorCriticSeparate(valueNet.clone(), policyNet.clone());
        nn.valueNet.setListeners(valueNet.getListeners());
        nn.policyNet.setListeners(policyNet.getListeners());
        return nn;
    }

    public void copy(NN from) {
        valueNet.setParams(from.valueNet.params());
        policyNet.setParams(from.policyNet.params());
    }

    public Gradient[] gradient(INDArray input, INDArray[] labels) {
        valueNet.setInput(input);
        valueNet.setLabels(labels[0]);
        valueNet.computeGradientAndScore();
        Collection<IterationListener> valueIterationListeners = valueNet.getListeners();
        if (valueIterationListeners != null && valueIterationListeners.size() > 0) {
            for (IterationListener l : valueIterationListeners) {
                if (l instanceof TrainingListener) {
                    ((TrainingListener) l).onGradientCalculation(valueNet);
                }
            }
        }

        policyNet.setInput(input);
        policyNet.setLabels(labels[1]);
        policyNet.computeGradientAndScore();
        Collection<IterationListener> policyIterationListeners = policyNet.getListeners();
        if (policyIterationListeners != null && policyIterationListeners.size() > 0) {
            for (IterationListener l : policyIterationListeners) {
                if (l instanceof TrainingListener) {
                    ((TrainingListener) l).onGradientCalculation(policyNet);
                }
            }
        }
        return new Gradient[] {valueNet.gradient(), policyNet.gradient()};
    }


    public void applyGradient(Gradient[] gradient, int batchSize) {
        MultiLayerConfiguration valueConf = valueNet.getLayerWiseConfigurations();
        int valueIterationCount = valueConf.getIterationCount();
        valueNet.getUpdater().update(valueNet, gradient[0], valueIterationCount, batchSize);
        valueNet.params().subi(gradient[0].gradient());
        Collection<IterationListener> valueIterationListeners = valueNet.getListeners();
        if (valueIterationListeners != null && valueIterationListeners.size() > 0) {
            for (IterationListener listener : valueIterationListeners) {
                listener.iterationDone(valueNet, valueIterationCount);
            }
        }
        valueConf.setIterationCount(valueIterationCount + 1);

        MultiLayerConfiguration policyConf = policyNet.getLayerWiseConfigurations();
        int policyIterationCount = policyConf.getIterationCount();
        policyNet.getUpdater().update(policyNet, gradient[1], policyIterationCount, batchSize);
        policyNet.params().subi(gradient[1].gradient());
        Collection<IterationListener> policyIterationListeners = policyNet.getListeners();
        if (policyIterationListeners != null && policyIterationListeners.size() > 0) {
            for (IterationListener listener : policyIterationListeners) {
                listener.iterationDone(policyNet, policyIterationCount);
            }
        }
        policyConf.setIterationCount(policyIterationCount + 1);
    }

    public double getLatestScore() {
        return valueNet.score();
    }

    public void save(OutputStream stream) throws IOException {
        throw new UnsupportedOperationException("Call save(streamValue, streamPolicy)");
    }

    public void save(String path) throws IOException {
        throw new UnsupportedOperationException("Call save(pathValue, pathPolicy)");
    }

    public void save(OutputStream streamValue, OutputStream streamPolicy) throws IOException {
        ModelSerializer.writeModel(valueNet, streamValue, true);
        ModelSerializer.writeModel(policyNet, streamPolicy, true);
    }

    public void save(String pathValue, String pathPolicy) throws IOException {
        ModelSerializer.writeModel(valueNet, pathValue, true);
        ModelSerializer.writeModel(policyNet, pathPolicy, true);
    }
}



package org.deeplearning4j.lstm;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.recurrent.CudnnLSTMHelper;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.lang.reflect.Field;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * Created by Alex on 18/07/2017.
 */
public class ValidateCudnnLSTM {

    @Test
    public void validateImplSimple() throws Exception {

        Nd4j.getRandom().setSeed(12345);
        int minibatch = 10;
        int inputSize = 3;
        int lstmLayerSize = 4;
        int timeSeriesLength = 3;
        int nOut = 2;
        INDArray input = Nd4j.rand(new int[] {minibatch, inputSize, timeSeriesLength});
        INDArray labels = Nd4j.zeros(minibatch, nOut, timeSeriesLength);
        Random r = new Random(12345);
        for (int i = 0; i < minibatch; i++) {
            for (int j = 0; j < timeSeriesLength; j++) {
                labels.putScalar(i, r.nextInt(nOut), j, 1.0);
            }
        }

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .inferenceWorkspaceMode(WorkspaceMode.NONE).trainingWorkspaceMode(WorkspaceMode.NONE)
                .learningRate(1.0)
                .regularization(false).updater(Updater.NONE).seed(12345L).weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0, 2)).list()
                .layer(0, new LSTM.Builder().nIn(input.size(1)).nOut(lstmLayerSize)
                        .gateActivationFunction(Activation.SIGMOID).activation(Activation.TANH).build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(lstmLayerSize).nOut(nOut).build())
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork mln1 = new MultiLayerNetwork(conf.clone());
        mln1.init();

        MultiLayerNetwork mln2 = new MultiLayerNetwork(conf.clone());
        mln2.init();


        assertEquals(mln1.params(), mln2.params());

        Field f = org.deeplearning4j.nn.layers.recurrent.LSTM.class.getDeclaredField("helper");
        f.setAccessible(true);

        Layer l0 = mln1.getLayer(0);
        f.set(l0, null);
        assertNull(f.get(l0));

        l0 = mln2.getLayer(0);
        assertTrue(f.get(l0) instanceof CudnnLSTMHelper);


        INDArray out1 = mln1.output(input);
        INDArray out2 = mln2.output(input);

        assertEquals(out1, out2);


        mln1.setInput(input);
        mln1.setLabels(labels);

        mln2.setInput(input);
        mln2.setLabels(labels);

        mln1.computeGradientAndScore();
        mln2.computeGradientAndScore();

        Gradient g1 = mln1.gradient();
        Gradient g2 = mln2.gradient();

        for(Map.Entry<String,INDArray> entry : g1.gradientForVariable().entrySet()){
            INDArray exp = entry.getValue();
            INDArray act = g2.gradientForVariable().get(entry.getKey());

            System.out.println(entry.getKey() + "\t" + exp.equals(act));
        }

        assertEquals(mln1.getFlattenedGradients(), mln2.getFlattenedGradients());
    }

    @Test
    public void validateImplMultiLayer() throws Exception {

        Nd4j.getRandom().setSeed(12345);
        int minibatch = 10;
        int inputSize = 3;
        int lstmLayerSize = 4;
        int timeSeriesLength = 3;
        int nOut = 2;
        INDArray input = Nd4j.rand(new int[] {minibatch, inputSize, timeSeriesLength});
        INDArray labels = Nd4j.zeros(minibatch, nOut, timeSeriesLength);
        Random r = new Random(12345);
        for (int i = 0; i < minibatch; i++) {
            for (int j = 0; j < timeSeriesLength; j++) {
                labels.putScalar(i, r.nextInt(nOut), j, 1.0);
            }
        }

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().learningRate(1.0)
                .inferenceWorkspaceMode(WorkspaceMode.NONE).trainingWorkspaceMode(WorkspaceMode.NONE)
                .regularization(false).updater(Updater.NONE).seed(12345L).weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0, 2)).list()
                .layer(0, new LSTM.Builder().nIn(input.size(1)).nOut(lstmLayerSize)
                        .gateActivationFunction(Activation.SIGMOID).activation(Activation.TANH).build())
                .layer(1, new LSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
                        .gateActivationFunction(Activation.SIGMOID).activation(Activation.TANH).build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(lstmLayerSize).nOut(nOut).build())
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork mln1 = new MultiLayerNetwork(conf.clone());
        mln1.init();

        MultiLayerNetwork mln2 = new MultiLayerNetwork(conf.clone());
        mln2.init();


        assertEquals(mln1.params(), mln2.params());

        Field f = org.deeplearning4j.nn.layers.recurrent.LSTM.class.getDeclaredField("helper");
        f.setAccessible(true);

        Layer l0 = mln1.getLayer(0);
        Layer l1 = mln1.getLayer(1);
        f.set(l0, null);
        f.set(l1, null);
        assertNull(f.get(l0));
        assertNull(f.get(l1));

        l0 = mln2.getLayer(0);
        l1 = mln2.getLayer(1);
        assertTrue(f.get(l0) instanceof CudnnLSTMHelper);
        assertTrue(f.get(l1) instanceof CudnnLSTMHelper);


        INDArray out1 = mln1.output(input);
        INDArray out2 = mln2.output(input);

        assertEquals(out1, out2);


        mln1.setInput(input);
        mln1.setLabels(labels);

        mln2.setInput(input);
        mln2.setLabels(labels);

        mln1.computeGradientAndScore();
        mln2.computeGradientAndScore();

        assertEquals(mln1.getFlattenedGradients(), mln2.getFlattenedGradients());
    }

}

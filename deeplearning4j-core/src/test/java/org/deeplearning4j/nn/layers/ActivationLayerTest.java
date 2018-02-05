package org.deeplearning4j.nn.layers;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.AutoEncoder;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationELU;
import org.nd4j.linalg.activations.impl.ActivationRationalTanh;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.List;

import static org.junit.Assert.*;

/**
 */

public class ActivationLayerTest extends BaseDL4JTest {

    @Test
    public void testInputTypes() {
        org.deeplearning4j.nn.conf.layers.ActivationLayer l =
                        new org.deeplearning4j.nn.conf.layers.ActivationLayer.Builder().activation(Activation.RELU)
                                        .build();


        InputType in1 = InputType.feedForward(20);
        InputType in2 = InputType.convolutional(28, 28, 1);

        assertEquals(in1, l.getOutputType(0, in1));
        assertEquals(in2, l.getOutputType(0, in2));
        assertNull(l.getPreProcessorForInputType(in1));
        assertNull(l.getPreProcessorForInputType(in2));
    }

    @Test
    public void testDenseActivationLayer() throws Exception {
        DataSetIterator iter = new MnistDataSetIterator(2, 2);
        DataSet next = iter.next();

        // Run without separate activation layer
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).seed(123)
                        .list()
                        .layer(0, new DenseLayer.Builder().nIn(28 * 28 * 1).nOut(10).activation(Activation.RELU)
                                        .weightInit(WeightInit.XAVIER).build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).weightInit(WeightInit.XAVIER)
                                                        .activation(Activation.SOFTMAX).nIn(10).nOut(10).build())
                        .backprop(true).pretrain(false).build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        network.fit(next);


        // Run with separate activation layer
        MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).seed(123)
                        .list()
                        .layer(0, new DenseLayer.Builder().nIn(28 * 28 * 1).nOut(10).activation(Activation.IDENTITY)
                                        .weightInit(WeightInit.XAVIER).build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.ActivationLayer.Builder()
                                        .activation(Activation.RELU).build())
                        .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                        .weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX).nIn(10).nOut(10)
                                        .build())
                        .backprop(true).pretrain(false).build();

        MultiLayerNetwork network2 = new MultiLayerNetwork(conf2);
        network2.init();
        network2.fit(next);

        // check parameters
        assertEquals(network.getLayer(0).getParam("W"), network2.getLayer(0).getParam("W"));
        assertEquals(network.getLayer(1).getParam("W"), network2.getLayer(2).getParam("W"));
        assertEquals(network.getLayer(0).getParam("b"), network2.getLayer(0).getParam("b"));
        assertEquals(network.getLayer(1).getParam("b"), network2.getLayer(2).getParam("b"));

        // check activations
        network.init();
        network.setInput(next.getFeatureMatrix());
        List<INDArray> activations = network.feedForward(true);

        network2.init();
        network2.setInput(next.getFeatureMatrix());
        List<INDArray> activations2 = network2.feedForward(true);

        assertEquals(activations.get(1).reshape(activations2.get(2).shape()), activations2.get(2));
        assertEquals(activations.get(2), activations2.get(3));


    }

    @Test
    public void testAutoEncoderActivationLayer() throws Exception {

        int minibatch = 3;
        int nIn = 5;
        int layerSize = 5;
        int nOut = 3;

        INDArray next = Nd4j.rand(new int[] {minibatch, nIn});
        INDArray labels = Nd4j.zeros(minibatch, nOut);
        for (int i = 0; i < minibatch; i++) {
            labels.putScalar(i, i % nOut, 1.0);
        }

        // Run without separate activation layer
        Nd4j.getRandom().setSeed(12345);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).seed(123)
                        .list()
                        .layer(0, new AutoEncoder.Builder().nIn(nIn).nOut(layerSize).corruptionLevel(0.0)
                                        .activation(Activation.SIGMOID).build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                                                        .activation(Activation.SOFTMAX).nIn(layerSize).nOut(nOut)
                                                        .build())
                        .backprop(true).pretrain(false).build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        network.fit(next, labels); //Labels are necessary for this test: layer activation function affect pretraining results, otherwise


        // Run with separate activation layer
        Nd4j.getRandom().setSeed(12345);
        MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).seed(123)
                        .list()
                        .layer(0, new AutoEncoder.Builder().nIn(nIn).nOut(layerSize).corruptionLevel(0.0)
                                        .activation(Activation.IDENTITY).build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.ActivationLayer.Builder()
                                        .activation(Activation.SIGMOID).build())
                        .layer(2, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                                                        .activation(Activation.SOFTMAX).nIn(layerSize).nOut(nOut)
                                                        .build())
                        .backprop(true).pretrain(false).build();

        MultiLayerNetwork network2 = new MultiLayerNetwork(conf2);
        network2.init();
        network2.fit(next, labels);

        // check parameters
        assertEquals(network.getLayer(0).getParam("W"), network2.getLayer(0).getParam("W"));
        assertEquals(network.getLayer(1).getParam("W"), network2.getLayer(2).getParam("W"));
        assertEquals(network.getLayer(0).getParam("b"), network2.getLayer(0).getParam("b"));
        assertEquals(network.getLayer(1).getParam("b"), network2.getLayer(2).getParam("b"));

        // check activations
        network.init();
        network.setInput(next);
        List<INDArray> activations = network.feedForward(true);

        network2.init();
        network2.setInput(next);
        List<INDArray> activations2 = network2.feedForward(true);

        assertEquals(activations.get(1).reshape(activations2.get(2).shape()), activations2.get(2));
        assertEquals(activations.get(2), activations2.get(3));


    }

    @Test
    public void testCNNActivationLayer() throws Exception {
        DataSetIterator iter = new MnistDataSetIterator(2, 2);
        DataSet next = iter.next();

        // Run without separate activation layer
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).seed(123)
                        .list()
                        .layer(0, new ConvolutionLayer.Builder(4, 4).stride(2, 2).nIn(1).nOut(20)
                                        .activation(Activation.RELU).weightInit(WeightInit.XAVIER).build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).weightInit(WeightInit.XAVIER)
                                                        .activation(Activation.SOFTMAX).nOut(10).build())
                        .backprop(true).pretrain(false).setInputType(InputType.convolutionalFlat(28, 28, 1)).build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        network.fit(next);


        // Run with separate activation layer
        MultiLayerConfiguration conf2 =
                        new NeuralNetConfiguration.Builder()
                                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                        .seed(123).list()
                                        .layer(0, new ConvolutionLayer.Builder(4, 4).stride(2, 2).nIn(1).nOut(20)
                                                        .activation(Activation.IDENTITY).weightInit(WeightInit.XAVIER)
                                                        .build())
                                        .layer(1, new org.deeplearning4j.nn.conf.layers.ActivationLayer.Builder()
                                                        .activation(Activation.RELU).build())
                                        .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                                        .weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX)
                                                        .nOut(10).build())
                                        .backprop(true).pretrain(false)
                                        .setInputType(InputType.convolutionalFlat(28, 28, 1)).build();

        MultiLayerNetwork network2 = new MultiLayerNetwork(conf2);
        network2.init();
        network2.fit(next);

        // check parameters
        assertEquals(network.getLayer(0).getParam("W"), network2.getLayer(0).getParam("W"));
        assertEquals(network.getLayer(1).getParam("W"), network2.getLayer(2).getParam("W"));
        assertEquals(network.getLayer(0).getParam("b"), network2.getLayer(0).getParam("b"));

        // check activations
        network.init();
        network.setInput(next.getFeatureMatrix());
        List<INDArray> activations = network.feedForward(true);

        network2.init();
        network2.setInput(next.getFeatureMatrix());
        List<INDArray> activations2 = network2.feedForward(true);

        assertEquals(activations.get(1).reshape(activations2.get(2).shape()), activations2.get(2));
        assertEquals(activations.get(2), activations2.get(3));
    }


    @Test
    public void testActivationInheritance() {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).seed(123)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RATIONALTANH)
                .list()
                .layer(new DenseLayer.Builder().nIn(10).nOut(10).build())
                .layer(new ActivationLayer())
                .layer(new ActivationLayer.Builder().build())
                .layer(new ActivationLayer.Builder().activation(Activation.ELU).build())
                .layer(new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(10).nOut(10).build())
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

        assertNotNull(((ActivationLayer)network.getLayer(1).conf().getLayer()).getActivationFn());

        assertTrue(((DenseLayer)network.getLayer(0).conf().getLayer()).getActivationFn() instanceof ActivationRationalTanh);
        assertTrue(((ActivationLayer)network.getLayer(1).conf().getLayer()).getActivationFn() instanceof ActivationRationalTanh);
        assertTrue(((ActivationLayer)network.getLayer(2).conf().getLayer()).getActivationFn() instanceof ActivationRationalTanh);
        assertTrue(((ActivationLayer)network.getLayer(3).conf().getLayer()).getActivationFn() instanceof ActivationELU);
        assertTrue(((OutputLayer)network.getLayer(4).conf().getLayer()).getActivationFn() instanceof ActivationSoftmax);
    }

    @Test
    public void testActivationInheritanceCG() {

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).seed(123)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RATIONALTANH)
                .graphBuilder()
                .addInputs("in")
                .addLayer("0", new DenseLayer.Builder().nIn(10).nOut(10).build(), "in")
                .addLayer("1", new ActivationLayer(), "0")
                .addLayer("2", new ActivationLayer.Builder().build(), "1")
                .addLayer("3", new ActivationLayer.Builder().activation(Activation.ELU).build(), "2")
                .addLayer("4", new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(10).nOut(10).build(), "3")
                .setOutputs("4")
                .build();

        ComputationGraph network = new ComputationGraph(conf);
        network.init();

        assertNotNull(((ActivationLayer)network.getLayer("1").conf().getLayer()).getActivationFn());

        assertTrue(((DenseLayer)network.getLayer("0").conf().getLayer()).getActivationFn() instanceof ActivationRationalTanh);
        assertTrue(((ActivationLayer)network.getLayer("1").conf().getLayer()).getActivationFn() instanceof ActivationRationalTanh);
        assertTrue(((ActivationLayer)network.getLayer("2").conf().getLayer()).getActivationFn() instanceof ActivationRationalTanh);
        assertTrue(((ActivationLayer)network.getLayer("3").conf().getLayer()).getActivationFn() instanceof ActivationELU);
        assertTrue(((OutputLayer)network.getLayer("4").conf().getLayer()).getActivationFn() instanceof ActivationSoftmax);
    }

}

package org.deeplearning4j.nn.layers;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;

/**
 */

public class ActivationLayerTest {

    @Test
    public void testDenseActivationLayer() throws Exception {
        DataSetIterator iter = new MnistDataSetIterator(2, 2);
        DataSet next = iter.next();

        // Run without separate activation layer
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .seed(123)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(28*28*1).nOut(10).activation("relu").weightInit(WeightInit.XAVIER).build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .nIn(10).nOut(10).build())
                .backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        network.fit(next);


        // Run with separate activation layer
        MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .seed(123)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(28*28*1).nOut(10).activation("identity").weightInit(WeightInit.XAVIER).build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.ActivationLayer.Builder().activation("relu").build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .nIn(10).nOut(10).build())
                .backprop(true).pretrain(false)
                .build();

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
        INDArray next = Nd4j.rand(new int[]{1,200});

        // Run without separate activation layer
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .iterations(1)
                .seed(123)
                .list()
                .layer(0, new AutoEncoder.Builder().nIn(200).nOut(20).activation("sigmoid").build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).activation("softmax").nIn(20).nOut(10).build())
                .backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        network.fit(next);


        // Run with separate activation layer
        MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .seed(123)
                .list()
                .layer(0, new AutoEncoder.Builder().nIn(200).nOut(20).activation("identity").build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.ActivationLayer.Builder().activation("sigmoid").build())
                .layer(2, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).activation("softmax").nIn(20).nOut(10).build())
                .backprop(true).pretrain(false)
                .build();

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
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .seed(123)
                .list()
                .layer(0, new ConvolutionLayer.Builder(4, 4).stride(2, 2).nIn(1).nOut(20).activation("relu").weightInit(WeightInit.XAVIER).build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .nOut(10).build())
                .backprop(true).pretrain(false)
                .cnnInputSize(28, 28, 1)
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        network.fit(next);


        // Run with separate activation layer
        MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .seed(123)
                .list()
                .layer(0, new ConvolutionLayer.Builder(4, 4).stride(2, 2).nIn(1).nOut(20).activation("identity").weightInit(WeightInit.XAVIER).build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.ActivationLayer.Builder().activation("relu").build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .nOut(10).build())
                .backprop(true).pretrain(false)
                .cnnInputSize(28, 28, 1)
                .build();

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

    // Standard identity does not work for LSTM setup. Need further dev to apply
//    @Test
//    public void testLSTMActivationLayer() throws Exception {
//        INDArray next = Nd4j.rand(new int[]{2,2,4});
//
//        // Run without separate activation layer
//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                .iterations(1)
//                .seed(123)
//                .list()
//                .layer(0, new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().activation("tanh").nIn(2).nOut(2).build())
//                .layer(1, new org.deeplearning4j.nn.conf.layers.RnnOutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(2).nOut(1).activation("tanh").build())
//                .backprop(true).pretrain(false)
//                .build();
//
//        MultiLayerNetwork network = new MultiLayerNetwork(conf);
//        network.init();
//        network.fit(next);
//
//
//        // Run with separate activation layer
//        MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                .iterations(1)
//                .seed(123)
//                .list()
//                .layer(0, new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder().activation("identity").nIn(2).nOut(2).build())
//                .layer(1, new org.deeplearning4j.nn.conf.layers.ActivationLayer.Builder().activation("tanh").build())
//                .layer(2, new org.deeplearning4j.nn.conf.layers.RnnOutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(2).nOut(1).activation("tanh").build())
//                .inputPreProcessor(1, new RnnToFeedForwardPreProcessor())
//                .inputPreProcessor(2, new FeedForwardToRnnPreProcessor())
//                .backprop(true).pretrain(false)
//                .build();
//
//        MultiLayerNetwork network2 = new MultiLayerNetwork(conf2);
//        network2.init();
//        network2.fit(next);
//
//        // check parameters
//        assertEquals(network.getLayer(0).getParam("W"), network2.getLayer(0).getParam("W"));
//        assertEquals(network.getLayer(1).getParam("W"), network2.getLayer(2).getParam("W"));
//        assertEquals(network.getLayer(0).getParam("b"), network2.getLayer(0).getParam("b"));
//
//        // check activations
//        network.init();
//        network.setInput(next);
//        List<INDArray> activations = network.feedForward(true);
//
//        network2.init();
//        network2.setInput(next);
//        List<INDArray> activations2 = network2.feedForward(true);
//
//        assertEquals(activations.get(1).permute(0,2,1).reshape(next.shape()[0]*next.shape()[2],next.shape()[1]), activations2.get(2));
//        assertEquals(activations.get(2), activations2.get(3));
//
//    }
//
//
//    @Test
//    public void testBiDiLSTMActivationLayer() throws Exception {
//        INDArray next = Nd4j.rand(new int[]{2,2,4});
//
//        // Run without separate activation layer
//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                .iterations(1)
//                .seed(123)
//                .list()
//                .layer(0, new org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM.Builder().activation("tanh").nIn(2).nOut(2).build())
//                .layer(1, new org.deeplearning4j.nn.conf.layers.RnnOutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(2).nOut(1).activation("tanh").build())
//                .backprop(true).pretrain(false)
//                .build();
//
//        MultiLayerNetwork network = new MultiLayerNetwork(conf);
//        network.init();
//        network.fit(next);
//
//
//        // Run with separate activation layer
//        MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                .iterations(1)
//                .seed(123)
//                .list()
//                .layer(0, new org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM.Builder().activation("identity").nIn(2).nOut(2).build())
//                .layer(1, new org.deeplearning4j.nn.conf.layers.ActivationLayer.Builder().activation("tanh").build())
//                .layer(2, new org.deeplearning4j.nn.conf.layers.RnnOutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(2).nOut(1).activation("tanh").build())
//                .inputPreProcessor(1, new RnnToFeedForwardPreProcessor())
//                .inputPreProcessor(2, new FeedForwardToRnnPreProcessor())
//                .backprop(true).pretrain(false)
//                .build();
//
//        MultiLayerNetwork network2 = new MultiLayerNetwork(conf2);
//        network2.init();
//        network2.fit(next);
//
//        // check parameters
//        assertEquals(network.getLayer(0).getParam("W"), network2.getLayer(0).getParam("W"));
//        assertEquals(network.getLayer(1).getParam("W"), network2.getLayer(2).getParam("W"));
//        assertEquals(network.getLayer(0).getParam("b"), network2.getLayer(0).getParam("b"));
//
//        // check activations
//        network.init();
//        network.setInput(next);
//        List<INDArray> activations = network.feedForward(true);
//
//        network2.init();
//        network2.setInput(next);
//        List<INDArray> activations2 = network2.feedForward(true);
//
//        assertEquals(activations.get(1).permute(0,2,1).reshape(next.shape()[0]*next.shape()[2],next.shape()[1]), activations2.get(2));
//        assertEquals(activations.get(2), activations2.get(3));
//    }

}

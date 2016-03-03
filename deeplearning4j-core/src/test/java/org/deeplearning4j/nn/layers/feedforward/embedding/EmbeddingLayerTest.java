package org.deeplearning4j.nn.layers.feedforward.embedding;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.*;

public class EmbeddingLayerTest {

    @Test
    public void testEmbeddingLayerConfig() {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .activation("tanh")
                .list()
                .layer(0, new EmbeddingLayer.Builder().nIn(10).nOut(5).build())
                .layer(1, new OutputLayer.Builder().nIn(5).nOut(4).build())
                .pretrain(false).backprop(true)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        Layer l0 = net.getLayer(0);

        assertEquals(org.deeplearning4j.nn.layers.feedforward.embedding.EmbeddingLayer.class, l0.getClass());
        assertEquals(10, ((FeedForwardLayer) l0.conf().getLayer()).getNIn());
        assertEquals(5, ((FeedForwardLayer) l0.conf().getLayer()).getNOut());

        INDArray weights = l0.getParam(DefaultParamInitializer.WEIGHT_KEY);
        INDArray bias = l0.getParam(DefaultParamInitializer.BIAS_KEY);
        assertArrayEquals(new int[]{10, 5}, weights.shape());
        assertArrayEquals(new int[]{1, 5}, bias.shape());
    }

    @Test
    public void testEmbeddingForwardPass() {
        //With the same parameters, embedding layer should have same activations as the equivalent one-hot representation
        // input with a DenseLayer

        int nClassesIn = 10;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .activation("tanh")
                .list()
                .layer(0, new EmbeddingLayer.Builder().nIn(nClassesIn).nOut(5).build())
                .layer(1, new OutputLayer.Builder().nIn(5).nOut(4).build())
                .pretrain(false).backprop(true)
                .build();
        MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                .activation("tanh")
                .list()
                .layer(0, new DenseLayer.Builder().nIn(nClassesIn).nOut(5).build())
                .layer(1, new OutputLayer.Builder().nIn(5).nOut(4).build())
                .pretrain(false).backprop(true)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        MultiLayerNetwork net2 = new MultiLayerNetwork(conf2);
        net.init();
        net2.init();

        net2.setParams(net.params().dup());

        int batchSize = 3;
        INDArray inEmbedding = Nd4j.create(batchSize, 1);
        INDArray inOneHot = Nd4j.create(batchSize, nClassesIn);

        Random r = new Random(12345);
        for (int i = 0; i < batchSize; i++) {
            int classIdx = r.nextInt(nClassesIn);
            inEmbedding.putScalar(i, classIdx);
            inOneHot.putScalar(new int[]{i, classIdx}, 1.0);
        }

        List<INDArray> activationsEmbedding = net.feedForward(inEmbedding, false);
        List<INDArray> activationsDense = net2.feedForward(inOneHot, false);
        for (int i = 1; i < 3; i++) {
            INDArray actE = activationsEmbedding.get(i);
            INDArray actD = activationsDense.get(i);
            assertEquals(actE, actD);
        }
    }

    @Test
    public void testEmbeddingBackwardPass() {
        //With the same parameters, embedding layer should have same activations as the equivalent one-hot representation
        // input with a DenseLayer

        int nClassesIn = 10;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .activation("tanh")
                .list()
                .layer(0, new EmbeddingLayer.Builder().nIn(nClassesIn).nOut(5).build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(5).nOut(4).activation("softmax").build())
                .pretrain(false).backprop(true)
                .build();
        MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                .activation("tanh")
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(nClassesIn).nOut(5).build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(5).nOut(4).activation("softmax").build())
                .pretrain(false).backprop(true)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        MultiLayerNetwork net2 = new MultiLayerNetwork(conf2);
        net.init();
        net2.init();

        net2.setParams(net.params().dup());

        int batchSize = 3;
        INDArray inEmbedding = Nd4j.create(batchSize, 1);
        INDArray inOneHot = Nd4j.create(batchSize, nClassesIn);
        INDArray outLabels = Nd4j.create(batchSize, 4);

        Random r = new Random(12345);
        for (int i = 0; i < batchSize; i++) {
            int classIdx = r.nextInt(nClassesIn);
            inEmbedding.putScalar(i, classIdx);
            inOneHot.putScalar(new int[]{i, classIdx}, 1.0);

            int labelIdx = r.nextInt(4);
            outLabels.putScalar(new int[]{i, labelIdx}, 1.0);
        }

        net.setInput(inEmbedding);
        net2.setInput(inOneHot);
        net.setLabels(outLabels);
        net2.setLabels(outLabels);

        net.computeGradientAndScore();
        net2.computeGradientAndScore();

        System.out.println(net.score() + "\t" + net2.score());
        assertEquals(net2.score(), net.score(), 1e-6);

        Map<String, INDArray> gradient = net.gradient().gradientForVariable();
        Map<String, INDArray> gradient2 = net2.gradient().gradientForVariable();
        assertEquals(gradient.size(), gradient2.size());

        for (String s : gradient.keySet()) {
            assertEquals(gradient2.get(s), gradient.get(s));
        }
    }

    @Test
    public void testEmbeddingLayerRNN() {

        int nClassesIn = 10;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .activation("tanh")
                .list()
                .layer(0, new EmbeddingLayer.Builder().nIn(nClassesIn).nOut(5).build())
                .layer(1, new GravesLSTM.Builder().nIn(5).nOut(7).activation("softsign").build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(7).nOut(4).activation("softmax").build())
                .inputPreProcessor(0, new RnnToFeedForwardPreProcessor())
                .inputPreProcessor(1, new FeedForwardToRnnPreProcessor())
                .pretrain(false).backprop(true)
                .build();
        MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                .activation("tanh")
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(nClassesIn).nOut(5).build())
                .layer(1, new GravesLSTM.Builder().nIn(5).nOut(7).activation("softsign").build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(7).nOut(4).activation("softmax").build())
                .inputPreProcessor(0, new RnnToFeedForwardPreProcessor())
                .inputPreProcessor(1, new FeedForwardToRnnPreProcessor())
                .pretrain(false).backprop(true)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        MultiLayerNetwork net2 = new MultiLayerNetwork(conf2);
        net.init();
        net2.init();

        net2.setParams(net.params().dup());

        int batchSize = 3;
        int timeSeriesLength = 8;
        INDArray inEmbedding = Nd4j.create(batchSize, 1, timeSeriesLength);
        INDArray inOneHot = Nd4j.create(batchSize, nClassesIn, timeSeriesLength);
        INDArray outLabels = Nd4j.create(batchSize, 4, timeSeriesLength);

        Random r = new Random(12345);
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < timeSeriesLength; j++) {
                int classIdx = r.nextInt(nClassesIn);
                inEmbedding.putScalar(new int[]{i, 0, j}, classIdx);
                inOneHot.putScalar(new int[]{i, classIdx, j}, 1.0);

                int labelIdx = r.nextInt(4);
                outLabels.putScalar(new int[]{i, labelIdx, j}, 1.0);
            }
        }

        net.setInput(inEmbedding);
        net2.setInput(inOneHot);
        net.setLabels(outLabels);
        net2.setLabels(outLabels);

        net.computeGradientAndScore();
        net2.computeGradientAndScore();

        System.out.println(net.score() + "\t" + net2.score());
        assertEquals(net2.score(), net.score(), 1e-6);

        Map<String, INDArray> gradient = net.gradient().gradientForVariable();
        Map<String, INDArray> gradient2 = net2.gradient().gradientForVariable();
        assertEquals(gradient.size(), gradient2.size());

        for (String s : gradient.keySet()) {
            assertEquals(gradient2.get(s), gradient.get(s));
        }

    }

    @Test
    public void testEmbeddingLayerWithMasking() {
        //Idea: have masking on the input with an embedding and dense layers on input
        //Ensure that the parameter gradients for the inputs don't depend on the inputs when inputs are masked

        int[] miniBatchSizes = {1,2,5};
        int nIn = 2;
        Random r = new Random(12345);

        int numInputClasses = 10;
        int timeSeriesLength = 5;

        for (int nExamples : miniBatchSizes) {
            Nd4j.getRandom().setSeed(12345);

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                    .updater(Updater.SGD)
                    .learningRate(0.1)
                    .seed(12345)
                    .list()
                    .layer(0, new EmbeddingLayer.Builder().activation("tanh").nIn(numInputClasses).nOut(5).build())
                    .layer(1, new DenseLayer.Builder().activation("tanh").nIn(5).nOut(4).build())
                    .layer(2, new GravesLSTM.Builder().activation("tanh").nIn(4).nOut(3).build())
                    .layer(3, new RnnOutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(3).nOut(4).build())
                    .inputPreProcessor(0, new RnnToFeedForwardPreProcessor())
                    .inputPreProcessor(2, new FeedForwardToRnnPreProcessor())
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                    .updater(Updater.SGD)
                    .learningRate(0.1)
                    .seed(12345)
                    .list()
                    .layer(0, new DenseLayer.Builder().activation("tanh").nIn(numInputClasses).nOut(5).build())
                    .layer(1, new DenseLayer.Builder().activation("tanh").nIn(5).nOut(4).build())
                    .layer(2, new GravesLSTM.Builder().activation("tanh").nIn(4).nOut(3).build())
                    .layer(3, new RnnOutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(3).nOut(4).build())
                    .inputPreProcessor(0, new RnnToFeedForwardPreProcessor())
                    .inputPreProcessor(2, new FeedForwardToRnnPreProcessor())
                    .build();

            MultiLayerNetwork net2 = new MultiLayerNetwork(conf2);
            net2.init();

            net2.setParams(net.params().dup());

            INDArray inEmbedding = Nd4j.zeros(nExamples, 1, timeSeriesLength);
            INDArray inDense = Nd4j.zeros(nExamples, numInputClasses, timeSeriesLength);

            INDArray labels = Nd4j.zeros(nExamples,4,timeSeriesLength);

            for( int i=0; i<nExamples; i++ ){
                for( int j=0; j<timeSeriesLength; j++ ){
                    int inIdx = r.nextInt(numInputClasses);
                    inEmbedding.putScalar(new int[]{i,0,j},inIdx);
                    inDense.putScalar(new int[]{i,inIdx,j},1.0);

                    int outIdx = r.nextInt(4);
                    labels.putScalar(new int[]{i,outIdx,j},1.0);
                }
            }

            INDArray inputMask = Nd4j.zeros(nExamples,timeSeriesLength);
            for( int i=0; i<nExamples; i++ ){
                for( int j=0; j<timeSeriesLength; j++ ){
                    inputMask.putScalar(new int[]{i,j},(r.nextBoolean() ? 1.0 : 0.0));
                }
            }

            net.setLayerMaskArrays(inputMask,null);
            net2.setLayerMaskArrays(inputMask,null);
            List<INDArray> actEmbedding = net.feedForward(inEmbedding,false);
            List<INDArray> actDense = net2.feedForward(inDense, false);
            for( int i=1; i<actEmbedding.size(); i++ ){
                assertEquals(actDense.get(i),actEmbedding.get(i));
            }

            net.setLabels(labels);
            net2.setLabels(labels);
            net.computeGradientAndScore();
            net2.computeGradientAndScore();

            System.out.println(net.score() + "\t" + net2.score());
            assertEquals(net2.score(),net.score(),1e-5);

            Map<String,INDArray> gradients = net.gradient().gradientForVariable();
            Map<String,INDArray> gradients2 = net2.gradient().gradientForVariable();
            assertEquals(gradients.keySet(),gradients2.keySet());
            for(String s : gradients.keySet()){
                assertEquals(gradients2.get(s),gradients.get(s));
            }
        }
    }

}

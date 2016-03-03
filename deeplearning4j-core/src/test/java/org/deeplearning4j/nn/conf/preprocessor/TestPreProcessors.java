package org.deeplearning4j.nn.conf.preprocessor;

import static org.junit.Assert.*;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.layers.convolution.ConvolutionLayer;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;
import java.util.Map;

public class TestPreProcessors {

    @Test
    public void testRnnToFeedForwardPreProcessor() {
        int[] miniBatchSizes = {5, 1, 5, 1};
        int[] timeSeriesLengths = {9, 9, 1, 1};

        for (int x = 0; x < miniBatchSizes.length; x++) {
            int miniBatchSize = miniBatchSizes[x];
            int layerSize = 7;
            int timeSeriesLength = timeSeriesLengths[x];

            RnnToFeedForwardPreProcessor proc = new RnnToFeedForwardPreProcessor();
            NeuralNetConfiguration nnc = new NeuralNetConfiguration.Builder()
                    .layer(new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
                            .nIn(layerSize).nOut(layerSize).build())
                    .build();

            DenseLayer layer = LayerFactories.getFactory(nnc.getLayer()).create(nnc);
            layer.setInputMiniBatchSize(miniBatchSize);

            INDArray activations3dc = Nd4j.create(new int[]{miniBatchSize, layerSize, timeSeriesLength}, 'c');
            INDArray activations3df = Nd4j.create(new int[]{miniBatchSize, layerSize, timeSeriesLength}, 'f');
            for (int i = 0; i < miniBatchSize; i++) {
                for (int j = 0; j < layerSize; j++) {
                    for (int k = 0; k < timeSeriesLength; k++) {
                        double value = 100 * i + 10 * j + k;    //value abc -> example=a, neuronNumber=b, time=c
                        activations3dc.putScalar(new int[]{i, j, k}, value);
                        activations3df.putScalar(new int[]{i, j, k}, value);
                    }
                }
            }
            assertEquals(activations3dc, activations3df);


            INDArray activations2dc = proc.preProcess(activations3dc, miniBatchSize);
            INDArray activations2df = proc.preProcess(activations3df, miniBatchSize);
            assertArrayEquals(activations2dc.shape(), new int[]{miniBatchSize * timeSeriesLength, layerSize});
            assertArrayEquals(activations2df.shape(), new int[]{miniBatchSize * timeSeriesLength, layerSize});
            assertEquals(activations2dc, activations2df);

            //Expect each row in activations2d to have order:
            //(example=0,t=0), (example=0,t=1), (example=0,t=2), ..., (example=1,t=0), (example=1,t=1), ...
            int nRows = activations2dc.rows();
            for( int i=0; i<nRows; i++ ){
                INDArray rowc = activations2dc.getRow(i);
                INDArray rowf = activations2df.getRow(i);
                assertArrayEquals(rowc.shape(),new int[]{1,layerSize});
                assertEquals(rowc,rowf);

                int origExampleNum = i / timeSeriesLength;
                int time = i % timeSeriesLength;
                INDArray expectedRow = activations3dc.tensorAlongDimension(time,1,0).getRow(origExampleNum);
                assertTrue(rowc.equals(expectedRow));
                assertTrue(rowf.equals(expectedRow));
            }

            //Given that epsilons and activations have same shape, we can do this (even though it's not the intended use)
            //Basically backprop should be exact opposite of preProcess
            INDArray outc = proc.backprop(activations2dc, miniBatchSize);
            INDArray outf = proc.backprop(activations2df, miniBatchSize);
            assertTrue(outc.equals(activations3dc));
            assertTrue(outf.equals(activations3df));

            //Also check case when epsilons are different orders:
            INDArray eps2d_c = Nd4j.create(activations2dc.shape(),'c');
            INDArray eps2d_f = Nd4j.create(activations2dc.shape(),'f');
            eps2d_c.assign(activations2dc);
            eps2d_f.assign(activations2df);
            INDArray eps3d_c = proc.backprop(eps2d_c, miniBatchSize);
            INDArray eps3d_f = proc.backprop(eps2d_f, miniBatchSize);
            assertEquals(activations3dc, eps3d_c);
            assertEquals(activations3df, eps3d_f);
        }
    }

    @Test
    public void testFeedForwardToRnnPreProcessor() {
        Nd4j.getRandom().setSeed(12345L);

        int[] miniBatchSizes = {5, 1, 5, 1};
        int[] timeSeriesLengths = {9, 9, 1, 1};

        for (int x = 0; x < miniBatchSizes.length; x++) {
            int miniBatchSize = miniBatchSizes[x];
            int layerSize = 7;
            int timeSeriesLength = timeSeriesLengths[x];

            FeedForwardToRnnPreProcessor proc = new FeedForwardToRnnPreProcessor();

            NeuralNetConfiguration nnc = new NeuralNetConfiguration.Builder()
                    .layer(new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
                            .nIn(layerSize).nOut(layerSize).build())
                    .build();

            DenseLayer layer = LayerFactories.getFactory(nnc.getLayer()).create(nnc);
            layer.setInputMiniBatchSize(miniBatchSize);

            INDArray rand = Nd4j.rand(miniBatchSize * timeSeriesLength, layerSize);
            INDArray activations2dc = Nd4j.create(new int[]{miniBatchSize * timeSeriesLength, layerSize}, 'c');
            INDArray activations2df = Nd4j.create(new int[]{miniBatchSize * timeSeriesLength, layerSize}, 'f');
            activations2dc.assign(rand);
            activations2df.assign(rand);
            assertEquals(activations2dc, activations2df);

            INDArray activations3dc = proc.preProcess(activations2dc, miniBatchSize);
            INDArray activations3df = proc.preProcess(activations2df, miniBatchSize);
            assertArrayEquals(new int[]{miniBatchSize, layerSize, timeSeriesLength}, activations3dc.shape());
            assertArrayEquals(new int[]{miniBatchSize, layerSize, timeSeriesLength}, activations3df.shape());
            assertEquals(activations3dc, activations3df);

            int nRows2D = miniBatchSize * timeSeriesLength;
            for (int i = 0; i < nRows2D; i++) {
                int time = i % timeSeriesLength;
                int example = i / timeSeriesLength;

                INDArray row2d = activations2dc.getRow(i);
                INDArray row3dc = activations3dc.tensorAlongDimension(time, 1, 0).getRow(example);
                INDArray row3df = activations3df.tensorAlongDimension(time, 1, 0).getRow(example);

                assertTrue(row2d.equals(row3dc));
                assertTrue(row2d.equals(row3df));
            }

            //Again epsilons and activations have same shape, we can do this (even though it's not the intended use)
            INDArray epsilon2d1 = proc.backprop(activations3dc, miniBatchSize);
            INDArray epsilon2d2 = proc.backprop(activations3df, miniBatchSize);
            assertTrue(epsilon2d1.equals(activations2dc));
            assertTrue(epsilon2d2.equals(activations2dc));

            //Also check backprop with 3d activations in f order vs. c order:
            INDArray act3d_c = Nd4j.create(activations3dc.shape(), 'c');
            act3d_c.assign(activations3dc);
            INDArray act3d_f = Nd4j.create(activations3dc.shape(), 'f');
            act3d_f.assign(activations3dc);

            assertEquals(activations2dc, proc.backprop(act3d_c, miniBatchSize));
            assertEquals(activations2dc, proc.backprop(act3d_f, miniBatchSize));
        }
    }

    @Test
    public void testCnnToRnnPreProcessor() {
        //Two ways to test this:
        // (a) check that doing preProcess + backprop on a given input gives same result
        // (b) compare to ComposableInputPreProcessor(CNNtoFF, FFtoRNN)

        int[] miniBatchSizes = {5, 1};
        int[] timeSeriesLengths = {9, 1};
        int[] inputHeights = {10, 30};
        int[] inputWidths = {10, 30};
        int[] numChannels = {1, 3, 6};
        int cnnNChannelsIn = 3;

        Nd4j.getRandom().setSeed(12345);

        System.out.println();
        for (int miniBatchSize : miniBatchSizes) {
            for (int timeSeriesLength : timeSeriesLengths) {
                for (int inputHeight : inputHeights) {
                    for (int inputWidth : inputWidths) {
                        for (int nChannels : numChannels) {
                            InputPreProcessor proc = new CnnToRnnPreProcessor(inputHeight, inputWidth, nChannels);

                            NeuralNetConfiguration nnc = new NeuralNetConfiguration.Builder()
                                    .layer(new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(inputWidth, inputHeight)
                                            .nIn(cnnNChannelsIn).nOut(nChannels).build()).build();

                            ConvolutionLayer layer = LayerFactories.getFactory(nnc.getLayer()).create(nnc);
                            layer.setInputMiniBatchSize(miniBatchSize);

                            INDArray activationsCnn = Nd4j.rand(
                                    new int[]{miniBatchSize * timeSeriesLength, nChannels, inputHeight, inputWidth});

                            //Check shape of outputs:
                            int prod = nChannels * inputHeight * inputWidth;
                            INDArray activationsRnn = proc.preProcess(activationsCnn, miniBatchSize);
                            assertArrayEquals(new int[]{miniBatchSize, prod, timeSeriesLength},
                                    activationsRnn.shape());

                            //Check backward pass. Given that activations and epsilons have same shape, they should
                            //be opposite operations - i.e., get the same thing back out
                            INDArray twiceProcessed = proc.backprop(activationsRnn, miniBatchSize);
                            assertArrayEquals(activationsCnn.shape(), twiceProcessed.shape());
                            assertEquals(activationsCnn, twiceProcessed);

                            //Second way to check: compare to ComposableInputPreProcessor(CNNtoFF, FFtoRNN)
                            InputPreProcessor compProc = new ComposableInputPreProcessor(
                                    new CnnToFeedForwardPreProcessor(inputHeight, inputWidth, nChannels),
                                    new FeedForwardToRnnPreProcessor());

                            INDArray activationsRnnComp = compProc.preProcess(activationsCnn, miniBatchSize);
                            assertEquals(activationsRnnComp, activationsRnn);

                            INDArray epsilonsRnn = Nd4j.rand(
                                    new int[]{miniBatchSize, nChannels * inputHeight * inputWidth, timeSeriesLength});
                            INDArray epsilonsCnnComp = compProc.backprop(epsilonsRnn, miniBatchSize);
                            INDArray epsilonsCnn = proc.backprop(epsilonsRnn, miniBatchSize);
                            if (!epsilonsCnn.equals(epsilonsCnnComp)) {
                                System.out.println(miniBatchSize + "\t" + timeSeriesLength + "\t" + inputHeight + "\t" +
                                        inputWidth + "\t" + nChannels);
                                System.out.println("expected - epsilonsCnnComp");
                                System.out.println(Arrays.toString(epsilonsCnnComp.shape()));
                                System.out.println(epsilonsCnnComp);
                                System.out.println("actual - epsilonsCnn");
                                System.out.println(Arrays.toString(epsilonsCnn.shape()));
                                System.out.println(epsilonsCnn);
                            }
                            assertEquals(epsilonsCnnComp, epsilonsCnn);
                        }
                    }
                }
            }
        }
    }


    @Test
    public void testRnnToCnnPreProcessor() {
        //Two ways to test this:
        // (a) check that doing preProcess + backprop on a given input gives same result
        // (b) compare to ComposableInputPreProcessor(CNNtoFF, FFtoRNN)

        int[] miniBatchSizes = {5, 1};
        int[] timeSeriesLengths = {9, 1};
        int[] inputHeights = {10, 30};
        int[] inputWidths = {10, 30};
        int[] numChannels = {1, 3, 6};
        int cnnNChannelsIn = 3;

        Nd4j.getRandom().setSeed(12345);

        System.out.println();
        for (int miniBatchSize : miniBatchSizes) {
            for (int timeSeriesLength : timeSeriesLengths) {
                for (int inputHeight : inputHeights) {
                    for (int inputWidth : inputWidths) {
                        for (int nChannels : numChannels) {
                            InputPreProcessor proc = new RnnToCnnPreProcessor(inputHeight, inputWidth, nChannels);

                            NeuralNetConfiguration nnc = new NeuralNetConfiguration.Builder()
                                    .layer(new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(inputWidth, inputHeight)
                                            .nIn(cnnNChannelsIn).nOut(nChannels).build()).build();

                            ConvolutionLayer layer = LayerFactories.getFactory(nnc.getLayer()).create(nnc);
                            layer.setInputMiniBatchSize(miniBatchSize);

                            int[] shape_rnn = new int[]{miniBatchSize, nChannels * inputHeight * inputWidth, timeSeriesLength};
                            INDArray rand = Nd4j.rand(shape_rnn);
                            INDArray activationsRnn_c = Nd4j.create(shape_rnn, 'c');
                            INDArray activationsRnn_f = Nd4j.create(shape_rnn, 'f');
                            activationsRnn_c.assign(rand);
                            activationsRnn_f.assign(rand);
                            assertEquals(activationsRnn_c,activationsRnn_f);

                            //Check shape of outputs:
                            INDArray activationsCnn_c = proc.preProcess(activationsRnn_c, miniBatchSize);
                            INDArray activationsCnn_f = proc.preProcess(activationsRnn_f, miniBatchSize);
                            int[] shape_cnn = new int[]{miniBatchSize * timeSeriesLength, nChannels, inputHeight, inputWidth};
                            assertArrayEquals(shape_cnn, activationsCnn_c.shape());
                            assertArrayEquals(shape_cnn, activationsCnn_f.shape());
                            assertEquals(activationsCnn_c,activationsCnn_f);

                            //Check backward pass. Given that activations and epsilons have same shape, they should
                            //be opposite operations - i.e., get the same thing back out
                            INDArray twiceProcessed_c = proc.backprop(activationsCnn_c, miniBatchSize);
                            INDArray twiceProcessed_f = proc.backprop(activationsCnn_c, miniBatchSize);
                            assertArrayEquals(shape_rnn, twiceProcessed_c.shape());
                            assertArrayEquals(shape_rnn, twiceProcessed_f.shape());
                            assertEquals(activationsRnn_c, twiceProcessed_c);
                            assertEquals(activationsRnn_c, twiceProcessed_f);

                            //Second way to check: compare to ComposableInputPreProcessor(RNNtoFF, FFtoCNN)
                            InputPreProcessor compProc = new ComposableInputPreProcessor(
                                    new RnnToFeedForwardPreProcessor(),
                                    new FeedForwardToCnnPreProcessor(inputHeight, inputWidth, nChannels));

                            INDArray activationsCnnComp_c = compProc.preProcess(activationsRnn_c, miniBatchSize);
                            INDArray activationsCnnComp_f = compProc.preProcess(activationsRnn_f, miniBatchSize);
                            assertEquals(activationsCnnComp_c, activationsCnn_c);
                            assertEquals(activationsCnnComp_f, activationsCnn_f);

                            int[] epsilonShape = new int[]{miniBatchSize * timeSeriesLength, nChannels, inputHeight, inputWidth};
                            rand = Nd4j.rand(epsilonShape);
                            INDArray epsilonsCnn_c = Nd4j.create(epsilonShape,'c');
                            INDArray epsilonsCnn_f = Nd4j.create(epsilonShape,'f');
                            epsilonsCnn_c.assign(rand);
                            epsilonsCnn_f.assign(rand);

                            INDArray epsilonsRnnComp_c = compProc.backprop(epsilonsCnn_c, miniBatchSize);
                            INDArray epsilonsRnnComp_f = compProc.backprop(epsilonsCnn_f, miniBatchSize);
                            assertEquals(epsilonsRnnComp_c,epsilonsRnnComp_f);
                            INDArray epsilonsRnn_c = proc.backprop(epsilonsCnn_c, miniBatchSize);
                            INDArray epsilonsRnn_f = proc.backprop(epsilonsCnn_f, miniBatchSize);
                            assertEquals(epsilonsRnn_c,epsilonsRnn_f);

                            if (!epsilonsRnn_c.equals(epsilonsRnnComp_c)) {
                                System.out.println(miniBatchSize + "\t" + timeSeriesLength + "\t" + inputHeight + "\t" +
                                        inputWidth + "\t" + nChannels);
                                System.out.println("expected - epsilonsRnnComp");
                                System.out.println(Arrays.toString(epsilonsRnnComp_c.shape()));
                                System.out.println(epsilonsRnnComp_c);
                                System.out.println("actual - epsilonsRnn");
                                System.out.println(Arrays.toString(epsilonsRnn_c.shape()));
                                System.out.println(epsilonsRnn_c);
                            }
                            assertEquals(epsilonsRnnComp_c, epsilonsRnn_c);
                            assertEquals(epsilonsRnnComp_c, epsilonsRnn_f);
                        }
                    }
                }
            }
        }
    }


    @Test
    public void testAutoAdditionOfPreprocessors(){
        //FF->RNN and RNN->FF
        MultiLayerConfiguration conf1 = new NeuralNetConfiguration.Builder()
                .list()
                .layer(0, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
                        .nIn(5).nOut(6).build())
                .layer(1, new GravesLSTM.Builder().nIn(6).nOut(7).build())
                .layer(2, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
                        .nIn(7).nOut(8).build())
                .layer(3, new RnnOutputLayer.Builder().nIn(8).nOut(9).build())
                .build();
        //Expect preprocessors: layer1: FF->RNN; 2: RNN->FF; 3: FF->RNN
        assertEquals(3, conf1.getInputPreProcessors().size());
        assertTrue(conf1.getInputPreProcess(1) instanceof FeedForwardToRnnPreProcessor);
        assertTrue(conf1.getInputPreProcess(2) instanceof RnnToFeedForwardPreProcessor);
        assertTrue(conf1.getInputPreProcess(3) instanceof FeedForwardToRnnPreProcessor);


        //FF-> CNN, CNN-> FF, FF->RNN
        MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                .list()
                .layer(0, new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder()
                        .nOut(10).kernelSize(5, 5).stride(1, 1).build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nOut(6).build())
                .layer(2, new RnnOutputLayer.Builder().nIn(6).nOut(5).build())
                .cnnInputSize(28, 28, 1)
                .build();
        //Expect preprocessors: 0: FF->CNN, 1: CNN->FF; 2: FF->RNN
        assertEquals(3, conf2.getInputPreProcessors().size());
        assertTrue(conf2.getInputPreProcess(0) instanceof FeedForwardToCnnPreProcessor);
        assertTrue(conf2.getInputPreProcess(1) instanceof CnnToFeedForwardPreProcessor);
        assertTrue(conf2.getInputPreProcess(2) instanceof FeedForwardToRnnPreProcessor);

        //FF->CNN and CNN->RNN:
        MultiLayerConfiguration conf3 = new NeuralNetConfiguration.Builder()
                .list()
                .layer(0, new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder()
                        .nOut(10).kernelSize(5, 5).stride(1, 1).build())
                .layer(1, new GravesLSTM.Builder().nOut(6).build())
                .layer(2, new RnnOutputLayer.Builder().nIn(6).nOut(5).build())
                .cnnInputSize(28, 28, 1)
                .build();
        //Expect preprocessors: 0: FF->CNN, 1: CNN->RNN;
        assertEquals(2, conf3.getInputPreProcessors().size());
        assertTrue(conf3.getInputPreProcess(0) instanceof FeedForwardToCnnPreProcessor);
        assertTrue(conf3.getInputPreProcess(1) instanceof CnnToRnnPreProcessor);
    }

    @Test
    public void testCnnToDense(){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        //.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.01) // default
                        //.momentum(0.9)
                .regularization(true)
                .list()
                .layer(0, new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(4, 4) // 28*28*1 => 15*15*10
                        .nIn(1)
                        .nOut(10)
                        .padding(2, 2)
                        .stride(2, 2)
                        .weightInit(WeightInit.RELU)
                        .activation("relu")
                        .build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().activation("relu")
                        .nOut(200).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.RMSE_XENT)
                        .nIn(200)
                        .nOut(5)
                        .weightInit(WeightInit.RELU)
                        .activation("softmax")
                        .updater(Updater.SGD)
                        .build())
                .cnnInputSize(28,28,1)
                .backprop(true).pretrain(false).build();

        assertNotNull(conf.getInputPreProcess(0));
        assertNotNull(conf.getInputPreProcess(1));

        assertTrue(conf.getInputPreProcess(0) instanceof FeedForwardToCnnPreProcessor);
        assertTrue(conf.getInputPreProcess(1) instanceof CnnToFeedForwardPreProcessor);

        FeedForwardToCnnPreProcessor ffcnn = (FeedForwardToCnnPreProcessor)conf.getInputPreProcess(0);
        CnnToFeedForwardPreProcessor cnnff = (CnnToFeedForwardPreProcessor)conf.getInputPreProcess(1);

        assertEquals(28,ffcnn.getInputHeight());
        assertEquals(28,ffcnn.getInputWidth());
        assertEquals(1,ffcnn.getNumChannels());

        assertEquals(15,cnnff.getInputHeight());
        assertEquals(15,cnnff.getInputWidth());
        assertEquals(10,cnnff.getNumChannels());

        assertEquals(15*15*10,((FeedForwardLayer)conf.getConf(1).getLayer()).getNIn());
    }
}

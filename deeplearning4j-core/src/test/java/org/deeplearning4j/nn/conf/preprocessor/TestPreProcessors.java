package org.deeplearning4j.nn.conf.preprocessor;

import static org.junit.Assert.*;

import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.convolution.ConvolutionLayer;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

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


            INDArray activations2dc = proc.preProcess(activations3dc, layer);
            INDArray activations2df = proc.preProcess(activations3df, layer);
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
            INDArray outc = proc.backprop(activations2dc,layer);
            INDArray outf = proc.backprop(activations2df, layer);
            assertTrue(outc.equals(activations3dc));
            assertTrue(outf.equals(activations3df));

            //Also check case when epsilons are different orders:
            INDArray eps2d_c = Nd4j.create(activations2dc.shape(),'c');
            INDArray eps2d_f = Nd4j.create(activations2dc.shape(),'f');
            eps2d_c.assign(activations2dc);
            eps2d_f.assign(activations2df);
            INDArray eps3d_c = proc.backprop(eps2d_c, layer);
            INDArray eps3d_f = proc.backprop(eps2d_f, layer);
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

            INDArray activations3dc = proc.preProcess(activations2dc, layer);
            INDArray activations3df = proc.preProcess(activations2df, layer);
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
            INDArray epsilon2d1 = proc.backprop(activations3dc, layer);
            INDArray epsilon2d2 = proc.backprop(activations3df, layer);
            assertTrue(epsilon2d1.equals(activations2dc));
            assertTrue(epsilon2d2.equals(activations2dc));

            //Also check backprop with 3d activations in f order vs. c order:
            INDArray act3d_c = Nd4j.create(activations3dc.shape(), 'c');
            act3d_c.assign(activations3dc);
            INDArray act3d_f = Nd4j.create(activations3dc.shape(), 'f');
            act3d_f.assign(activations3dc);

            assertEquals(activations2dc, proc.backprop(act3d_c, layer));
            assertEquals(activations2dc, proc.backprop(act3d_f, layer));
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
                            INDArray activationsRnn = proc.preProcess(activationsCnn, layer);
                            assertArrayEquals(new int[]{miniBatchSize, prod, timeSeriesLength},
                                    activationsRnn.shape());

                            //Check backward pass. Given that activations and epsilons have same shape, they should
                            //be opposite operations - i.e., get the same thing back out
                            INDArray twiceProcessed = proc.backprop(activationsRnn, layer);
                            assertArrayEquals(activationsCnn.shape(), twiceProcessed.shape());
                            assertEquals(activationsCnn, twiceProcessed);

                            //Second way to check: compare to ComposableInputPreProcessor(CNNtoFF, FFtoRNN)
                            InputPreProcessor compProc = new ComposableInputPreProcessor(
                                    new CnnToFeedForwardPreProcessor(inputHeight, inputWidth, nChannels),
                                    new FeedForwardToRnnPreProcessor());

                            INDArray activationsRnnComp = compProc.preProcess(activationsCnn, layer);
                            assertEquals(activationsRnnComp, activationsRnn);

                            INDArray epsilonsRnn = Nd4j.rand(
                                    new int[]{miniBatchSize, nChannels * inputHeight * inputWidth, timeSeriesLength});
                            INDArray epsilonsCnnComp = compProc.backprop(epsilonsRnn, layer);
                            INDArray epsilonsCnn = proc.backprop(epsilonsRnn, layer);
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
                            INDArray activationsCnn_c = proc.preProcess(activationsRnn_c, layer);
                            INDArray activationsCnn_f = proc.preProcess(activationsRnn_f, layer);
                            int[] shape_cnn = new int[]{miniBatchSize * timeSeriesLength, nChannels, inputHeight, inputWidth};
                            assertArrayEquals(shape_cnn, activationsCnn_c.shape());
                            assertArrayEquals(shape_cnn, activationsCnn_f.shape());
                            assertEquals(activationsCnn_c,activationsCnn_f);

                            //Check backward pass. Given that activations and epsilons have same shape, they should
                            //be opposite operations - i.e., get the same thing back out
                            INDArray twiceProcessed_c = proc.backprop(activationsCnn_c, layer);
                            INDArray twiceProcessed_f = proc.backprop(activationsCnn_c, layer);
                            assertArrayEquals(shape_rnn, twiceProcessed_c.shape());
                            assertArrayEquals(shape_rnn, twiceProcessed_f.shape());
                            assertEquals(activationsRnn_c, twiceProcessed_c);
                            assertEquals(activationsRnn_c, twiceProcessed_f);

                            //Second way to check: compare to ComposableInputPreProcessor(RNNtoFF, FFtoCNN)
                            InputPreProcessor compProc = new ComposableInputPreProcessor(
                                    new RnnToFeedForwardPreProcessor(),
                                    new FeedForwardToCnnPreProcessor(inputHeight, inputWidth, nChannels));

                            INDArray activationsCnnComp_c = compProc.preProcess(activationsRnn_c, layer);
                            INDArray activationsCnnComp_f = compProc.preProcess(activationsRnn_f, layer);
                            assertEquals(activationsCnnComp_c, activationsCnn_c);
                            assertEquals(activationsCnnComp_f, activationsCnn_f);

                            int[] epsilonShape = new int[]{miniBatchSize * timeSeriesLength, nChannels, inputHeight, inputWidth};
                            rand = Nd4j.rand(epsilonShape);
                            INDArray epsilonsCnn_c = Nd4j.create(epsilonShape,'c');
                            INDArray epsilonsCnn_f = Nd4j.create(epsilonShape,'f');
                            epsilonsCnn_c.assign(rand);
                            epsilonsCnn_f.assign(rand);

                            INDArray epsilonsRnnComp_c = compProc.backprop(epsilonsCnn_c, layer);
                            INDArray epsilonsRnnComp_f = compProc.backprop(epsilonsCnn_f, layer);
                            assertEquals(epsilonsRnnComp_c,epsilonsRnnComp_f);
                            INDArray epsilonsRnn_c = proc.backprop(epsilonsCnn_c, layer);
                            INDArray epsilonsRnn_f = proc.backprop(epsilonsCnn_f, layer);
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
}

package org.deeplearning4j.nn.conf.preprocessor;

import static org.junit.Assert.*;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 **/

public class CNNProcessorTest {
    private static int rows = 28;
    private static int cols = 28;
    private static INDArray in2D = Nd4j.create(1, 784);
    private static INDArray in3D = Nd4j.create(20, 784, 7);
    private static INDArray in4D = Nd4j.create(20, 1, 28, 28);


    @Test
    public void testFeedForwardToCnnPreProcessor() {
        FeedForwardToCnnPreProcessor convProcessor = new FeedForwardToCnnPreProcessor(rows, cols, 1);

        INDArray check2to4 = convProcessor.preProcess(in2D,-1);
        int val2to4 = check2to4.shape().length;
        assertTrue(val2to4 == 4);
        assertEquals(Nd4j.create(1, 1, 28, 28), check2to4);

        INDArray check4to4 = convProcessor.preProcess(in4D,-1);
        int val4to4 = check4to4.shape().length;
        assertTrue(val4to4 == 4);
        assertEquals(Nd4j.create(20, 1, 28, 28), check4to4);
    }

    @Test
    public void testFeedForwardToCnnPreProcessor2() {
        int[] nRows = {1, 5, 20};
        int[] nCols = {1, 5, 20};
        int[] nDepth = {1, 3};
        int[] nMiniBatchSize = {1, 5};
        for( int rows : nRows ){
            for( int cols : nCols) {
                for( int d : nDepth) {
                    FeedForwardToCnnPreProcessor convProcessor = new FeedForwardToCnnPreProcessor(rows, cols, d);

                    for( int miniBatch : nMiniBatchSize ) {
                        int[] ffShape = new int[]{miniBatch,rows*cols*d};
                        INDArray rand = Nd4j.rand(ffShape);
                        INDArray ffInput_c = Nd4j.create(ffShape, 'c');
                        INDArray ffInput_f = Nd4j.create(ffShape,'f');
                        ffInput_c.assign(rand);
                        ffInput_f.assign(rand);
                        assertEquals(ffInput_c, ffInput_f);

                        //Test forward pass:
                        INDArray convAct_c = convProcessor.preProcess(ffInput_c,-1);
                        INDArray convAct_f = convProcessor.preProcess(ffInput_f,-1);
                        int[] convShape = {miniBatch,d,rows,cols};
                        assertArrayEquals(convShape, convAct_c.shape());
                        assertArrayEquals(convShape, convAct_f.shape());
                        assertEquals(convAct_c,convAct_f);

                        //Check values:
                        //CNN reshaping (for each example) takes a 1d vector and converts it to 3d
                        // (4d total, for minibatch data)
                        //1d vector is assumed to be rows from depth 0 concatenated, followed by depth 1, etc
                        for( int ex=0; ex<miniBatch; ex++ ) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = 0; c < cols; c++) {
                                    for (int depth = 0; depth < d; depth++) {
                                        int origPosition = depth * (rows * cols) + r * cols + c;  //pos in vector
                                        double vecValue = ffInput_c.getDouble(ex,origPosition);
                                        double convValue = convAct_c.getDouble(ex,depth,r,c);
                                        assertEquals(vecValue,convValue,0.0);
                                    }
                                }
                            }
                        }

                        //Test backward pass:
                        //Idea is that backward pass should do opposite to forward pass
                        INDArray epsilon4_c = Nd4j.create(convShape,'c');
                        INDArray epsilon4_f = Nd4j.create(convShape,'f');
                        epsilon4_c.assign(convAct_c);
                        epsilon4_f.assign(convAct_f);
                        INDArray epsilon2_c = convProcessor.backprop(epsilon4_c,-1);
                        INDArray epsilon2_f = convProcessor.backprop(epsilon4_f,-1);
                        assertEquals(ffInput_c,epsilon2_c);
                        assertEquals(ffInput_c,epsilon2_f);
                    }
                }
            }
        }
    }


    @Test
    public void testFeedForwardToCnnPreProcessorBackprop() {
        FeedForwardToCnnPreProcessor convProcessor = new FeedForwardToCnnPreProcessor(rows, cols, 1);
        convProcessor.preProcess(in2D,-1);

        INDArray check2to2 = convProcessor.backprop(in2D,-1);
        int val2to2 = check2to2.shape().length;
        assertTrue(val2to2 == 2);
        assertEquals(Nd4j.create(1, 784), check2to2);
    }

    @Test
    public void testCnnToFeedForwardProcessor() {
        CnnToFeedForwardPreProcessor convProcessor = new CnnToFeedForwardPreProcessor(rows, cols, 1);

        INDArray check2to4 = convProcessor.backprop(in2D,-1);
        int val2to4 = check2to4.shape().length;
        assertTrue(val2to4 == 4);
        assertEquals(Nd4j.create(1, 1, 28, 28), check2to4);

        INDArray check4to4 = convProcessor.backprop(in4D,-1);
        int val4to4 = check4to4.shape().length;
        assertTrue(val4to4 == 4);
        assertEquals(Nd4j.create(20, 1, 28, 28), check4to4);
    }

    @Test
    public void testCnnToFeedForwardPreProcessorBackprop() {
        CnnToFeedForwardPreProcessor convProcessor = new CnnToFeedForwardPreProcessor(rows, cols, 1);
        convProcessor.preProcess(in4D,-1);

        INDArray check2to2 = convProcessor.preProcess(in2D,-1);
        int val2to2 = check2to2.shape().length;
        assertTrue(val2to2 == 2);
        assertEquals(Nd4j.create(1, 784), check2to2);

        INDArray check4to2 = convProcessor.preProcess(in4D, -1);
        int val4to2 = check4to2.shape().length;
        assertTrue(val4to2 == 2);
        assertEquals(Nd4j.create(20, 784), check4to2);
    }

    @Test
    public void testCnnToFeedForwardPreProcessor2() {
        int[] nRows = {1, 5, 20};
        int[] nCols = {1, 5, 20};
        int[] nDepth = {1, 3};
        int[] nMiniBatchSize = {1, 5};
        for( int rows : nRows ){
            for( int cols : nCols ){
                for( int d : nDepth ){
                    CnnToFeedForwardPreProcessor convProcessor = new CnnToFeedForwardPreProcessor(rows, cols, d);

                    for( int miniBatch : nMiniBatchSize ) {
                        int[] convActShape = new int[]{miniBatch,d,rows,cols};
                        INDArray rand = Nd4j.rand(convActShape);
                        INDArray convInput_c = Nd4j.create(convActShape, 'c');
                        INDArray convInput_f = Nd4j.create(convActShape, 'f');
                        convInput_c.assign(rand);
                        convInput_f.assign(rand);
                        assertEquals(convInput_c, convInput_f);

                        //Test forward pass:
                        INDArray ffAct_c = convProcessor.preProcess(convInput_c,-1);
                        INDArray ffAct_f = convProcessor.preProcess(convInput_f,-1);
                        int[] ffActShape = {miniBatch,d*rows*cols};
                        assertArrayEquals(ffActShape, ffAct_c.shape());
                        assertArrayEquals(ffActShape, ffAct_f.shape());
                        assertEquals(ffAct_c,ffAct_f);

                        //Check values:
                        //CNN reshaping (for each example) takes a 1d vector and converts it to 3d
                        // (4d total, for minibatch data)
                        //1d vector is assumed to be rows from depth 0 concatenated, followed by depth 1, etc
                        for( int ex=0; ex<miniBatch; ex++ ) {
                            for (int r = 0; r < rows; r++) {
                                for (int c = 0; c < cols; c++) {
                                    for (int depth = 0; depth < d; depth++) {
                                        int vectorPosition = depth * (rows * cols) + r * cols + c;  //pos in vector after reshape
                                        double vecValue = ffAct_c.getDouble(ex,vectorPosition);
                                        double convValue = convInput_c.getDouble(ex,depth,r,c);
                                        assertEquals(convValue,vecValue,0.0);
                                    }
                                }
                            }
                        }

                        //Test backward pass:
                        //Idea is that backward pass should do opposite to forward pass
                        INDArray epsilon2_c = Nd4j.create(ffActShape,'c');
                        INDArray epsilon2_f = Nd4j.create(ffActShape,'f');
                        epsilon2_c.assign(ffAct_c);
                        epsilon2_f.assign(ffAct_c);
                        INDArray epsilon4_c = convProcessor.backprop(epsilon2_c,-1);
                        INDArray epsilon4_f = convProcessor.backprop(epsilon2_f,-1);
                        assertEquals(convInput_c,epsilon4_c);
                        assertEquals(convInput_c,epsilon4_f);
                    }
                }
            }
        }
    }

    @Test
    public void testCNNInputPreProcessorMnist() throws Exception {
        int numSamples = 1;
        int batchSize = 1;

        DataSet mnistIter = new MnistDataSetIterator(batchSize, numSamples, true).next();
        MultiLayerNetwork model = getCNNMnistConfig();
        model.init();
        model.fit(mnistIter);

        int val2to4 = model.getLayer(0).input().shape().length;
        assertTrue(val2to4 == 4);

        int val4to4 = model.getLayer(1).input().shape().length;
        assertTrue(val4to4 == 4);

    }


    public static MultiLayerNetwork getCNNMnistConfig()  {

        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        MultiLayerConfiguration.Builder conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .iterations(5)
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .list()
                .layer(0, new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(new int[]{9, 9},new int[]{1,1})
                        .nOut(20)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(20)
                        .nOut(10)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .build());
        new ConvolutionLayerSetup(conf,28,28,1);
        return new MultiLayerNetwork(conf.build());
    }
}

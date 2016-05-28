package org.deeplearning4j.nn.layers.normalization;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.BatchNormalizationParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Random;

import static org.junit.Assert.*;

/**
 */
public class BatchNormalizationTest {
    protected INDArray dnnInput = Nd4j.linspace(0,31,32).reshape(2,16);
    protected INDArray dnnEpsilon = Nd4j.linspace(0,31,32).reshape(2,16);

    protected INDArray cnnInput = Nd4j.linspace(0,63,64).reshape(2, 2, 4, 4);
    protected INDArray cnnEpsilon = Nd4j.linspace(0,63,64).reshape(2, 2, 4, 4);

    @Before
    public void doBefore() {
    }

    protected Layer setupActivations(int nIn, int nOut){
        BatchNormalization bN = new BatchNormalization.Builder().nIn(nIn).nOut(nOut).build();
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(1).layer(bN).build();

        int numParams = LayerFactories.getFactory(conf).initializer().numParams(conf,true);
        INDArray params = Nd4j.create(1, numParams);
        Layer layer =  LayerFactories.getFactory(conf).create(conf, null, 0, params);
        return layer;
    }

    @Test
    public void testDnnShapeBatchNormForward() {
        Layer layer = setupActivations(2, 16);
        // Confirm param initial shape before override
        assertArrayEquals(new int[]{1,16}, layer.getParam("gamma").shape());
        assertArrayEquals(new int[]{1,16}, layer.getParam("beta").shape());
        layer.setParam("gamma", Nd4j.linspace(0,15,16));
        layer.setParam("beta", Nd4j.linspace(0,15,16));


        INDArray activationsActual = layer.preOutput(dnnInput);
        INDArray activationsExpected = Nd4j.create(new double[] {
                0.00000000e+00,   7.81248399e-11,   1.56249680e-10,
                2.34374298e-10,   3.12499360e-10,   3.90624422e-10,
                4.68748595e-10,   5.46873657e-10,   6.24998719e-10,
                7.03122893e-10,   7.81248843e-10,   8.59373017e-10,
                9.37497191e-10,   1.01562314e-09,   1.09374731e-09,
                1.17187327e-09,   0.00000000e+00,   2.00000000e+00,
                4.00000000e+00,   6.00000000e+00,   8.00000000e+00,
                1.00000000e+01,   1.20000000e+01,   1.40000000e+01,
                1.60000000e+01,   1.80000000e+01,   2.00000000e+01,
                2.20000000e+01,   2.40000000e+01,   2.60000000e+01,
                2.80000000e+01,   3.00000000e+01
        },new int[]{2, 16});

        assertEquals(activationsExpected, activationsActual);
        assertArrayEquals(activationsExpected.shape(), activationsActual.shape());
    }


    @Test
    public void testDnnShapeBatchNormBack(){
        Layer layer = setupActivations(2, 16);
        layer.setParam("gamma", Nd4j.linspace(0,15,16));
        layer.setParam("beta", Nd4j.linspace(0,15,16));

        layer.preOutput(dnnInput);
        layer.setBackpropGradientsViewArray(Nd4j.create(1,32));
        Pair<Gradient, INDArray> actualOut = layer.backpropGradient(dnnEpsilon);

        INDArray dnnExpectedEpsilonOut = Nd4j.create(new double[] {
                0.00000000e+00,  -1.56249680e-10,  -3.12499360e-10,
                -4.68748595e-10,  -6.24998719e-10,  -7.81248843e-10,
                -9.37497191e-10,  -1.09374731e-09,  -1.24999744e-09,
                -1.40624934e-09,  -1.56249413e-09,  -1.71874603e-09,
                -1.87499438e-09,  -2.03124273e-09,  -2.18749818e-09,
                -2.34373942e-09,   0.00000000e+00,   1.56249680e-10,
                3.12499804e-10,   4.68748595e-10,   6.24997831e-10,
                7.81248843e-10,   9.37497191e-10,   1.09374731e-09,
                1.24999744e-09,   1.40624579e-09,   1.56249769e-09,
                1.71874603e-09,   1.87499438e-09,   2.03124983e-09,
                2.18749818e-09,   2.34374653e-09
        },new int[]{2, 16});


        // short calculation expected output
        INDArray dnnExpectedEpsilonOutOther = Nd4j.create(new double[] {
                16.,  15.,  14.,  13.,  12.,  11.,  10.,   9.,   8.,   7.,   6.,
                5.,   4.,   3.,   2.,   1., -16., -15., -14., -13., -12., -11.,
                -10.,  -9.,  -8.,  -7.,  -6.,  -5.,  -4.,  -3.,  -2.,  -1.
        },new int[]{2, 16});

        INDArray expectedGGamma = Nd4j.create(new double[]
                {
                    16.,  16.,  16.,  16.,  16.,  16.,  16.,  16.,  16.,  16.,  16.,
                    16.,  16.,  16.,  16.,  16.
                }, new int[] {1, 16});

        INDArray expectedBeta = Nd4j.create(new double[]
                {
                    16.,  18.,  20.,  22.,  24.,  26.,  28.,  30.,  32.,  34.,  36.,
                    38.,  40.,  42.,  44.,  46.
                }, new int[] {1, 16});

        // arrays are the same but assert does not see that
        assertEquals(dnnExpectedEpsilonOut, actualOut.getSecond());
        assertEquals(expectedGGamma, actualOut.getFirst().getGradientFor("gamma"));
        assertEquals(expectedBeta, actualOut.getFirst().getGradientFor("beta"));
    }

    @Test
    public void testCnnShapeBatchNormForward() {
        Layer layer = setupActivations(2, 2);
        // Confirm param initial shape before override
        assertArrayEquals(new int[]{1,2}, layer.getParam("gamma").shape());
        assertArrayEquals(new int[]{1,2}, layer.getParam("beta").shape());

        layer.setParam("gamma", Nd4j.linspace(2,3,2));
        layer.setParam("beta", Nd4j.linspace(2,3,2));
        INDArray activationsActual = layer.preOutput(cnnInput);
        INDArray activationsExpected = Nd4j.create(new double[] {
                3.90625310e-11,   3.90625310e-11,   3.90625310e-11,
                3.90625310e-11,   3.90625310e-11,   3.90625310e-11,
                3.90625310e-11,   3.90625310e-11,   3.90625310e-11,
                3.90625310e-11,   3.90625310e-11,   3.90625310e-11,
                3.90625310e-11,   3.90625310e-11,   3.90625310e-11,
                3.90625310e-11,   5.85940185e-11,   5.85940185e-11,
                5.85940185e-11,   5.85940185e-11,   5.85940185e-11,
                5.85940185e-11,   5.85940185e-11,   5.85940185e-11,
                5.85940185e-11,   5.85940185e-11,   5.85940185e-11,
                5.85940185e-11,   5.85940185e-11,   5.85940185e-11,
                5.85940185e-11,   5.85940185e-11,   4.00000000e+00,
                4.00000000e+00,   4.00000000e+00,   4.00000000e+00,
                4.00000000e+00,   4.00000000e+00,   4.00000000e+00,
                4.00000000e+00,   4.00000000e+00,   4.00000000e+00,
                4.00000000e+00,   4.00000000e+00,   4.00000000e+00,
                4.00000000e+00,   4.00000000e+00,   4.00000000e+00,
                6.00000000e+00,   6.00000000e+00,   6.00000000e+00,
                6.00000000e+00,   6.00000000e+00,   6.00000000e+00,
                6.00000000e+00,   6.00000000e+00,   6.00000000e+00,
                6.00000000e+00,   6.00000000e+00,   6.00000000e+00,
                6.00000000e+00,   6.00000000e+00,   6.00000000e+00,
                6.00000000e+00
        },new int[]{2,2,4,4});

        assertEquals(activationsExpected, activationsActual);
        assertArrayEquals(activationsExpected.shape(), activationsActual.shape());
    }

    @Test
    public void testCnnShapeBatchNormBack(){
        Layer layer = setupActivations(2, 2);
        layer.setParam("gamma", Nd4j.linspace(2,3,2));
        layer.setParam("beta", Nd4j.linspace(2,3,2));
        layer.preOutput(cnnInput);
        layer.setBackpropGradientsViewArray(Nd4j.create(1,4));
        Pair<Gradient, INDArray> actualOut = layer.backpropGradient(cnnEpsilon);

        INDArray expectedEpsilonOut = Nd4j.create(new double[] {
                -7.81250620e-11,  -7.81250620e-11,  -7.81250620e-11,
                -7.81250620e-11,  -7.81250620e-11,  -7.81250620e-11,
                -7.81250620e-11,  -7.81250620e-11,  -7.81250620e-11,
                -7.81250620e-11,  -7.81250620e-11,  -7.81250620e-11,
                -7.81250620e-11,  -7.81250620e-11,  -7.81250620e-11,
                -7.81250620e-11,  -1.17187149e-10,  -1.17187149e-10,
                -1.17187149e-10,  -1.17187149e-10,  -1.17188037e-10,
                -1.17187149e-10,  -1.17187149e-10,  -1.17187149e-10,
                -1.17187149e-10,  -1.17187149e-10,  -1.17187149e-10,
                -1.17188037e-10,  -1.17186261e-10,  -1.17186261e-10,
                -1.17188037e-10,  -1.17188037e-10,   7.81250620e-11,
                7.81250620e-11,   7.81246179e-11,   7.81255061e-11,
                7.81250620e-11,   7.81250620e-11,   7.81246179e-11,
                7.81255061e-11,   7.81255061e-11,   7.81250620e-11,
                7.81246179e-11,   7.81246179e-11,   7.81255061e-11,
                7.81255061e-11,   7.81250620e-11,   7.81246179e-11,
                1.17187149e-10,   1.17187149e-10,   1.17188037e-10,
                1.17187149e-10,   1.17188037e-10,   1.17187149e-10,
                1.17187149e-10,   1.17188037e-10,   1.17187149e-10,
                1.17188037e-10,   1.17187149e-10,   1.17186261e-10,
                1.17188037e-10,   1.17188037e-10,   1.17188037e-10,
                1.17186261e-10
        },new int[]{2, 2, 4, 4});

        INDArray expectedGGamma = Nd4j.create(new double[]
                {
                        512, 512
                }, new int[] {1, 2});

        INDArray expectedBeta = Nd4j.create(new double[]
                {
                        752, 1264
                }, new int[] {1, 2});

        // arrays are the same but assert does not see that
        assertEquals(expectedEpsilonOut, actualOut.getSecond());
        assertEquals(expectedGGamma, actualOut.getFirst().getGradientFor("gamma"));
        assertEquals(expectedBeta, actualOut.getFirst().getGradientFor("beta"));

    }

    @Test
    public void testMultiCNNBNLayer() throws Exception {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .seed(123)
                .list()
                .layer(0, new ConvolutionLayer.Builder().nIn(1).nOut(6).weightInit(WeightInit.XAVIER).activation("relu").build())
                .layer(1, new BatchNormalization.Builder().build())
                .layer(2, new DenseLayer.Builder().nOut(2).build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .nIn(2).nOut(10).build())
                .backprop(true).pretrain(false)
                .cnnInputSize(28,28,1)
                .build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        DataSetIterator iter = new MnistDataSetIterator(2, 2);
        DataSet next = iter.next();

        network.setInput(next.getFeatureMatrix());
        INDArray activationsActual = network.preOutput(next.getFeatureMatrix());
        assertEquals(10, activationsActual.shape()[1], 1e-2);

        network.fit(next);
        INDArray actualGammaParam = network.getLayer(1).getParam(BatchNormalizationParamInitializer.GAMMA);
        INDArray actualBetaParam = network.getLayer(1).getParam(BatchNormalizationParamInitializer.BETA);
        assertTrue(actualGammaParam != null);
        assertTrue(actualBetaParam != null);
    }

    @Test
    public void testDBNBNMultiLayer() throws Exception {
        DataSetIterator iter = new MnistDataSetIterator(2, 2);
        DataSet next = iter.next();

        // Run with separate activation layer
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(2)
                .seed(123)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(28*28*1).nOut(10).weightInit(WeightInit.XAVIER).activation("relu").build())
                .layer(1, new BatchNormalization.Builder().nOut(10).build())
                .layer(2, new ActivationLayer.Builder().activation("relu").build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .nIn(10).nOut(10).build())
                .backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

        network.setInput(next.getFeatureMatrix());
        INDArray activationsActual = network.preOutput(next.getFeatureMatrix());
        assertEquals(10, activationsActual.shape()[1], 1e-2);

        network.fit(next);
        INDArray actualGammaParam = network.getLayer(1).getParam(BatchNormalizationParamInitializer.GAMMA);
        INDArray actualBetaParam = network.getLayer(1).getParam(BatchNormalizationParamInitializer.BETA);
        assertTrue(actualGammaParam != null);
        assertTrue(actualBetaParam != null);
    }

    @Ignore@Test
    public void testMultiLSTMBNLayer() throws Exception {
        // TODO once BatchNorm setup for RNN, expand this test
        int nChannelsIn = 3;
        int inputSize = 10*10*nChannelsIn;	//10px x 10px x 3 channels
        int miniBatchSize = 4;
        int timeSeriesLength = 10;
        int nClasses = 3;

        Nd4j.getRandom().setSeed(12345);
        INDArray input = Nd4j.rand(new int[]{miniBatchSize,inputSize,timeSeriesLength});
        INDArray labels = Nd4j.zeros(miniBatchSize, nClasses, timeSeriesLength);
        Random r = new Random(12345);
        for( int i = 0; i < miniBatchSize; i++ ){
            for(int j = 0; j < timeSeriesLength; j++) {
                int idx = r.nextInt(nClasses);
                labels.putScalar(new int[]{i,idx,j}, 1.0);
            }
        }
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5).nIn(3).nOut(5).stride(1, 1)
                        .activation("identity").weightInit(WeightInit.XAVIER).updater(Updater.NONE).build())	//Out: (10-5)/1+1 = 6 -> 6x6x5
                .layer(1, new BatchNormalization.Builder().build())
                .layer(2, new ActivationLayer.Builder().activation("relu").build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2).stride(1, 1).build())	//Out: (6-2)/1+1 = 5 -> 5x5x5
                .layer(4, new DenseLayer.Builder().nIn(5 * 5 * 5).nOut(4)
                        .updater(Updater.NONE).weightInit(WeightInit.XAVIER).activation("relu")
                        .build())
                .layer(5, new GravesLSTM.Builder().nIn(4).nOut(3)
                        .activation("identity").updater(Updater.NONE).weightInit(WeightInit.XAVIER)
                        .build())
//                .layer(6, new BatchNormalization.Builder().build())
//                .layer(7, new ActivationLayer.Builder().activation("tanh").build())
                .layer(6, new RnnOutputLayer.Builder().nIn(3).nOut(nClasses)
                        .activation("softmax").lossFunction(LossFunctions.LossFunction.MCXENT)
                        .updater(Updater.NONE).build())
                .cnnInputSize(10, 10, 3)
                .pretrain(false).backprop(true)
                .build();

        conf.getInputPreProcessors().put(0,new RnnToCnnPreProcessor(10, 10, 3));

        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
        mln.init();
        mln.setInput(input);
        mln.setLabels(labels);
        mln.fit();

    }

    @Test
    public void testCNNBNActivationCombo() throws Exception {
        DataSetIterator iter = new MnistDataSetIterator(2, 2);
        DataSet next = iter.next();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(2)
                .seed(123)
                .list()
                .layer(0, new ConvolutionLayer.Builder().nIn(1).nOut(6).weightInit(WeightInit.XAVIER).activation("identity").build())
                .layer(1, new BatchNormalization.Builder().build())
                .layer(2, new ActivationLayer.Builder().activation("relu").build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .nOut(10).build())
                .backprop(true).pretrain(false)
                .cnnInputSize(28,28,1)
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        network.fit(next);
        
        assertNotEquals(null, network.getLayer(0).getParam("W"));
        assertNotEquals(null, network.getLayer(0).getParam("b"));

    }

    @Ignore@Test
    public void testMultiLSTMLayer() throws Exception {
        // TODO use this to test when batch norm implemented for RNN

        int inputSize = 10*10;
        int miniBatchSize = 4;
        int timeSeriesLength = 10;
        int nClasses = 3;

        Nd4j.getRandom().setSeed(12345);
        INDArray input = Nd4j.rand(new int[]{miniBatchSize,inputSize,timeSeriesLength});
        INDArray labels = Nd4j.zeros(miniBatchSize, nClasses, timeSeriesLength);
        Random r = new Random(12345);
        for( int i = 0; i < miniBatchSize; i++ ){
            for(int j = 0; j < timeSeriesLength; j++) {
                int idx = r.nextInt(nClasses);
                labels.putScalar(new int[]{i,idx,j}, 1.0);
            }
        }
        // Run without separate activation layer
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .list()
                .layer(0, new GravesLSTM.Builder().nIn(100).nOut(3)
                        .activation("tanh").updater(Updater.NONE).weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new RnnOutputLayer.Builder().nIn(3).nOut(nClasses)
                        .activation("softmax").lossFunction(LossFunctions.LossFunction.MCXENT)
                        .updater(Updater.NONE).build())
                .pretrain(false).backprop(true)
                .build();

        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
        mln.init();
        mln.setInput(input);
        mln.setLabels(labels);
        mln.fit();


        // Run with separate activation layer
        MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .list()
                .layer(0, new GravesLSTM.Builder().nIn(100).nOut(3)
                        .activation("identity").updater(Updater.NONE).weightInit(WeightInit.XAVIER)
                        .build())
//                .layer(, new BatchNormalization.Builder().build())
                .layer(1, new ActivationLayer.Builder().activation("tanh").build())
                .layer(2, new RnnOutputLayer.Builder().nIn(3).nOut(nClasses)
                        .activation("softmax").lossFunction(LossFunctions.LossFunction.MCXENT)
                        .updater(Updater.NONE).build())
                .pretrain(false).backprop(true)
                .build();

        conf2.getInputPreProcessors().put(1,new RnnToFeedForwardPreProcessor());
        conf2.getInputPreProcessors().put(2,new FeedForwardToRnnPreProcessor());

        MultiLayerNetwork mln2 = new MultiLayerNetwork(conf2);
        mln2.init();
        mln2.setInput(input);
        mln2.setLabels(labels);
        mln2.fit();

//        assertEquals(mln.getLayer(0).getParam("W"), mln2.getLayer(0).getParam("W"));
//        assertEquals(mln.getLayer(3).getParam("W"), mln2.getLayer(4).getParam("W"));
//        assertEquals(mln.getLayer(0).getParam("b"), mln2.getLayer(0).getParam("b"));
//        assertEquals(mln.getLayer(3).getParam("b"), mln2.getLayer(4).getParam("b"));
    }


    @Ignore@Test
    public void testMultiCNNLSTMLayer() throws Exception {
        // TODO use this to test when batch norm implemented for RNN
        int nChannelsIn = 3;
        int inputSize = 10*10*nChannelsIn;	//10px x 10px x 3 channels
        int miniBatchSize = 4;
        int timeSeriesLength = 10;
        int nClasses = 3;

        Nd4j.getRandom().setSeed(12345);
        INDArray input = Nd4j.rand(new int[]{miniBatchSize,inputSize,timeSeriesLength});
        INDArray labels = Nd4j.zeros(miniBatchSize, nClasses, timeSeriesLength);
        Random r = new Random(12345);
        for( int i = 0; i < miniBatchSize; i++ ){
            for(int j = 0; j < timeSeriesLength; j++) {
                int idx = r.nextInt(nClasses);
                labels.putScalar(new int[]{i,idx,j}, 1.0);
            }
        }
        // Run without separate activation layer
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5).nIn(3).nOut(5).stride(1, 1)
                        .activation("relu").weightInit(WeightInit.XAVIER).updater(Updater.NONE).build())	//Out: (10-5)/1+1 = 6 -> 6x6x5
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2).stride(1, 1).build())	//Out: (6-2)/1+1 = 5 -> 5x5x5
                .layer(2, new DenseLayer.Builder().nIn(5 * 5 * 5).nOut(4)
                        .updater(Updater.NONE).weightInit(WeightInit.XAVIER).activation("relu")
                        .build())
                .layer(3, new GravesLSTM.Builder().nIn(4).nOut(3)
                        .activation("tanh").updater(Updater.NONE).weightInit(WeightInit.XAVIER)
                        .build())
                .layer(4, new RnnOutputLayer.Builder().nIn(3).nOut(nClasses)
                        .activation("softmax").lossFunction(LossFunctions.LossFunction.MCXENT)
                        .updater(Updater.NONE).build())
                .cnnInputSize(10, 10, 3)
                .pretrain(false).backprop(true)
                .build();

        conf.getInputPreProcessors().put(0,new RnnToCnnPreProcessor(10, 10, 3));

        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
        mln.init();
        mln.setInput(input);
        mln.setLabels(labels);
        mln.fit();


        // Run with separate activation layer
        MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5).nIn(3).nOut(5).stride(1, 1)
                        .activation("identity").weightInit(WeightInit.XAVIER).updater(Updater.NONE).build())	//Out: (10-5)/1+1 = 6 -> 6x6x5
                .layer(1, new BatchNormalization.Builder().build())
                .layer(2, new ActivationLayer.Builder().activation("relu").build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2).stride(1, 1).build())	//Out: (6-2)/1+1 = 5 -> 5x5x5
                .layer(4, new DenseLayer.Builder().nIn(5 * 5 * 5).nOut(4)
                        .updater(Updater.NONE).weightInit(WeightInit.XAVIER).activation("relu")
                        .build())
                .layer(5, new GravesLSTM.Builder().nIn(4).nOut(3)
                        .activation("identity").updater(Updater.NONE).weightInit(WeightInit.XAVIER)
                        .build())
                .layer(6, new BatchNormalization.Builder().build())
                .layer(7, new ActivationLayer.Builder().activation("tanh").build())
                .layer(8, new RnnOutputLayer.Builder().nIn(3).nOut(nClasses)
                        .activation("softmax").lossFunction(LossFunctions.LossFunction.MCXENT)
                        .updater(Updater.NONE).build())
                .cnnInputSize(10, 10, 3)
                .pretrain(false).backprop(true)
                .build();

        conf2.getInputPreProcessors().put(0,new RnnToCnnPreProcessor(10, 10, 3));

        MultiLayerNetwork mln2 = new MultiLayerNetwork(conf2);
        mln2.init();
        mln2.setInput(input);
        mln2.setLabels(labels);
        mln2.fit();
//
//        assertEquals(mln.getLayer(0).getParam("W"), mln2.getLayer(0).getParam("W"));
//        assertEquals(mln.getLayer(3).getParam("W"), mln2.getLayer(4).getParam("W"));
//        assertEquals(mln.getLayer(0).getParam("b"), mln2.getLayer(0).getParam("b"));
//        assertEquals(mln.getLayer(3).getParam("b"), mln2.getLayer(4).getParam("b"));
    }

}

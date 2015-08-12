package org.deeplearning4j.gradientcheck;

import static org.junit.Assert.*;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class GradientCheckTests {

    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-2;

    static {
        Nd4j.dtype = DataBuffer.Type.DOUBLE;
        NDArrayFactory factory = Nd4j.factory();
        factory.setDType(DataBuffer.Type.DOUBLE);
    }

    @Test
    public void testGradientMLP2LayerIrisSimple(){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .activationFunction("tanh")
                .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0,1))
                .regularization(false)
                .seed(12345L)
                .list(2)
                .layer(0, new DenseLayer.Builder().nIn(4).nOut(3).build())
                .layer(1, new OutputLayer.Builder(LossFunction.MCXENT).activation("softmax").nIn(3).nOut(3).build())
                .build();

        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
        mln.init();

        for( int i=0; i<mln.getnLayers(); i++ ){
            System.out.println("Layer " + i + " # params: " + mln.getLayer(i).numParams());
        }

        DataSet ds = new IrisDataSetIterator(150,150).next();
        ds.normalizeZeroMeanZeroUnitVariance();
        INDArray input = ds.getFeatureMatrix();
        INDArray labels = ds.getLabels();

        boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels, false);

        assertTrue(gradOK);
    }

    @Test
    public void testGradientMLP2LayerIrisL2(){
        //As above, but with L2 regularization
        //Need to run gradient through updater, so that L2 can be applied

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .activationFunction("tanh")
                .weightInit(WeightInit.XAVIER).dist(new UniformDistribution(0,1))
                .regularization(true)
                .l2(0.3).l1(0.0).dropOut(0.0)
                .updater(Updater.NONE)
                .seed(12345L)
                .list(2)
                .layer(0, new DenseLayer.Builder().nIn(4).nOut(10).build())
                .layer(1, new OutputLayer.Builder(LossFunction.MCXENT).activation("softmax").nIn(10).nOut(3).build())
                .build();

        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
        mln.init();

        for( int i=0; i< mln.getnLayers(); i++ ){
            System.out.println("Layer " + i + " # params: " + mln.getLayer(i).numParams());
        }

        DataSet ds = new IrisDataSetIterator(150,150).next();
        ds.normalizeZeroMeanZeroUnitVariance();
        INDArray input = ds.getFeatureMatrix();
        INDArray labels = ds.getLabels();

        boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels, true);

        assertTrue(gradOK);
    }

    @Test
    public void testGradientMLP2LayerIrisL1(){
        //As above, but with L2 regularization
        //Need to run gradient through updater, so that L2 can be applied

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .activationFunction("tanh")
                .weightInit(WeightInit.XAVIER).dist(new UniformDistribution(0,1))
                .regularization(true)
                .l2(0.0).l1(0.1).dropOut(0.0)
                .updater(Updater.NONE)
                .seed(12345L)
                .list(2)
                .layer(0, new DenseLayer.Builder().nIn(4).nOut(10).build())
                .layer(1, new OutputLayer.Builder(LossFunction.MCXENT).activation("softmax").nIn(10).nOut(3).build())
                .build();

        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
        mln.init();

        for( int i = 0; i< mln.getnLayers(); i++ ){
            System.out.println("Layer " + i + " # params: " + mln.getLayer(i).numParams());
        }

        DataSet ds = new IrisDataSetIterator(150,150).next();
        ds.normalizeZeroMeanZeroUnitVariance();
        INDArray input = ds.getFeatureMatrix();
        INDArray labels = ds.getLabels();

        boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels, true);

        assertTrue(gradOK);
    }

    @Test
    public void testGradientMLP2LayerIrisL1L2(){
        //L1 and L2 regularization
        //As above, but with L2 regularization
        //Need to run gradient through updater, so that L2 can be applied

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .activationFunction("tanh")
                .weightInit(WeightInit.XAVIER).dist(new UniformDistribution(0,1))
                .regularization(true)
                .l2(0.1).l1(0.1).dropOut(0.0)
                .updater(Updater.NONE)
                .seed(12345L)
                .list(2)
                .layer(0, new DenseLayer.Builder().nIn(4).nOut(10).build())
                .layer(1, new OutputLayer.Builder(LossFunction.MCXENT).activation("softmax").nIn(10).nOut(3).build())
                .build();

        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
        mln.init();

        for( int i = 0; i< mln.getnLayers(); i++) {
            System.out.println("Layer " + i + " # params: " + mln.getLayer(i).numParams());
        }

        DataSet ds = new IrisDataSetIterator(150,150).next();
        ds.normalizeZeroMeanZeroUnitVariance();
        INDArray input = ds.getFeatureMatrix();
        INDArray labels = ds.getLabels();

        boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels, true);

        assertTrue(gradOK);
    }

}

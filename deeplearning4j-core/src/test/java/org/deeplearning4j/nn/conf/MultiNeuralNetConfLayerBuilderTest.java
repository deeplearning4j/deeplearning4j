package org.deeplearning4j.nn.conf;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.*;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.RBM.*;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer.PoolingType;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import static org.junit.Assert.*;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**
 * @author Jeffrey Tang.
 */
public class MultiNeuralNetConfLayerBuilderTest {
    int numIn = 10;
    int numOut = 5;
    double drop = 0.3;
    String act = "softmax";
    PoolingType poolType = PoolingType.MAX;
    int[] filterSize = new int[]{2, 2};
    int filterDepth = 6;
    int[] stride = new int[]{2, 2};
    HiddenUnit hidden = HiddenUnit.RECTIFIED;
    VisibleUnit visible = VisibleUnit.GAUSSIAN;
    int k  = 1;
    Convolution.Type convType = Convolution.Type.FULL;
    LossFunction loss = LossFunction.MCXENT;
    WeightInit weight = WeightInit.XAVIER;
    double corrupt = 0.4;
    double sparsity = 0.3;

    @Test
    public void testNeuralNetConfigAPI() {
        LossFunction newLoss = LossFunction.SQUARED_LOSS;
        int newNumIn = numIn + 1;
        int newNumOut = numOut + 1;
        WeightInit newWeight = WeightInit.UNIFORM;
        double newDrop = 0.5;
        int[] newFS = new int[]{3, 3};
        int newFD = 7;
        int[] newStride = new int[]{3, 3};
        Convolution.Type newConvType = Convolution.Type.SAME;
        PoolingType newPoolType = PoolingType.AVG;
        double newCorrupt = 0.5;
        double newSparsity = 0.5;
        HiddenUnit newHidden = HiddenUnit.BINARY;
        VisibleUnit newVisible = VisibleUnit.BINARY;

        MultiLayerConfiguration multiConf1 = new NeuralNetConfiguration.Builder()
                .list()
                .layer(0, new DenseLayer.Builder().nIn(newNumIn).nOut(newNumOut).activation(act).build())
                .layer(1, new DenseLayer.Builder().nIn(newNumIn + 1).nOut(newNumOut + 1).activation(act).build())
                .build();
        NeuralNetConfiguration firstLayer = multiConf1.getConf(0);
        NeuralNetConfiguration secondLayer = multiConf1.getConf(1);

        assertFalse(firstLayer.equals(secondLayer));
    }

    @Test
    public void testRbmSetup() throws Exception {
        MultiLayerConfiguration multiLayerConfiguration = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .seed(123)
                .iterations(5)
                .maxNumLineSearchIterations(10) // Magical Optimisation Stuff
                .regularization(true)
                .list()
                .layer(0, new RBM.Builder(RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN).nIn(784).nOut(1000).weightInit(WeightInit.XAVIER).activation("relu").build())
                .layer(1, new RBM.Builder(RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN).nIn(1000).nOut(500).weightInit(WeightInit.XAVIER).activation("relu").build())
                .layer(2, new RBM.Builder(RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN).nIn(500).nOut(250).weightInit(WeightInit.XAVIER).activation("relu").build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).weightInit(WeightInit.XAVIER).activation("softmax")
                        .nIn(250).nOut(10).build())
                        // Pretrain is unsupervised pretraining and finetuning on output layer
                        // Backward is full propagation on ALL layers.
                .pretrain(false).backprop(true)
                .build();
        MultiLayerNetwork network = new MultiLayerNetwork(multiLayerConfiguration);
        network.init();
        DataSet d = new MnistDataSetIterator(2,2).next();
        org.deeplearning4j.nn.api.Layer firstRbm = network.getLayer(0);
        org.deeplearning4j.nn.api.Layer secondRbm = network.getLayer(1);
        org.deeplearning4j.nn.api.Layer thirdRbm = network.getLayer(2);
        org.deeplearning4j.nn.api.Layer fourthRbm = network.getLayer(3);
        INDArray[] weightMatrices = new INDArray[] {
                firstRbm.getParam(DefaultParamInitializer.WEIGHT_KEY),
                secondRbm.getParam(DefaultParamInitializer.WEIGHT_KEY),
                thirdRbm.getParam(DefaultParamInitializer.WEIGHT_KEY),
                fourthRbm.getParam(DefaultParamInitializer.WEIGHT_KEY),

        };
        INDArray[] hiddenBiases = new INDArray[] {
                firstRbm.getParam(DefaultParamInitializer.BIAS_KEY),
                secondRbm.getParam(DefaultParamInitializer.BIAS_KEY),
                thirdRbm.getParam(DefaultParamInitializer.BIAS_KEY),
                fourthRbm.getParam(DefaultParamInitializer.BIAS_KEY),

        };


        int[][] shapeAssertions = new int[][]{
                {784,1000},
                {1000,500},
                {500,250},
                {250,10},
        };

        int[][] biasAssertions = new int[][] {
                {1,1000},
                {1,500},
                {1,250},
                {1,10},

        };

        for(int i = 0; i < shapeAssertions.length; i++) {
            assertArrayEquals(shapeAssertions[i],weightMatrices[i].shape());
            assertArrayEquals(biasAssertions[i],hiddenBiases[i].shape());
        }

        network.fit(d);


    }

}

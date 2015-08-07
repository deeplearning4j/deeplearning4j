package org.deeplearning4j.nn.conf;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.RBM.*;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer.PoolingType;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import static org.junit.Assert.*;

import org.nd4j.linalg.convolution.Convolution;
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
        String newAct = "rectify";
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
                .activationFunction(act)
                .nIn(numIn).nOut(numOut)
                .lossFunction(loss)
                .list(2)
                .layer(0, new DenseLayer.Builder().nIn(newNumIn).nOut(newNumOut).build())
                .layer(1, new DenseLayer.Builder().nIn(newNumIn + 1).nOut(newNumOut + 1).build())
                .build();
        NeuralNetConfiguration firstLayer = multiConf1.getConf(0);
        NeuralNetConfiguration secondLayer = multiConf1.getConf(1);

        assertFalse(firstLayer.equals(secondLayer));
    }

    @Test
    public void testRbmSetup() {
        MultiLayerConfiguration multiLayerConfiguration = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .seed(123)
                .iterations(5)
                .maxNumLineSearchIterations(10) // Magical Optimisation Stuff
                .activationFunction("relu")
                .k(1) // Annoying dl4j bug that is yet to be fixed.
                .weightInit(WeightInit.XAVIER)
                .constrainGradientToUnitNorm(true)
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .regularization(true)
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                .list(4)
                .layer(0, new RBM.Builder().nIn(784).nOut(1000).build())
                .layer(1, new RBM.Builder().nIn(1000).nOut(500).build())
                .layer(2, new RBM.Builder().nIn(500).nOut(250).build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation("softmax")
                        .nIn(250).nOut(numOut).build())
                        // Pretrain is unsupervised pretraining and finetuning on output layer
                        // Backward is full propagation on ALL layers.
                .pretrain(false).backprop(true)
                .build();
        MultiLayerNetwork network = new MultiLayerNetwork(multiLayerConfiguration);
        network.init();

    }

}

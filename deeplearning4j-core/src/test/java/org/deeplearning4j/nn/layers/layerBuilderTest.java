package org.deeplearning4j.nn.layers;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.AutoEncoder;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.layers.RBM.*;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer.poolingType;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import static org.junit.Assert.*;

import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**
 * @author Jeffrey Tang.
 */
public class layerBuilderTest {
    int numIn = 10;
    int numOut = 5;
    double drop = 0.3;
    String act = "softmax";
    poolingType poolType = poolingType.MAX;
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
    public void testLayerBuilderAPI() {
        // Make new Convolutional layer
        ConvolutionLayer conv = new ConvolutionLayer.Builder(filterSize, filterDepth, convType).activation(act).build();
        // Make new Subsampling layer
        SubsamplingLayer sample = new SubsamplingLayer.Builder(poolType, stride).build();
        // Make new RBM layer
        RBM rbm = new RBM.Builder(hidden, visible, k).nIn(numIn).nOut(numOut).build();
        // Make new Output layer
        OutputLayer out = new OutputLayer.Builder(loss).nIn(numIn).nOut(numOut).activation(act).dropOut(drop).build();

        // Test Output layer API
        assertTrue(out.getNIn() == numIn);
        assertTrue(out.getNOut() == numOut);
        assertTrue(out.getDropOut() == drop);
        assertTrue(out.getActivationFunction().equals(act));
        // Test Convolution layer API
        assertTrue(conv.getConvolutionType() == convType);
        assertTrue(conv.getFilterSize() == filterSize);
        assertTrue(conv.getFilterDepth() == filterDepth && conv.getActivationFunction().equals(act));
        // Test Subsampling layer API
        assertTrue(sample.getPoolingType() == poolType);
        assertTrue(sample.getStride() == stride);
        // Test RBM layer API
        assertTrue(rbm.getNIn() == numIn);
        assertTrue(rbm.getNOut() == numOut);
        assertTrue(rbm.getHiddenUnit() == hidden);
        assertTrue(rbm.getVisibleUnit() == visible);
    }

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
        SubsamplingLayer.poolingType newPoolType = poolingType.AVG;
        double newCorrupt = 0.5;
        double newSparsity = 0.5;
        HiddenUnit newHidden = HiddenUnit.BINARY;
        VisibleUnit newVisible = VisibleUnit.BINARY;

        // Dense Layer
        NeuralNetConfiguration denseConf = new NeuralNetConfiguration.Builder()
                .activationFunction(act)
                .nIn(numIn).nOut(numOut)
                .dropOut(drop)
                .weightInit(weight)
                .layer(new DenseLayer.Builder().nIn(newNumIn).nOut(newNumOut)
                        .weightInit(newWeight).activation(newAct).dropOut(newDrop).build())
                .build();

        Layer dense = LayerFactories.getFactory(denseConf.getLayer()).create(denseConf);

        assertTrue(dense.conf().getNIn() == newNumIn);
        assertTrue(dense.conf().getNOut() == newNumOut);
        assertTrue(dense.conf().getActivationFunction().equals(newAct));
        assertTrue(dense.conf().getWeightInit().equals(newWeight));

        // Output layer
        NeuralNetConfiguration outputConf = new NeuralNetConfiguration.Builder()
                .activationFunction(act)
                .nIn(numIn).nOut(numOut)
                .lossFunction(loss)
                .layer(new OutputLayer.Builder(newLoss).nIn(newNumIn).nOut(newNumOut).build())
                .build();

        Layer out = LayerFactories.getFactory(outputConf.getLayer()).create(outputConf);

        assertTrue(out.conf().getNIn() == newNumIn);
        assertTrue(out.conf().getNOut() == newNumOut);
        assertTrue(out.conf().getLossFunction().equals(newLoss));

        // Convolutional layer
        NeuralNetConfiguration convConf = new NeuralNetConfiguration.Builder()
                .activationFunction(act)
                .filterSize(filterSize)
                .filterDepth(filterDepth)
                .layer(new ConvolutionLayer.Builder(newFS, newFD, newConvType).activation(newAct).build())
                .build();

        Layer conv = LayerFactories.getFactory(convConf.getLayer()).create(convConf);

        assertTrue(conv.conf().getFilterSize() == newFS);
        assertTrue(conv.conf().getFilterDepth() == newFD);
        assertTrue(conv.conf().getConvolutionType().equals(newConvType));

        // Subsampling layer
        NeuralNetConfiguration poolConf = new NeuralNetConfiguration.Builder()
                .stride(stride)
                .poolingType(poolType)
                .layer(new SubsamplingLayer.Builder(newPoolType, newStride).build())
                .build();

        Layer pool = LayerFactories.getFactory(poolConf.getLayer()).create(poolConf);

        assertTrue(pool.conf().getPoolingType() == newPoolType);
        assertTrue(pool.conf().getStride() == newStride);

        // AutoEncoder layer
        NeuralNetConfiguration autoConf = new NeuralNetConfiguration.Builder()
                .nIn(numIn).nOut(numOut)
                .corruptionLevel(corrupt)
                .sparsity(sparsity)
                .activationFunction(act)
                .weightInit(weight)
                .layer(new AutoEncoder.Builder(newCorrupt, newSparsity).nIn(newNumIn).nOut(newNumOut)
                        .activation(newAct).weightInit(newWeight).build())
                .build();

        Layer auto = LayerFactories.getFactory(outputConf.getLayer()).create(autoConf);

        assertTrue(auto.conf().getNIn() == newNumIn);
        assertTrue(auto.conf().getNOut() == newNumOut);
        assertTrue(auto.conf().getCorruptionLevel() == newCorrupt);
        assertTrue(auto.conf().getSparsity() == newSparsity);
        assertTrue(auto.conf().getActivationFunction().equals(newAct));
        assertTrue(auto.conf().getWeightInit().equals(newWeight));

        // RBM layer
        NeuralNetConfiguration rbmConf = new NeuralNetConfiguration.Builder()
                .activationFunction(act)
                .nIn(numIn).nOut(numOut)
                .hiddenUnit(hidden)
                .visibleUnit(visible)
                .weightInit(weight)
                .layer(new RBM.Builder(newHidden, newVisible).nIn(newNumIn).nOut(newNumOut)
                        .weightInit(newWeight).build())
                .build();

        BaseLayer rbm = LayerFactories.getFactory(outputConf.getLayer()).create(rbmConf);

        assertTrue(rbm.conf.getNIn() == newNumIn);
        assertTrue(rbm.conf.getNOut() == newNumOut);
        assertTrue(rbm.conf.getHiddenUnit().equals(newHidden));
        assertTrue(rbm.conf.getVisibleUnit().equals(newVisible));
        assertTrue(rbm.conf.getWeightInit().equals(newWeight));
    }
}

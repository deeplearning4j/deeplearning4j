package org.deeplearning4j.nn.layers;

import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM.*;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer.PoolingType;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import static org.junit.Assert.*;

import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**
 * @author Jeffrey Tang.
 */
public class LayerBuilderTest {
    int numIn = 10;
    int numOut = 5;
    double drop = 0.3;
    String act = "softmax";
    PoolingType poolType = PoolingType.MAX;
    int[] kernelSize = new int[]{2, 2};
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
        ConvolutionLayer conv = new ConvolutionLayer.Builder(kernelSize,convType).activation(act).build();
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
         // Test Subsampling layer API
        assertTrue(sample.getPoolingType() == poolType);
        assertTrue(sample.getStride() == stride);
        // Test RBM layer API
        assertTrue(rbm.getNIn() == numIn);
        assertTrue(rbm.getNOut() == numOut);
        assertTrue(rbm.getHiddenUnit() == hidden);
        assertTrue(rbm.getVisibleUnit() == visible);
    }
}

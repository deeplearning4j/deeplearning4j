package org.deeplearning4j.nn.layers;

import junit.framework.Assert;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM.*;

import org.deeplearning4j.nn.conf.layers.SubsamplingLayer.poolingType;
import org.junit.Test;
import static org.junit.Assert.*;

import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.util.Arrays;

/**
 * Created by jeffreytang on 7/20/15.
 */
public class layerBuilderTest {

    @Test
    public void testAPI() {
        int numIn = 10;
        int numOut = 5;
        double drop = 0.3;
        String act = "softmax";
        poolingType poolType = poolingType.MAX;
        int[] filterSize = new int[]{2, 2};
        int[] stride = new int[]{2, 2};
        HiddenUnit hidden = HiddenUnit.RECTIFIED;
        VisibleUnit visible = VisibleUnit.GAUSSIAN;
        int k  = 1;
        int filterDepth = 6;
        Convolution.Type convType = Convolution.Type.FULL;

        ConvolutionLayer conv = new ConvolutionLayer.Builder(filterSize, filterDepth, convType).activation(act).build();

        SubsamplingLayer sample = new SubsamplingLayer.Builder(poolType, filterSize, stride).build();
        RBM rbm = new RBM.Builder(hidden, visible, k).nIn(numIn).nOut(numOut).build();
        OutputLayer out = new OutputLayer.Builder(LossFunction.MCXENT).nIn(numIn).nOut(numOut).activation(act).dropOut(drop).build();

        //Convolution layer API
        assertTrue(conv.getConvolutionType() == convType && conv.getFilterSize() == filterSize
                && conv.getFilterDepth() == filterDepth && conv.getActivationFunction().equals(act));
        //Pooling layer API
        assertTrue(sample.getPoolingType() == poolType && sample.getFilterSize() == filterSize
                && sample.getStride() == stride);
        //RBM layer API
        assertTrue(rbm.getNIn() == numIn && rbm.getNOut() == numOut && rbm.getHidden() == hidden
                && rbm.getVisible() == visible);
        //Output layer API
        assertTrue(out.getNIn() == numIn && out.getNOut() == numOut && out.getDropOut() == drop
                && out.getActivationFunction().equals(act));
    }
}

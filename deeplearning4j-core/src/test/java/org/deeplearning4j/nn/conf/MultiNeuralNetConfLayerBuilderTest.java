package org.deeplearning4j.nn.conf;

import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
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
public class MultiNeuralNetConfLayerBuilderTest {
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

        
    }
}

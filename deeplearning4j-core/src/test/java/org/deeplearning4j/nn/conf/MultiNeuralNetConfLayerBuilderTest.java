package org.deeplearning4j.nn.conf;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer.PoolingType;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertFalse;

/**
 * @author Jeffrey Tang.
 */
public class MultiNeuralNetConfLayerBuilderTest extends BaseDL4JTest {
    int numIn = 10;
    int numOut = 5;
    double drop = 0.3;
    Activation act = Activation.SOFTMAX;
    PoolingType poolType = PoolingType.MAX;
    int[] filterSize = new int[] {2, 2};
    int filterDepth = 6;
    int[] stride = new int[] {2, 2};
    int k = 1;
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
        int[] newFS = new int[] {3, 3};
        int newFD = 7;
        int[] newStride = new int[] {3, 3};
        Convolution.Type newConvType = Convolution.Type.SAME;
        PoolingType newPoolType = PoolingType.AVG;
        double newCorrupt = 0.5;
        double newSparsity = 0.5;

        MultiLayerConfiguration multiConf1 =
                        new NeuralNetConfiguration.Builder().list()
                                        .layer(0, new DenseLayer.Builder().nIn(newNumIn).nOut(newNumOut).activation(act)
                                                        .build())
                                        .layer(1, new DenseLayer.Builder().nIn(newNumIn + 1).nOut(newNumOut + 1)
                                                        .activation(act).build())
                                        .build();
        NeuralNetConfiguration firstLayer = multiConf1.getConf(0);
        NeuralNetConfiguration secondLayer = multiConf1.getConf(1);

        assertFalse(firstLayer.equals(secondLayer));
    }
}

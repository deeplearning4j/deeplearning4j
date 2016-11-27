package org.deeplearning4j.gradientcheck;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.util.Random;

import static junit.framework.TestCase.fail;
import static org.junit.Assert.assertTrue;

/**
 * @author Alex Black 14 Aug 2015
 */
public class VaeGradientCheckTests {

    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-8;

    static {
        //Force Nd4j initialization, then set data type to double:
        Nd4j.zeros(1);
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
    }

    @Test
    public void testVaeAsMLP() {

        //Post pre-training: a VAE can be used as a MLP, by taking the mean value from p(z|x) as the output
        //This gradient check tests this part

//        String[] activFns = {"identity","tanh"};    //activation functions such as relu and hardtanh: may randomly fail due to discontinuities
        String[] activFns = {"tanh"};    //activation functions such as relu and hardtanh: may randomly fail due to discontinuities

//        LossFunction[] lossFunctions = {LossFunction.MCXENT, LossFunction.MSE};
//        String[] outputActivations = {"softmax", "tanh"};    //i.e., lossFunctions[i] used with outputActivations[i] here

        LossFunction[] lossFunctions = {LossFunction.MSE};
        String[] outputActivations = {"tanh"};

        DataNormalization scaler = new NormalizerMinMaxScaler();
        DataSetIterator iter = new IrisDataSetIterator(150, 150);
        scaler.fit(iter);
        iter.setPreProcessor(scaler);
        DataSet ds = iter.next();

        INDArray input = ds.getFeatureMatrix();
        INDArray labels = ds.getLabels();

        for (String afn : activFns) {
            for (int i = 0; i < lossFunctions.length; i++) {
                LossFunction lf = lossFunctions[i];
                String outputActivation = outputActivations[i];

                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .regularization(false)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .learningRate(1.0)
                        .seed(12345L)
                        .list()
                        .layer(0, new VariationalAutoencoder.Builder()
                                .nIn(4).nOut(3)
                                .encoderLayerSizes(5)
                                .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                                .activation(afn)
                                .updater(Updater.SGD)
                                .build())
                        .layer(1, new OutputLayer.Builder(lf)
                                .activation(outputActivation)
                                .nIn(3).nOut(3)
                                .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                                .updater(Updater.SGD)
                                .build())
                        .pretrain(false).backprop(true)
                        .build();

                MultiLayerNetwork mln = new MultiLayerNetwork(conf);
                mln.init();

                if (PRINT_RESULTS) {
                    System.out.println("testVaeAsMLP() - activationFn=" + afn + ", lossFn=" + lf + ", outputActivation=" + outputActivation);
                    for (int j = 0; j < mln.getnLayers(); j++)
                        System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
                }

                boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR, DEFAULT_MIN_ABS_ERROR,
                        PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

                String msg = "testVaeAsMLP() - activationFn=" + afn + ", lossFn=" + lf + ", outputActivation=" + outputActivation;
                assertTrue(msg, gradOK);
            }
        }
    }

    @Test
    public void testVaeAsMLP_L1L2Simple() {
        fail();
    }
}

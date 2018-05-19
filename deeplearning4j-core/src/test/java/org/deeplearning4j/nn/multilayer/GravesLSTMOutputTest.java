package org.deeplearning4j.nn.multilayer;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.stepfunctions.NegativeDefaultStepFunction;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer.Type;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.util.FeatureUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

/**
 * Created by Kirill Lebedev (drlebedev.com) on 8/31/2015.
 */
@Slf4j
public class GravesLSTMOutputTest extends BaseDL4JTest {

    private static int nIn = 20;
    private static int layerSize = 15;
    private static int window = 300;
    private static INDArray data;
    private static Type type;

    @BeforeClass
    public static void setUp() {
        data = getData();
    }

    @Test
    public void testSameLabelsOutput() {
        MultiLayerNetwork network = new MultiLayerNetwork(getNetworkConf(false));
        network.init();
        network.setListeners(new ScoreIterationListener(1));
        for( int j=0; j<40; j++ ) {
            network.fit(reshapeInput(data.dup()), data.dup());
        }
        Evaluation ev = eval(network);
        Assert.assertTrue(ev.f1() > 0.90);
    }

    @Test
    public void testSameLabelsOutputWithTBPTT() {
        MultiLayerNetwork network = new MultiLayerNetwork(getNetworkConf(true));
        network.init();
        network.setListeners(new ScoreIterationListener(1));
        for (int i = 0; i < window / 100; i++) {
            INDArray d = data.get(NDArrayIndex.interval(100 * i, 100 * (i + 1)), NDArrayIndex.all());
            for( int j=0; j<40; j++ ) {
                network.fit(reshapeInput(d.dup()), reshapeInput(d.dup()));
            }
        }
        Evaluation ev = eval(network);
    }

    private Evaluation eval(MultiLayerNetwork network) {
        Evaluation ev = new Evaluation(nIn);
        INDArray predict = network.output(reshapeInput(data));
        ev.eval(data, predict);
        log.info(ev.stats());
        return ev;
    }

    private MultiLayerConfiguration getNetworkConf(boolean useTBPTT) {
        MultiLayerConfiguration.Builder builder =
                        new NeuralNetConfiguration.Builder()
                                        .updater(new AdaGrad(0.1)).l2(0.0025)
                                        .stepFunction(new NegativeDefaultStepFunction())
                                        .list()
                                        .layer(0, new GravesLSTM.Builder().weightInit(WeightInit.DISTRIBUTION)
                                                        .dist(new NormalDistribution(0.0, 0.01)).nIn(nIn)
                                                        .nOut(layerSize).activation(Activation.TANH).build())
                                        .layer(1, new RnnOutputLayer.Builder(
                                                        LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nIn(layerSize)
                                                                        .nOut(nIn).activation(Activation.SOFTMAX)
                                                                        .build())
                                        .pretrain(false);
        if (useTBPTT) {
            builder.backpropType(BackpropType.TruncatedBPTT);
            builder.tBPTTBackwardLength(window / 3);
            builder.tBPTTForwardLength(window / 3);
        }
        return builder.build();
    }

    private static INDArray getData() {
        Random r = new Random(1);
        int[] result = new int[window];
        for (int i = 0; i < window; i++) {
            result[i] = r.nextInt(nIn);
        }
        return FeatureUtil.toOutcomeMatrix(result, nIn);
    }

    private INDArray reshapeInput(INDArray inp) {
        val shape = inp.shape();
        int miniBatchSize = 1;
        INDArray reshaped = inp.reshape(miniBatchSize, shape[0] / miniBatchSize, shape[1]);
        return reshaped.permute(0, 2, 1);
    }
}

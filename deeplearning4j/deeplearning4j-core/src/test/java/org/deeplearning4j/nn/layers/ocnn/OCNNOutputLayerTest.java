package org.deeplearning4j.nn.layers.ocnn;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.gradientcheck.GradientCheckUtil;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.*;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;
import org.nd4j.linalg.util.ArrayUtil;

import java.io.File;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

public class OCNNOutputLayerTest {

    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-8;

    static {
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
    }


    @Test
    public void testLayer() {
        DataSetIterator dataSetIterator = getNormalizedIterator();
        boolean doLearningFirst = true;
        MultiLayerNetwork network = getGradientCheckNetwork(2);


        DataSet ds = dataSetIterator.next();
        INDArray arr = ds.getFeatureMatrix();
        network.setInput(arr);

        if (doLearningFirst) {
            //Run a number of iterations of learning
            network.setInput(arr);
            network.setListeners(new ScoreIterationListener(1));
            network.computeGradientAndScore();
            double scoreBefore = network.score();
            for (int j = 0; j < 10; j++)
                network.fit(ds);
            network.computeGradientAndScore();
            double scoreAfter = network.score();
            //Can't test in 'characteristic mode of operation' if not learning
            String msg = "testLayer() - score did not (sufficiently) decrease during learning - activationFn="
                    + "relu" + ", lossFn=" + "ocnn" + ", "  + "sigmoid"
                    + ", doLearningFirst=" + doLearningFirst + " (before=" + scoreBefore
                    + ", scoreAfter=" + scoreAfter + ")";
           // assertTrue(msg, scoreAfter <  scoreBefore);
        }

        if (PRINT_RESULTS) {
            System.out.println("testLayer() - activationFn=" + "relu" + ", lossFn="
                    + "ocnn"  + "sigmoid" + ", doLearningFirst="
                    + doLearningFirst);
            for (int j = 0; j < network.getnLayers(); j++)
                System.out.println("Layer " + j + " # params: " + network.getLayer(j).numParams());
        }

        boolean gradOK = GradientCheckUtil.checkGradients(network, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, ds.getFeatures(), ds.getLabels());

        String msg = "testLayer() - activationFn=" + "relu" + ", lossFn=" + "ocnn"
                + ",=" + "sigmoid" + ", doLearningFirst=" + doLearningFirst;
        assertTrue(msg, gradOK);



    }


    @Test
    public void testLabelProbabilities() {
        Nd4j.getRandom().setSeed(42);
        DataSetIterator dataSetIterator = getNormalizedIterator();
        MultiLayerNetwork network = getSingleLayer();
        DataSet next = dataSetIterator.next();
        DataSet filtered = next.filterBy(new int[]{0, 1});
        for (int i = 0; i < 4; i++) {
            network.setEpochCount(i);
            network.getLayerWiseConfigurations().setEpochCount(i);
            network.fit(filtered);
        }

        DataSet anomalies = next.filterBy(new int[] {2});
        INDArray output = network.labelProbabilities(anomalies.getFeatureMatrix());
        INDArray normalOutput = network.output(anomalies.getFeatureMatrix(),false);
        assertEquals(output.lt(0.0).sumNumber().doubleValue(),normalOutput.eq(0.0).sumNumber().doubleValue(),1e-1);

        System.out.println("Labels " + anomalies.getLabels());
        System.out.println("Anomaly output " + normalOutput);
        System.out.println(output);

        INDArray normalProbs = network.labelProbabilities(filtered.getFeatureMatrix());
        INDArray outputForNormalSamples = network.output(filtered.getFeatureMatrix(),false);
        System.out.println("Normal probabilities " + normalProbs);
        System.out.println("Normal raw output " + outputForNormalSamples);
    }


    public DataSetIterator getNormalizedIterator() {
        DataSetIterator dataSetIterator = new IrisDataSetIterator(150,150);
        NormalizerStandardize normalizerStandardize = new NormalizerStandardize();
        normalizerStandardize.fit(dataSetIterator);
        dataSetIterator.reset();
        dataSetIterator.setPreProcessor(normalizerStandardize);
        return dataSetIterator;
    }

    private MultiLayerNetwork getSingleLayer() {
        int numHidden = 2;

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .miniBatch(true)
                .updater(Nesterovs.builder()
                        .momentum(0.1)
                        .learningRateSchedule(new StepSchedule(
                                ScheduleType.EPOCH,
                                1e-2,
                                0.1,
                                20)).build())
                .list(new DenseLayer.Builder().activation(new ActivationReLU())
                                .nIn(4).nOut(2).build(),
                        new  org.deeplearning4j.nn.conf.ocnn.OCNNOutputLayer.Builder()
                                .nIn(2).activation(new ActivationSigmoid())
                                .nu(0.1)
                                .hiddenLayerSize(numHidden).build())
                .build();
        MultiLayerNetwork network = new MultiLayerNetwork(configuration);
        network.init();
        network.setListeners(new ScoreIterationListener(1));
        return network;
    }


    public MultiLayerNetwork getGradientCheckNetwork(int numHidden) {
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(42).updater(new NoOp()).miniBatch(false)
                .list(new DenseLayer.Builder().activation(new ActivationIdentity()).nIn(4).nOut(4).build(),
                        new  org.deeplearning4j.nn.conf.ocnn.OCNNOutputLayer.Builder().nIn(4)
                                .nu(0.002).activation(new ActivationSigmoid())
                                .hiddenLayerSize(numHidden).build())
                .build();
        MultiLayerNetwork network = new MultiLayerNetwork(configuration);
        network.init();
        return network;
    }
}

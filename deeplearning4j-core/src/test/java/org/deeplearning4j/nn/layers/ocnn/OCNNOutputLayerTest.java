package org.deeplearning4j.nn.layers.ocnn;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.gradientcheck.GradientCheckUtil;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.NoOp;
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
        DataSetIterator dataSetIterator = new IrisDataSetIterator(150,150);
        DataSet ds = dataSetIterator.next();
        NormalizerStandardize normalizerStandardize = new NormalizerStandardize();
        normalizerStandardize.fit(dataSetIterator);
        dataSetIterator.reset();
        dataSetIterator.setPreProcessor(normalizerStandardize);

        int numHidden = 2;
        int nIn = 4;
        boolean doLearningFirst = true;
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(42).updater(new NoOp()).miniBatch(false)
                .list(new  org.deeplearning4j.nn.conf.ocnn.OCNNOutputLayer.Builder().nIn(4)
                        .nu(0.002)
                        .nOut(2)
                        .hiddenLayerSize(numHidden).build())
                .build();
        MultiLayerNetwork network = new MultiLayerNetwork(configuration);
        network.init();

        int expectedLength = numHidden +  (numHidden * nIn) + 1;
        assertEquals(expectedLength,network.params().length());

        DataSet next = dataSetIterator.next();
        INDArray arr = next.getFeatureMatrix();
        normalizerStandardize.transform(arr);
        network.setInput(arr);
        ds = new DataSet(arr,next.getLabels());
        network.setLabels(next.getLabels());

        if (doLearningFirst) {
            //Run a number of iterations of learning
            network.setInput(arr);
            network.setListeners(new ScoreIterationListener(1));
            ds = new DataSet(arr,ds.getLabels());
            network.setLabels(ds.getLabels());
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
            assertTrue(msg, scoreAfter <  scoreBefore);
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
    public void testOutput() throws Exception {
        DataSetIterator dataSetIterator = new IrisDataSetIterator(150,150);
        DataSet ds = dataSetIterator.next();
        NormalizerStandardize normalizerStandardize = new NormalizerStandardize();
        normalizerStandardize.fit(dataSetIterator);
        dataSetIterator.reset();
        dataSetIterator.setPreProcessor(normalizerStandardize);

        int numHidden = 2;
        int nIn = 4;
        boolean doLearningFirst = true;
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(42).miniBatch(false)
                .list(new  org.deeplearning4j.nn.conf.ocnn.OCNNOutputLayer.Builder().nIn(4)
                        .nu(0.002)
                        .nOut(2)
                        .hiddenLayerSize(numHidden).build())
                .build();
        MultiLayerNetwork network = new MultiLayerNetwork(configuration);
        network.init();

        DataSet next = dataSetIterator.next();
        INDArray arr = next.getFeatureMatrix();
        normalizerStandardize.transform(arr);
        network.setInput(arr);
        ds = new DataSet(arr.getRows(0,101),next.getLabels());
        network.setLabels(next.getLabels());
        network.setListeners(new ScoreIterationListener(1));

        for (int j = 0; j < 1000; j++) {
            network.fit(ds);
        }

        INDArray output = network.output(arr.getRows(ArrayUtil.range(102,149)));
        System.out.println(output);
        File tmpFile = new File("tmp-ocnn-zip");
        tmpFile.deleteOnExit();
        ModelSerializer.writeModel(network,tmpFile,true);

        MultiLayerNetwork loaded = ModelSerializer.restoreMultiLayerNetwork(tmpFile);
        assertNotNull(loaded);
    }


}

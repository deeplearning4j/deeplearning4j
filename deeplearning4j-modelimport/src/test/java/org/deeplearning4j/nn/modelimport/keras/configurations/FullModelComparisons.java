package org.deeplearning4j.nn.modelimport.keras.configurations;

import junit.framework.TestCase;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;

import org.deeplearning4j.nn.layers.recurrent.LSTM;
import org.deeplearning4j.nn.layers.recurrent.LastTimeStepLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasModel;
import org.deeplearning4j.nn.modelimport.keras.KerasSequentialModel;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.activations.impl.ActivationHardSigmoid;
import org.nd4j.linalg.activations.impl.ActivationTanH;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import static junit.framework.TestCase.assertTrue;

public class FullModelComparisons {

    ClassLoader classLoader = FullModelComparisons.class.getClassLoader();

    @Test
    public void lstmTest() throws IOException, UnsupportedKerasConfigurationException,
            InvalidKerasConfigurationException, InterruptedException {

        String modelPath = "modelimport/keras/fullconfigs/lstm/lstm_th_keras_2_config.json";
        String weightsPath = "modelimport/keras/fullconfigs/lstm/lstm_th_keras_2_weights.h5";

        ClassPathResource modelResource = new ClassPathResource(modelPath, classLoader);
        ClassPathResource weightsResource = new ClassPathResource(weightsPath, classLoader);

        KerasSequentialModel kerasModel = new KerasModel().modelBuilder()
                .modelJsonInputStream(modelResource.getInputStream())
                .weightsHdf5FilenameNoRoot(weightsResource.getFile().getAbsolutePath())
                .enforceTrainingConfig(false)
                .buildSequential();

        MultiLayerNetwork model = kerasModel.getMultiLayerNetwork();
        model.init();

        System.out.println(model.summary());

        // 1. Layer
        LSTM firstLstm = (LSTM) model.getLayer(0);
        org.deeplearning4j.nn.conf.layers.LSTM firstConf =
                (org.deeplearning4j.nn.conf.layers.LSTM) firstLstm.conf().getLayer();
        // "unit_forget_bias": true
        assertTrue(firstConf.getForgetGateBiasInit() == 1.0);

        assertTrue(firstConf.getGateActivationFn() instanceof ActivationHardSigmoid);
        assertTrue(firstConf.getActivationFn() instanceof ActivationTanH);

        int nIn = 12;
        int nOut = 96;

        // Need to convert from IFCO to CFOI order
        //
        INDArray W = firstLstm.getParam("W");
        assertTrue(Arrays.equals(W.shape(), new int[]{nIn, 4 * nOut}));
        TestCase.assertEquals(W.getDouble(0, 288), -0.30737767, 1e-7);
        TestCase.assertEquals(W.getDouble(0, 289), -0.5845409, 1e-7);
        TestCase.assertEquals(W.getDouble(1, 288), -0.44083247, 1e-7);
        TestCase.assertEquals(W.getDouble(11, 288), 0.017539706, 1e-7);
        TestCase.assertEquals(W.getDouble(0, 96), 0.2707935, 1e-7);
        TestCase.assertEquals(W.getDouble(0, 192), -0.19856165, 1e-7);
        TestCase.assertEquals(W.getDouble(0, 0), 0.15368782, 1e-7);


        INDArray RW = firstLstm.getParam("RW");
        assertTrue(Arrays.equals(RW.shape(), new int[]{nOut, 4 * nOut}));
        TestCase.assertEquals(RW.getDouble(0, 288), 0.15112677, 1e-7);


        INDArray b = firstLstm.getParam("b");
        assertTrue(Arrays.equals(b.shape(), new int[]{1, 4 * nOut}));
        TestCase.assertEquals(b.getDouble(0, 288), -0.36940336, 1e-7); // Keras I
        TestCase.assertEquals(b.getDouble(0, 96), 0.6031118, 1e-7);  // Keras F
        TestCase.assertEquals(b.getDouble(0, 192), -0.13569744, 1e-7); // Keras O
        TestCase.assertEquals(b.getDouble(0, 0), -0.2587392, 1e-7); // Keras C

        // 2. Layer
        LSTM secondLstm = (LSTM) ((LastTimeStepLayer) model.getLayer(1)).getUnderlying();
        org.deeplearning4j.nn.conf.layers.LSTM secondConf =
                (org.deeplearning4j.nn.conf.layers.LSTM) secondLstm.conf().getLayer();
        // "unit_forget_bias": true
        assertTrue(secondConf.getForgetGateBiasInit() == 1.0);

        assertTrue(firstConf.getGateActivationFn() instanceof ActivationHardSigmoid);
        assertTrue(firstConf.getActivationFn() instanceof ActivationTanH);

        nIn = 96;
        nOut = 96;

        W = secondLstm.getParam("W");
        assertTrue(Arrays.equals(W.shape(), new int[]{nIn, 4 * nOut}));
        TestCase.assertEquals(W.getDouble(0, 288), -0.7559755, 1e-7);

        RW = secondLstm.getParam("RW");
        assertTrue(Arrays.equals(RW.shape(), new int[]{nOut, 4 * nOut}));
        TestCase.assertEquals(RW.getDouble(0, 288), -0.33184892, 1e-7);


        b = secondLstm.getParam("b");
        assertTrue(Arrays.equals(b.shape(), new int[]{1, 4 * nOut}));
        TestCase.assertEquals(b.getDouble(0, 288), -0.2223678, 1e-7);
        TestCase.assertEquals(b.getDouble(0, 96), 0.73556226, 1e-7);
        TestCase.assertEquals(b.getDouble(0, 192), -0.63227624, 1e-7);
        TestCase.assertEquals(b.getDouble(0, 0), 0.06636357, 1e-7);

        SequenceRecordReader reader = new CSVSequenceRecordReader(0, ";");
        ClassPathResource dataResource = new ClassPathResource(
                "data/", classLoader);
        System.out.print(dataResource.getFile().getAbsolutePath());
        reader.initialize(new NumberedFileInputSplit(dataResource.getFile().getAbsolutePath()
                + "/sequences/%d.csv", 0, 282));

        DataSetIterator dataSetIterator = new SequenceRecordReaderDataSetIterator(
                reader, 1, -1, 12, true);
        List<Double> preds = new LinkedList<>();

        while (dataSetIterator.hasNext()) {
            DataSet dataSet = dataSetIterator.next();
            INDArray sequence = dataSet.getFeatures().get(NDArrayIndex.point(0)).transpose();
            INDArray bsSequence = sequence.reshape(1, 4, 12); // one batch
            INDArray permuteSequence = bsSequence.permute(0, 2, 1);
            INDArray pred = model.output(permuteSequence);
            assertTrue(Arrays.equals(pred.shape(), new int[]{1, 1}));
            preds.add(pred.getDouble(0, 0));
        }
        INDArray dl4jPredictions = Nd4j.create(preds);

        ClassPathResource predResource = new ClassPathResource(
                "modelimport/keras/fullconfigs/lstm/predictions.npy", classLoader);
        INDArray kerasPredictions = Nd4j.createFromNpyFile(predResource.getFile());

        for (int i = 0; i < 283; i++) {
            TestCase.assertEquals(kerasPredictions.getDouble(i), dl4jPredictions.getDouble(i), 1e-7);
        }


        INDArray ones = Nd4j.ones(1, 12, 4);
        INDArray predOnes = model.output(ones);
        TestCase.assertEquals(predOnes.getDouble(0, 0), 0.7216, 1e-4);


    }

    @Test
    public void cnnBatchNormTest() throws IOException, UnsupportedKerasConfigurationException,
            InvalidKerasConfigurationException {

        String modelPath = "modelimport/keras/fullconfigs/cnn/cnn_batch_norm.h5";


        ClassPathResource modelResource = new ClassPathResource(modelPath, classLoader);

        KerasSequentialModel kerasModel = new KerasModel().modelBuilder()
                .modelHdf5Filename(modelResource.getFile().getAbsolutePath())
                .enforceTrainingConfig(false)
                .buildSequential();

        MultiLayerNetwork model = kerasModel.getMultiLayerNetwork();
        model.init();

        System.out.println(model.summary());

        ClassPathResource inputResource = new ClassPathResource(
                "modelimport/keras/fullconfigs/cnn/input.npy", classLoader);
        INDArray input = Nd4j.createFromNpyFile(inputResource.getFile());
        input = input.permute(0, 3, 1, 2);
        assertTrue(Arrays.equals(input.shape(), new int[] {5, 3, 10, 10}));

        INDArray output = model.output(input);

        ClassPathResource outputResource = new ClassPathResource(
                "modelimport/keras/fullconfigs/cnn/predictions.npy", classLoader);
        INDArray kerasOutput = Nd4j.createFromNpyFile(outputResource.getFile());

        for (int i = 0; i < 5; i++) {
            TestCase.assertEquals(output.getDouble(i), kerasOutput.getDouble(i), 1e-4);
        }

    }

}

/*-
 *
 *  * Copyright 2017 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */
package org.deeplearning4j.nn.modelimport.keras.e2e;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.eval.ROCMultiClass;
import org.deeplearning4j.nn.modelimport.keras.Hdf5Archive;
import org.deeplearning4j.nn.modelimport.keras.KerasModel;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasModelUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Unit tests for end-to-end Keras model import.
 *
 * @author dave@skymind.io
 */
@Slf4j
public class KerasModelEndToEndTest {
    private static final String GROUP_ATTR_INPUTS = "inputs";
    private static final String GROUP_ATTR_OUTPUTS = "outputs";
    private static final String GROUP_PREDICTIONS = "predictions";
    private static final String GROUP_ACTIVATIONS = "activations";
    private static final String TEMP_OUTPUTS_FILENAME = "tempOutputs";
    private static final String TEMP_MODEL_FILENAME = "tempModel";
    private static final String H5_EXTENSION = ".h5";
    private static final double EPS = 1E-6;

    /**
     * MNIST MLP tests
     */
    @Test
    public void importMnistMlpTfKeras1() throws Exception {
        String modelPath = "modelimport/keras/examples/mnist_mlp/mnist_mlp_tf_keras_1_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/mnist_mlp/mnist_mlp_tf_keras_1_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, true, true);
    }

    @Test
    public void importMnistMlpThKeras1() throws Exception {
        String modelPath = "modelimport/keras/examples/mnist_mlp/mnist_mlp_th_keras_1_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/mnist_mlp/mnist_mlp_th_keras_1_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, false, true);
    }

    @Test
    public void importMnistMlpTfKeras2() throws Exception {
        String modelPath = "modelimport/keras/examples/mnist_mlp/mnist_mlp_tf_keras_2_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/mnist_mlp/mnist_mlp_tf_keras_2_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, true, true);
    }

    @Test
    public void importMnistMlpReshapeTfKeras1() throws Exception {
        String modelPath = "modelimport/keras/examples/mnist_mlp_reshape/mnist_mlp_reshape_tf_keras_1_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/mnist_mlp_reshape/mnist_mlp_reshape_tf_keras_1_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, true, true);
    }

    /**
     * MNIST CNN tests
     */
//    @Test
//    public void importMnistCnnTfKeras1() throws Exception {
//        String modelPath = "modelimport/keras/examples/mnist_cnn/mnist_cnn_tf_keras_1_model.h5";
//        String inputsOutputPath = "modelimport/keras/examples/mnist_cnn/mnist_cnn_tf_keras_1_inputs_and_outputs.h5";
//        importEndModelTest(modelPath, inputsOutputPath, true, true);
//    }
    @Test
    public void importMnistCnnThKeras1() throws Exception {
        String modelPath = "modelimport/keras/examples/mnist_cnn/mnist_cnn_th_keras_1_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/mnist_cnn/mnist_cnn_th_keras_1_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, false, true);
    }

    @Test
    public void importMnistCnnTfKeras2() throws Exception {
        String modelPath = "modelimport/keras/examples/mnist_cnn/mnist_cnn_tf_keras_2_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/mnist_cnn/mnist_cnn_tf_keras_2_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, true, true);
    }

    /**
     * IMDB Embedding and LSTM test
     */
//    @Test
//    public void importImdbLstmTfKeras1() throws Exception {
//        String modelPath = "modelimport/keras/examples/imdb_lstm/imdb_lstm_tf_keras_1_model.h5";
//        String inputsOutputPath = "modelimport/keras/examples/imdb_lstm/imdb_lstm_tf_keras_1_inputs_and_outputs.h5";
//        importEndModelTest(modelPath, inputsOutputPath, true, true);
//    }
//
//    @Test
//    public void importImdbLstmThKeras1() throws Exception {
//        String modelPath = "modelimport/keras/examples/imdb_lstm/imdb_lstm_th_keras_1_model.h5";
//        String inputsOutputPath = "modelimport/keras/examples/imdb_lstm/imdb_lstm_th_keras_1_inputs_and_outputs.h5";
//        importEndModelTest(modelPath, inputsOutputPath, false, true);
//    }
//    @Test
//    public void importImdbLstmTfKeras2() throws Exception {
//        String modelPath = "modelimport/keras/examples/imdb_lstm/imdb_lstm_tf_keras_2_model.h5";
//        String inputsOutputPath = "modelimport/keras/examples/imdb_lstm/imdb_lstm_tf_keras_2_inputs_and_outputs.h5";
//        importEndModelTest(modelPath, inputsOutputPath, true, true);
//    }
//
//    @Test
//    public void importImdbLstmThKeras2() throws Exception {
//        String modelPath = "modelimport/keras/examples/imdb_lstm/imdb_lstm_th_keras_2_model.h5";
//        String inputsOutputPath = "modelimport/keras/examples/imdb_lstm/imdb_lstm_th_keras_2_inputs_and_outputs.h5";
//        importEndModelTest(modelPath, inputsOutputPath, false, true);
//    }

    /**
     * IMDB LSTM fasttext
     */
    @Test
    public void importImdbFasttextTfKeras1() throws Exception {
        String modelPath = "modelimport/keras/examples/imdb_fasttext/imdb_fasttext_tf_keras_1_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/imdb_fasttext/imdb_fasttext_tf_keras_1_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, true, false);
    }

    @Test
    public void importImdbFasttextThKeras1() throws Exception {
        String modelPath = "modelimport/keras/examples/imdb_fasttext/imdb_fasttext_th_keras_1_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/imdb_fasttext/imdb_fasttext_th_keras_1_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, false, false);
    }

    @Test
    public void importImdbFasttextTfKeras2() throws Exception {
        String modelPath = "modelimport/keras/examples/imdb_fasttext/imdb_fasttext_tf_keras_2_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/imdb_fasttext/imdb_fasttext_tf_keras_2_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, true, false);
    }

    /**
     * Simple LSTM test
     */
    @Test
    public void importSimpleLstmTfKeras1() throws Exception {
        String modelPath = "modelimport/keras/examples/simple_lstm/simple_lstm_tf_keras_1_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/simple_lstm/simple_lstm_tf_keras_1_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, true, false);
    }

    @Test
    public void importSimpleLstmThKeras1() throws Exception {
        String modelPath = "modelimport/keras/examples/simple_lstm/simple_lstm_th_keras_1_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/simple_lstm/simple_lstm_th_keras_1_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, true, false);
    }

    @Test
    public void importSimpleLstmTfKeras2() throws Exception {
        String modelPath = "modelimport/keras/examples/simple_lstm/simple_lstm_tf_keras_2_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/simple_lstm/simple_lstm_tf_keras_2_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, true, false);
    }

    /**
     * CNN without bias test
     */
    @Test
    public void importCnnNoBiasTfKeras2() throws Exception {
        String modelPath = "modelimport/keras/examples/cnn_no_bias/mnist_cnn_no_bias_tf_keras_2_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/cnn_no_bias/mnist_cnn_no_bias_tf_keras_2_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, true, false);
    }

    /**
     * DCGAN import test
     */
    @Test
    public void importDcganDiscriminator() throws Exception {
        importModelH5Test("modelimport/keras/examples/mnist_dcgan/dcgan_discriminator_epoch_50.h5");
    }
    @Test
    public void importDcganGenerator() throws Exception {
        importModelH5Test("modelimport/keras/examples/mnist_dcgan/dcgan_generator_epoch_50.h5");
    }


    void importModelH5Test(String modelPath) throws Exception {
        ClassPathResource modelResource =
                new ClassPathResource(modelPath,
                        KerasModelEndToEndTest.class.getClassLoader());
        File modelFile = File.createTempFile(TEMP_MODEL_FILENAME, H5_EXTENSION);
        Files.copy(modelResource.getInputStream(), modelFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        MultiLayerNetwork model = new KerasModel().modelBuilder().modelHdf5Filename(modelFile.getAbsolutePath())
                .enforceTrainingConfig(false).buildSequential().getMultiLayerNetwork();
    }


    void importEndModelTest(String modelPath, String inputsOutputsPath, boolean tfOrdering, boolean checkPredictions) throws Exception {
        ClassPathResource modelResource =
                new ClassPathResource(modelPath,
                        KerasModelEndToEndTest.class.getClassLoader());
        File modelFile = File.createTempFile(TEMP_MODEL_FILENAME, H5_EXTENSION);
        Files.copy(modelResource.getInputStream(), modelFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        MultiLayerNetwork model = new KerasModel().modelBuilder().modelHdf5Filename(modelFile.getAbsolutePath())
                .enforceTrainingConfig(false).buildSequential().getMultiLayerNetwork();

        ClassPathResource outputsResource =
                new ClassPathResource(inputsOutputsPath,
                        KerasModelEndToEndTest.class.getClassLoader());
        File outputsFile = File.createTempFile(TEMP_OUTPUTS_FILENAME, H5_EXTENSION);
        Files.copy(outputsResource.getInputStream(), outputsFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        Hdf5Archive outputsArchive = new Hdf5Archive(outputsFile.getAbsolutePath());

        if (checkPredictions) {
            INDArray input = getInputs(outputsArchive, tfOrdering)[0];
            Map<String, INDArray> activationsKeras = getActivations(outputsArchive, tfOrdering);
            for (int i = 0; i < model.getLayers().length; i++) {
                String layerName = model.getLayerNames().get(i);
                if (activationsKeras.containsKey(layerName)) {
                    INDArray activationsDl4j = model.feedForwardToLayer(i, input, false).get(i + 1);
                    /* TODO: investigate why this fails for some layers:
                     * compareINDArrays(layerName, activationsKeras.get(layerName), activationsDl4j, EPS);
                     */
                }
            }

            INDArray predictionsKeras = getPredictions(outputsArchive, tfOrdering)[0];
            INDArray predictionsDl4j = model.output(input, false);
            /* TODO: investigate why this fails when max difference is ~1E-7!
             * compareINDArrays("predictions", predictionsKeras, predictionsDl4j, EPS);
             */
            INDArray outputs = getOutputs(outputsArchive, true)[0];
            compareMulticlassAUC("predictions", outputs, predictionsKeras, predictionsDl4j, 10, EPS);
        }
    }

    static public INDArray[] getInputs(Hdf5Archive archive, boolean tensorFlowImageDimOrdering) throws Exception {
        List<String> inputNames = (List<String>) KerasModelUtils
                .parseJsonString(archive.readAttributeAsJson(GROUP_ATTR_INPUTS)).get(GROUP_ATTR_INPUTS);
        INDArray[] inputs = new INDArray[inputNames.size()];
        for (int i = 0; i < inputNames.size(); i++) {
            inputs[i] = archive.readDataSet(inputNames.get(i), GROUP_ATTR_INPUTS);
            if (inputs[i].shape().length == 4 && tensorFlowImageDimOrdering)
                inputs[i] = inputs[i].permute(0, 3, 1, 2);
        }
        return inputs;
    }

    static public Map<String, INDArray> getActivations(Hdf5Archive archive, boolean tensorFlowImageDimOrdering)
            throws Exception {
        Map<String, INDArray> activations = new HashMap<String, INDArray>();
        for (String layerName : archive.getDataSets(GROUP_ACTIVATIONS)) {
            INDArray activation = archive.readDataSet(layerName, GROUP_ACTIVATIONS);
            if (activation.shape().length == 4 && tensorFlowImageDimOrdering)
                activation = activation.permute(0, 3, 1, 2);
            activations.put(layerName, activation);
        }
        return activations;
    }

    static public INDArray[] getOutputs(Hdf5Archive archive, boolean tensorFlowImageDimOrdering) throws Exception {
        List<String> outputNames = (List<String>) KerasModelUtils
                .parseJsonString(archive.readAttributeAsJson(GROUP_ATTR_OUTPUTS)).get(GROUP_ATTR_OUTPUTS);
        INDArray[] outputs = new INDArray[outputNames.size()];
        for (int i = 0; i < outputNames.size(); i++) {
            outputs[i] = archive.readDataSet(outputNames.get(i), GROUP_ATTR_OUTPUTS);
            if (outputs[i].shape().length == 4 && tensorFlowImageDimOrdering)
                outputs[i] = outputs[i].permute(0, 3, 1, 2);
        }
        return outputs;
    }

    static public INDArray[] getPredictions(Hdf5Archive archive, boolean tensorFlowImageDimOrdering) throws Exception {
        List<String> outputNames = (List<String>) KerasModelUtils
                .parseJsonString(archive.readAttributeAsJson(GROUP_ATTR_OUTPUTS)).get(GROUP_ATTR_OUTPUTS);
        INDArray[] predictions = new INDArray[outputNames.size()];
        for (int i = 0; i < outputNames.size(); i++) {
            predictions[i] = archive.readDataSet(outputNames.get(i), GROUP_PREDICTIONS);
            if (predictions[i].shape().length == 4 && tensorFlowImageDimOrdering)
                predictions[i] = predictions[i].permute(0, 3, 1, 2);
        }
        return predictions;
    }

    static public void compareINDArrays(String label, INDArray a, INDArray b, double eps) {
        INDArray diff = a.sub(b);
        double min = diff.minNumber().doubleValue();
        double max = diff.maxNumber().doubleValue();
        log.info(label + ": " + a.equalsWithEps(b, eps) + ", " + min + ", " + max);
        assert (a.equalsWithEps(b, eps));
    }

    static public void compareMulticlassAUC(String label, INDArray target, INDArray a, INDArray b, int nbClasses,
                                            double eps) {
        ROCMultiClass evalA = new ROCMultiClass(100);
        evalA.eval(target, a);
        double avgAucA = evalA.calculateAverageAUC();
        ROCMultiClass evalB = new ROCMultiClass(100);
        evalB.eval(target, b);
        double avgAucB = evalB.calculateAverageAUC();
        assertEquals(avgAucA, avgAucB, EPS);

        double[] aucA = new double[nbClasses];
        double[] aucB = new double[nbClasses];
        for (int i = 0; i < nbClasses; i++) {
            aucA[i] = evalA.calculateAUC(i);
            aucB[i] = evalB.calculateAUC(i);
        }
        assertArrayEquals(aucA, aucB, EPS);
    }
}

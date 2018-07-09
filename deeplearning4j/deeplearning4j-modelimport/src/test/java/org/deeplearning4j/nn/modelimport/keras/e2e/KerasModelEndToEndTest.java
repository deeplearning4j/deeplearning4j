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
import lombok.val;
import org.deeplearning4j.eval.ROCMultiClass;
import org.deeplearning4j.gradientcheck.GradientCheckUtil;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.conf.layers.CnnLossLayer;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.recurrent.LSTM;
import org.deeplearning4j.nn.layers.recurrent.LastTimeStepLayer;
import org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer;
import org.deeplearning4j.nn.modelimport.keras.Hdf5Archive;
import org.deeplearning4j.nn.modelimport.keras.KerasModel;
import org.deeplearning4j.nn.modelimport.keras.KerasSequentialModel;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasModelBuilder;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasModelUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.junit.Ignore;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.*;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Unit tests for end-to-end Keras model import.
 *
 * @author dave@skymind.io, Max Pumperla
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
    private static final double EPS = 1E-5;

    private static final boolean SKIP_GRAD_CHECKS = false;

    @Rule
    public final TemporaryFolder testDir = new TemporaryFolder();

    @Test(expected = FileNotFoundException.class)
    public void fileNotFoundEndToEnd() throws Exception {
        String modelPath = "modelimport/keras/examples/foo/bar.h5";
        importEndModelTest(modelPath, null, true, true, false);
    }

    /**
     * MNIST MLP tests
     */
    @Test
    public void importMnistMlpTfKeras1() throws Exception {
        String modelPath = "modelimport/keras/examples/mnist_mlp/mnist_mlp_tf_keras_1_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/mnist_mlp/mnist_mlp_tf_keras_1_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, true, true, false);
    }

    @Test
    public void importMnistMlpThKeras1() throws Exception {
        String modelPath = "modelimport/keras/examples/mnist_mlp/mnist_mlp_th_keras_1_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/mnist_mlp/mnist_mlp_th_keras_1_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, false, true, false);
    }

    @Test
    public void importMnistMlpTfKeras2() throws Exception {
        String modelPath = "modelimport/keras/examples/mnist_mlp/mnist_mlp_tf_keras_2_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/mnist_mlp/mnist_mlp_tf_keras_2_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, true, true, false);
    }

    @Test
    public void importMnistMlpReshapeTfKeras1() throws Exception {
        String modelPath = "modelimport/keras/examples/mnist_mlp_reshape/mnist_mlp_reshape_tf_keras_1_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/mnist_mlp_reshape/mnist_mlp_reshape_tf_keras_1_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, true, true, true);
    }

    /**
     * MNIST CNN tests
     */
    @Test
    public void importMnistCnnTfKeras1() throws Exception {
        String modelPath = "modelimport/keras/examples/mnist_cnn/mnist_cnn_tf_keras_1_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/mnist_cnn/mnist_cnn_tf_keras_1_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, true, false, false);
    }

    @Test
    public void importMnistCnnThKeras1() throws Exception {
        String modelPath = "modelimport/keras/examples/mnist_cnn/mnist_cnn_th_keras_1_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/mnist_cnn/mnist_cnn_th_keras_1_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, false, true, true);
    }

    @Test
    public void importMnistCnnTfKeras2() throws Exception {
        String modelPath = "modelimport/keras/examples/mnist_cnn/mnist_cnn_tf_keras_2_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/mnist_cnn/mnist_cnn_tf_keras_2_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, true, true, true);
    }

    /**
     * IMDB Embedding and LSTM test
     */
    @Test
    public void importImdbLstmTfKeras1() throws Exception {
        String modelPath = "modelimport/keras/examples/imdb_lstm/imdb_lstm_tf_keras_1_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/imdb_lstm/imdb_lstm_tf_keras_1_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, true, true, false);
    }

    @Test
    public void importImdbLstmThKeras1() throws Exception {
        String modelPath = "modelimport/keras/examples/imdb_lstm/imdb_lstm_th_keras_1_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/imdb_lstm/imdb_lstm_th_keras_1_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, true, true, false);
    }

    @Test
    public void importImdbLstmTfKeras2() throws Exception {
        String modelPath = "modelimport/keras/examples/imdb_lstm/imdb_lstm_tf_keras_2_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/imdb_lstm/imdb_lstm_tf_keras_2_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, true, true, false);
    }

    @Test
    public void importImdbLstmThKeras2() throws Exception {
        String modelPath = "modelimport/keras/examples/imdb_lstm/imdb_lstm_th_keras_2_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/imdb_lstm/imdb_lstm_th_keras_2_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, false, true, false);
    }

    /**
     * IMDB LSTM fasttext
     */
    // TODO: prediction checks fail due to globalpooling for fasttext, very few grads fail as well
    @Test
    public void importImdbFasttextTfKeras1() throws Exception {
        String modelPath = "modelimport/keras/examples/imdb_fasttext/imdb_fasttext_tf_keras_1_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/imdb_fasttext/imdb_fasttext_tf_keras_1_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, false, false, false);
    }

    @Test
    public void importImdbFasttextThKeras1() throws Exception {
        String modelPath = "modelimport/keras/examples/imdb_fasttext/imdb_fasttext_th_keras_1_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/imdb_fasttext/imdb_fasttext_th_keras_1_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, false, false, false);
    }

    @Test
    public void importImdbFasttextTfKeras2() throws Exception {
        String modelPath = "modelimport/keras/examples/imdb_fasttext/imdb_fasttext_tf_keras_2_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/imdb_fasttext/imdb_fasttext_tf_keras_2_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, true, false, false);
    }

    /**
     * Simple LSTM (return sequences = false) into Dense layer test
     */
    @Test
    public void importSimpleLstmTfKeras1() throws Exception {
        String modelPath = "modelimport/keras/examples/simple_lstm/simple_lstm_tf_keras_1_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/simple_lstm/simple_lstm_tf_keras_1_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, true, true, false);
    }

    @Test
    public void importSimpleLstmThKeras1() throws Exception {
        String modelPath = "modelimport/keras/examples/simple_lstm/simple_lstm_th_keras_1_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/simple_lstm/simple_lstm_th_keras_1_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, true, true, false);
    }

    @Test
    public void importSimpleLstmTfKeras2() throws Exception {
        String modelPath = "modelimport/keras/examples/simple_lstm/simple_lstm_tf_keras_2_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/simple_lstm/simple_lstm_tf_keras_2_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, true, false, false);
    }


    /**
     * Simple LSTM (return sequences = true) into flatten into Dense layer test
     */
    @Test
    public void importSimpleFlattenLstmTfKeras2() throws Exception {
        String modelPath = "modelimport/keras/examples/simple_flatten_lstm/simple_flatten_lstm_tf_keras_2_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/simple_flatten_lstm/" +
                "simple_flatten_lstm_tf_keras_2_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, true, true, false);
    }

    /**
     * Simple RNN (return sequences = true) into flatten into Dense layer test
     */
    @Test
    public void importSimpleFlattenRnnTfKeras2() throws Exception {
        String modelPath = "modelimport/keras/examples/simple_flatten_rnn/simple_flatten_rnn_tf_keras_2_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/simple_flatten_rnn/" +
                "simple_flatten_rnn_tf_keras_2_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, true, true, false);
    }

    /**
     * Simple RNN (return sequences = false) into Dense layer test
     */
    @Test
    public void importSimpleRnnTfKeras2() throws Exception {
        String modelPath = "modelimport/keras/examples/simple_rnn/simple_rnn_tf_keras_2_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/simple_rnn/" +
                "simple_rnn_tf_keras_2_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, true, true, false);
    }

    /**
     * CNN without bias test
     */
    @Test
    public void importCnnNoBiasTfKeras2() throws Exception {
        String modelPath = "modelimport/keras/examples/cnn_no_bias/mnist_cnn_no_bias_tf_keras_2_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/cnn_no_bias/mnist_cnn_no_bias_tf_keras_2_inputs_and_outputs.h5";
        importEndModelTest(modelPath, inputsOutputPath, true, false, true);
    }

    /**
     * GAN import tests
     */
    @Test
    public void importDcganMnistDiscriminator() throws Exception {
        importSequentialModelH5Test("modelimport/keras/examples/mnist_dcgan/dcgan_discriminator_epoch_50.h5");
    }

    @Test
    public void importDcganMnistGenerator() throws Exception {
        importSequentialModelH5Test("modelimport/keras/examples/mnist_dcgan/dcgan_generator_epoch_50.h5");
    }

    /**
     * Deep convolutional GAN import test
     */
    @Test
    public void importDcganDiscriminator() throws Exception {
        importSequentialModelH5Test("modelimport/keras/examples/gans/dcgan_discriminator.h5");
    }

    @Test
    public void importDcganGenerator() throws Exception {
        importSequentialModelH5Test("modelimport/keras/examples/gans/dcgan_generator.h5");
    }

    /**
     * Wasserstein GAN import test
     */
    @Test
    public void importWganDiscriminator() throws Exception {
        for (int i = 0; i < 100; i++) {
            // run a few times to make sure HDF5 doesn't crash
            importSequentialModelH5Test("modelimport/keras/examples/gans/wgan_discriminator.h5");
        }
    }

    @Test
    public void importWganGenerator() throws Exception {
        importSequentialModelH5Test("modelimport/keras/examples/gans/wgan_generator.h5");
    }

    @Test
    public void importCnn1d() throws Exception {
        importSequentialModelH5Test("modelimport/keras/examples/cnn1d/cnn1d_flatten_tf_keras2.h5");
    }

    /**
     * DGA classifier test
     */
    @Test
    public void importDgaClassifier() throws Exception {
        importSequentialModelH5Test("modelimport/keras/examples/dga_classifier/keras2_dga_classifier_tf_model.h5");
    }

    /**
     * Reshape flat input into 3D to fit into an LSTM model
     */
    @Test
    public void importFlatIntoLSTM() throws Exception {
        importFunctionalModelH5Test("modelimport/keras/examples/reshape_to_rnn/reshape_model.h5");
    }

    /**
     * Functional LSTM test
     */
    @Test
    public void importFunctionalLstmTfKeras2() throws Exception {
        String modelPath = "modelimport/keras/examples/functional_lstm/lstm_functional_tf_keras_2.h5";

        // No training enabled
        ComputationGraph graphNoTrain = importFunctionalModelH5Test(modelPath, null, false);
        System.out.println(graphNoTrain.summary());

        // Training enabled
        ComputationGraph graph = importFunctionalModelH5Test(modelPath, null, true);
        System.out.println(graph.summary());

        // Make predictions
        int miniBatch = 32;
        INDArray input = Nd4j.ones(miniBatch, 4, 10);
        INDArray[] out = graph.output(input);

        // Fit model
        graph.fit(new INDArray[]{input}, out);
    }

    /**
     * U-Net
     */
    @Test
    public void importUnetTfKeras2() throws Exception {
        importFunctionalModelH5Test(
                "modelimport/keras/examples/unet/unet_keras_2_tf.h5", null, true);
    }

    /**
     * ResNet50
     */
    @Test
    public void importResnet50() throws Exception {
        importFunctionalModelH5Test("modelimport/keras/examples/resnet/resnet50_weights_tf_dim_ordering_tf_kernels.h5");
    }

    /**
     * DenseNet
     */
    @Test
    public void importDenseNet() throws Exception {
        importFunctionalModelH5Test("modelimport/keras/examples/densenet/densenet121_tf_keras_2.h5");
    }

    /**
     * SqueezeNet
     */
    @Test
    public void importSqueezeNet() throws Exception {
        importFunctionalModelH5Test("modelimport/keras/examples/squeezenet/squeezenet.h5");
    }


    /**
     * MobileNet
     */
    @Test
    public void importMobileNet() throws Exception {
        importFunctionalModelH5Test("modelimport/keras/examples/mobilenet/mobilenet_tf_keras_2.h5");
    }

    /**
     * InceptionV3 Keras 2 no top
     */
    @Test
    public void importInceptionKeras2() throws Exception {
        int[] inputShape = new int[]{299, 299, 3};
        ComputationGraph graph = importFunctionalModelH5Test(
                "modelimport/keras/examples/inception/inception_tf_keras_2.h5", inputShape, false);
        INDArray input = Nd4j.ones(10, 3, 299, 299);
        graph.output(input);
        System.out.println(graph.summary());
    }

    /**
     * InceptionV3
     */
    @Test
    @Ignore
    // Takes unreasonably long, but works
    public void importInception() throws Exception {
        ComputationGraph graph = importFunctionalModelH5Test(
                "modelimport/keras/examples/inception/inception_v3_complete.h5");
        INDArray input = Nd4j.ones(10, 3, 299, 299);
        graph.output(input);
        System.out.println(graph.summary());
    }

    /**
     * Xception
     */
    @Test
    public void importXception() throws Exception {
        int[] inputShape = new int[]{299, 299, 3};
        ComputationGraph graph = importFunctionalModelH5Test(
                "modelimport/keras/examples/xception/xception_tf_keras_2.h5", inputShape, false);
    }

    /**
     * Seq2seq model
     */
    @Test
    @Ignore // does not work yet, needs DL4J enhancements
    public void importSeq2Seq() throws Exception {
        importFunctionalModelH5Test("modelimport/keras/examples/seq2seq/full_model_seq2seq_5549.h5");

    }


    private ComputationGraph importFunctionalModelH5Test(String modelPath) throws Exception {
        return importFunctionalModelH5Test(modelPath, null, false);
    }

    private ComputationGraph importFunctionalModelH5Test(String modelPath, int[] inputShape, boolean train) throws Exception {
        ClassPathResource modelResource =
                new ClassPathResource(modelPath,
                        KerasModelEndToEndTest.class.getClassLoader());
        File modelFile = createTempFile(TEMP_MODEL_FILENAME, H5_EXTENSION);
        Files.copy(modelResource.getInputStream(), modelFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        KerasModelBuilder builder = new KerasModel().modelBuilder().modelHdf5Filename(modelFile.getAbsolutePath())
                .enforceTrainingConfig(train);
        if (inputShape != null) {
            builder.inputShape(inputShape);
        }
        KerasModel model = builder.buildModel();
        return model.getComputationGraph();
    }

    private void importSequentialModelH5Test(String modelPath) throws Exception {
        importSequentialModelH5Test(modelPath, null);
    }


    private void importSequentialModelH5Test(String modelPath, int[] inputShape) throws Exception {
        ClassPathResource modelResource =
                new ClassPathResource(modelPath,
                        KerasModelEndToEndTest.class.getClassLoader());
        File modelFile = createTempFile(TEMP_MODEL_FILENAME, H5_EXTENSION);
        Files.copy(modelResource.getInputStream(), modelFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        KerasModelBuilder builder = new KerasModel().modelBuilder().modelHdf5Filename(modelFile.getAbsolutePath())
                .enforceTrainingConfig(false);
        if (inputShape != null) {
            builder.inputShape(inputShape);
        }
        KerasSequentialModel model = builder.buildSequential();
        model.getMultiLayerNetwork();
    }


    private void importEndModelTest(String modelPath, String inputsOutputsPath, boolean tfOrdering, boolean checkPredictions) throws Exception {
        importEndModelTest(modelPath, inputsOutputsPath, tfOrdering, checkPredictions, false);
    }

    private void importEndModelTest(String modelPath, String inputsOutputsPath, boolean tfOrdering, boolean checkPredictions,
                                    boolean checkGradients) throws Exception {
        ClassPathResource modelResource =
                new ClassPathResource(modelPath,
                        KerasModelEndToEndTest.class.getClassLoader());
        File modelFile = createTempFile(TEMP_MODEL_FILENAME, H5_EXTENSION);
        Files.copy(modelResource.getInputStream(), modelFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        KerasSequentialModel kerasModel = new KerasModel().modelBuilder().modelHdf5Filename(modelFile.getAbsolutePath())
                .enforceTrainingConfig(false).buildSequential();

        MultiLayerNetwork model = kerasModel.getMultiLayerNetwork();

        ClassPathResource outputsResource =
                new ClassPathResource(inputsOutputsPath,
                        KerasModelEndToEndTest.class.getClassLoader());
        File outputsFile = createTempFile(TEMP_OUTPUTS_FILENAME, H5_EXTENSION);
        Files.copy(outputsResource.getInputStream(), outputsFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        try (Hdf5Archive outputsArchive = new Hdf5Archive(outputsFile.getAbsolutePath())) {

            if (checkPredictions) {
                INDArray input = getInputs(outputsArchive, tfOrdering)[0];
                Map<String, INDArray> activationsKeras = getActivations(outputsArchive, tfOrdering);
                for (int i = 0; i < model.getLayers().length; i++) {
                    String layerName = model.getLayerNames().get(i);
                    if (activationsKeras.containsKey(layerName)) {
                        INDArray activationsDl4j = model.feedForwardToLayer(i, input, false).get(i + 1);
                        if (activationsDl4j.shape().length == 3)
                            activationsDl4j = activationsDl4j.permute(0, 2, 1);
                        compareINDArrays(layerName, activationsKeras.get(layerName), activationsDl4j, EPS);

                    }
                }

                INDArray predictionsKeras = getPredictions(outputsArchive, tfOrdering)[0];
                INDArray predictionsDl4j = model.output(input, false);
                compareINDArrays("predictions", predictionsKeras, predictionsDl4j, EPS);
                INDArray outputs = getOutputs(outputsArchive, true)[0];

                if (outputs.shape()[0] == 1) {
                    outputs = outputs.reshape(outputs.shape()[1], outputs.shape()[0]);
                }

                // FIXME: int cast
                val nOut = (int) outputs.shape()[outputs.shape().length - 1];
                compareMulticlassAUC("predictions", outputs, predictionsKeras, predictionsDl4j, nOut, EPS);
            }

            if (checkGradients && ! SKIP_GRAD_CHECKS) {
                Random r = new Random(12345);
                INDArray input = getInputs(outputsArchive, tfOrdering)[0];
                INDArray predictionsDl4j = model.output(input, false);

                //Infer one-hot labels... this probably won't work for all
                INDArray testLabels = Nd4j.create(predictionsDl4j.shape());
                if (testLabels.rank() == 2) {
                    for (int i = 0; i < testLabels.size(0); i++) {
                        // FIXME: int cast
                        testLabels.putScalar(i, r.nextInt((int) testLabels.size(1)), 1.0);
                    }
                } else if (testLabels.rank() == 3) {
                    for (int i = 0; i < testLabels.size(0); i++) {
                        for (int j = 0; j < testLabels.size(1); j++) {
                            // FIXME: int cast
                            testLabels.putScalar(i, j, r.nextInt((int) testLabels.size(1)), 1.0);
                        }
                    }
                } else {
                    throw new RuntimeException("Cannot gradient check 4d output array");
                }
                checkGradients(model, input, testLabels);
            }
        }
    }

    private static INDArray[] getInputs(Hdf5Archive archive, boolean tensorFlowImageDimOrdering) throws Exception {
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

    private static Map<String, INDArray> getActivations(Hdf5Archive archive, boolean tensorFlowImageDimOrdering)
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

    private static INDArray[] getOutputs(Hdf5Archive archive, boolean tensorFlowImageDimOrdering) throws
            Exception {
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

    private static INDArray[] getPredictions(Hdf5Archive archive, boolean tensorFlowImageDimOrdering)
            throws Exception {
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

    private static void compareINDArrays(String label, INDArray a, INDArray b, double eps) {
        INDArray diff = a.sub(b);
        double min = diff.minNumber().doubleValue();
        double max = diff.maxNumber().doubleValue();
        log.info(label + ": " + a.equalsWithEps(b, eps) + ", " + min + ", " + max);
        double threshold = 1e-7;
        double aAbsMax = Math.max(Math.abs(a.minNumber().doubleValue()), Math.abs(a.maxNumber().doubleValue()));
        double bAbsMax = Math.max(Math.abs(b.minNumber().doubleValue()), Math.abs(b.maxNumber().doubleValue()));

        // skip too small absolute inputs
        if (Math.abs(aAbsMax) > threshold && Math.abs(bAbsMax) > threshold) {
            assertTrue(a.equalsWithEps(b, eps));
        }

    }

    private static void compareMulticlassAUC(String label, INDArray target, INDArray a, INDArray b, int nbClasses,
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
        if (nbClasses > 1) {
            for (int i = 0; i < nbClasses; i++) {
                aucA[i] = evalA.calculateAUC(i);
                aucB[i] = evalB.calculateAUC(i);
            }
            assertArrayEquals(aucA, aucB, EPS);
        }
    }

    public static void checkGradients(MultiLayerNetwork net, INDArray input, INDArray labels) {
        double eps = 1e-6;
        double max_rel_error = 1e-3;
        double min_abs_error = 1e-8;

        MultiLayerNetwork netToTest;
        if (net.getOutputLayer() instanceof IOutputLayer) {
            netToTest = net;
        } else {
            org.deeplearning4j.nn.conf.layers.Layer l;
            if (labels.rank() == 2) {
                l = new LossLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .build();
            } else {
                //Rank 3
                l = new RnnOutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(labels.size(1))
                        .nOut(labels.size(1))
                        .build();
            }
            netToTest = new TransferLearning.Builder(net)
                    .fineTuneConfiguration(new FineTuneConfiguration.Builder()
                            .updater(new NoOp())
                            .dropOut(0.0)
                            .build())
                    .addLayer(l)
                    .build();
        }

        log.info("Num params: " + net.numParams());

        for (Layer l : netToTest.getLayers()) {
            // Remove any dropout manually - until this is fixed:
            // https://github.com/deeplearning4j/deeplearning4j/issues/4368
             l.conf().getLayer().setIDropout(null);

            //Also swap out activation functions... this is a bit of a hack, but should make the net gradient checkable...
            if (l.conf().getLayer() instanceof FeedForwardLayer) {
                FeedForwardLayer ffl = (FeedForwardLayer) l.conf().getLayer();
                IActivation activation = ffl.getActivationFn();
                if (activation instanceof ActivationReLU || activation instanceof ActivationLReLU) {
                    ffl.setActivationFn(new ActivationSoftPlus());
                } else if (activation instanceof ActivationHardTanH) {
                    ffl.setActivationFn(new ActivationTanH());
                }
            }
        }

        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
        boolean passed = GradientCheckUtil.checkGradients(netToTest, eps, max_rel_error, min_abs_error, true, false,
                input, labels, null, null, true, 9);
        assertTrue("Gradient check failed", passed);
    }

    private File createTempFile(String prefix, String suffix) throws IOException {
        return testDir.newFile(prefix + "-" + System.nanoTime() + suffix);
    }
}

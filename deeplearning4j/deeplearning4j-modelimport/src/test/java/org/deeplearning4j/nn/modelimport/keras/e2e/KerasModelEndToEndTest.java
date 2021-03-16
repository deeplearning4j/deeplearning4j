/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.deeplearning4j.nn.modelimport.keras.e2e;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.eval.ROCMultiClass;
import org.deeplearning4j.gradientcheck.GradientCheckUtil;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.layers.Convolution1DLayer;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.modelimport.keras.Hdf5Archive;
import org.deeplearning4j.nn.modelimport.keras.KerasModel;
import org.deeplearning4j.nn.modelimport.keras.KerasSequentialModel;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasModelBuilder;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasModelUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.junit.jupiter.api.Disabled;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.*;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.function.BiFunction;
import org.nd4j.common.function.Function;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.impl.LossSparseMCXENT;
import org.nd4j.common.resources.Resources;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.DisplayName;
import java.nio.file.Path;
import static org.junit.jupiter.api.Assertions.assertThrows;
import org.junit.jupiter.api.extension.ExtendWith;

/**
 * Unit tests for end-to-end Keras model import.
 *
 * @author dave@skymind.io, Max Pumperla
 */
@Slf4j
@DisplayName("Keras Model End To End Test")
class KerasModelEndToEndTest extends BaseDL4JTest {

    private static final String GROUP_ATTR_INPUTS = "inputs";

    private static final String GROUP_ATTR_OUTPUTS = "outputs";

    private static final String GROUP_PREDICTIONS = "predictions";

    private static final String GROUP_ACTIVATIONS = "activations";

    private static final String TEMP_OUTPUTS_FILENAME = "tempOutputs";

    private static final String TEMP_MODEL_FILENAME = "tempModel";

    private static final String H5_EXTENSION = ".h5";

    private static final double EPS = 1E-5;

    private static final boolean SKIP_GRAD_CHECKS = true;



    @Override
    public long getTimeoutMilliseconds() {
        // Most benchmarks should run very quickly; large timeout is to avoid issues with unusually slow download of test resources
        return 900000000L;
    }

    @Test
    @DisplayName("File Not Found End To End")
    void fileNotFoundEndToEnd(@TempDir Path tempDir) {
        assertThrows(IllegalStateException.class, () -> {
            String modelPath = "modelimport/keras/examples/foo/bar.h5";
            importEndModelTest(tempDir,modelPath, null, true, true, false, false);
        });
    }

    /**
     * MNIST MLP tests
     */
    @Test
    @DisplayName("Import Mnist Mlp Tf Keras 1")
    void importMnistMlpTfKeras1(@TempDir Path tempDir) throws Exception {
        String modelPath = "modelimport/keras/examples/mnist_mlp/mnist_mlp_tf_keras_1_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/mnist_mlp/mnist_mlp_tf_keras_1_inputs_and_outputs.h5";
        importEndModelTest(tempDir,modelPath, inputsOutputPath, true, true, false, false);
    }

    @Test
    @DisplayName("Import Mnist Mlp Th Keras 1")
    void importMnistMlpThKeras1(@TempDir Path tempDir) throws Exception {
        String modelPath = "modelimport/keras/examples/mnist_mlp/mnist_mlp_th_keras_1_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/mnist_mlp/mnist_mlp_th_keras_1_inputs_and_outputs.h5";
        importEndModelTest(tempDir,modelPath, inputsOutputPath, false, true, false, false);
    }

    @Test
    @DisplayName("Import Mnist Mlp Tf Keras 2")
    void importMnistMlpTfKeras2(@TempDir Path tempDir) throws Exception {
        String modelPath = "modelimport/keras/examples/mnist_mlp/mnist_mlp_tf_keras_2_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/mnist_mlp/mnist_mlp_tf_keras_2_inputs_and_outputs.h5";
        importEndModelTest(tempDir,modelPath, inputsOutputPath, true, true, false, false);
    }

    @Test
    @DisplayName("Import Mnist Mlp Reshape Tf Keras 1")
    void importMnistMlpReshapeTfKeras1(@TempDir Path tempDir) throws Exception {
        String modelPath = "modelimport/keras/examples/mnist_mlp_reshape/mnist_mlp_reshape_tf_keras_1_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/mnist_mlp_reshape/mnist_mlp_reshape_tf_keras_1_inputs_and_outputs.h5";
        importEndModelTest(tempDir,modelPath, inputsOutputPath, true, true, true, false);
    }

    /**
     * MNIST CNN tests
     */
    @Test
    @DisplayName("Import Mnist Cnn Tf Keras 1")
    void importMnistCnnTfKeras1(@TempDir Path tempDir) throws Exception {
        String modelPath = "modelimport/keras/examples/mnist_cnn/mnist_cnn_tf_keras_1_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/mnist_cnn/mnist_cnn_tf_keras_1_inputs_and_outputs.h5";
        importEndModelTest(tempDir,modelPath, inputsOutputPath, true, false, false, false);
    }

    @Test
    @DisplayName("Import Mnist Cnn Th Keras 1")
    void importMnistCnnThKeras1(@TempDir Path tempDir) throws Exception {
        String modelPath = "modelimport/keras/examples/mnist_cnn/mnist_cnn_th_keras_1_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/mnist_cnn/mnist_cnn_th_keras_1_inputs_and_outputs.h5";
        importEndModelTest(tempDir,modelPath, inputsOutputPath, false, true, true, false);
    }

    @Test
    @DisplayName("Import Mnist Cnn Tf Keras 2")
    void importMnistCnnTfKeras2(@TempDir Path tempDir) throws Exception {
        String modelPath = "modelimport/keras/examples/mnist_cnn/mnist_cnn_tf_keras_2_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/mnist_cnn/mnist_cnn_tf_keras_2_inputs_and_outputs.h5";
        importEndModelTest(tempDir,modelPath, inputsOutputPath, true, true, true, false);
    }

    /**
     * IMDB Embedding and LSTM test
     */
    @Test
    @DisplayName("Import Imdb Lstm Tf Keras 1")
    void importImdbLstmTfKeras1(@TempDir Path tempDir) throws Exception {
        String modelPath = "modelimport/keras/examples/imdb_lstm/imdb_lstm_tf_keras_1_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/imdb_lstm/imdb_lstm_tf_keras_1_inputs_and_outputs.h5";
        importEndModelTest(tempDir,modelPath, inputsOutputPath, true, true, false, false, true, null, null);
    }

    @Test
    @DisplayName("Import Imdb Lstm Th Keras 1")
    void importImdbLstmThKeras1(@TempDir Path tempDir) throws Exception {
        String modelPath = "modelimport/keras/examples/imdb_lstm/imdb_lstm_th_keras_1_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/imdb_lstm/imdb_lstm_th_keras_1_inputs_and_outputs.h5";
        importEndModelTest(tempDir,modelPath, inputsOutputPath, true, true, false, false, true, null, null);
    }

    @Test
    @DisplayName("Import Imdb Lstm Tf Keras 2")
    void importImdbLstmTfKeras2(@TempDir Path tempDir) throws Exception {
        String modelPath = "modelimport/keras/examples/imdb_lstm/imdb_lstm_tf_keras_2_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/imdb_lstm/imdb_lstm_tf_keras_2_inputs_and_outputs.h5";
        importEndModelTest(tempDir,modelPath, inputsOutputPath, true, true, false, false, true, null, null);
    }

    @Test
    @DisplayName("Import Imdb Lstm Th Keras 2")
    void importImdbLstmThKeras2(@TempDir Path tempDir) throws Exception {
        String modelPath = "modelimport/keras/examples/imdb_lstm/imdb_lstm_th_keras_2_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/imdb_lstm/imdb_lstm_th_keras_2_inputs_and_outputs.h5";
        importEndModelTest(tempDir,modelPath, inputsOutputPath, false, true, false, false, true, null, null);
    }

    /**
     * IMDB LSTM fasttext
     */
    // TODO: prediction checks fail due to globalpooling for fasttext, very few grads fail as well
    @Test
    @DisplayName("Import Imdb Fasttext Tf Keras 1")
    void importImdbFasttextTfKeras1(@TempDir Path tempDir) throws Exception {
        String modelPath = "modelimport/keras/examples/imdb_fasttext/imdb_fasttext_tf_keras_1_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/imdb_fasttext/imdb_fasttext_tf_keras_1_inputs_and_outputs.h5";
        importEndModelTest(tempDir,modelPath, inputsOutputPath, false, false, false, false);
    }

    @Test
    @DisplayName("Import Imdb Fasttext Th Keras 1")
    void importImdbFasttextThKeras1(@TempDir Path tempDir) throws Exception {
        String modelPath = "modelimport/keras/examples/imdb_fasttext/imdb_fasttext_th_keras_1_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/imdb_fasttext/imdb_fasttext_th_keras_1_inputs_and_outputs.h5";
        importEndModelTest(tempDir,modelPath, inputsOutputPath, false, false, false, false);
    }

    @Test
    @DisplayName("Import Imdb Fasttext Tf Keras 2")
    void importImdbFasttextTfKeras2(@TempDir Path tempDir) throws Exception {
        String modelPath = "modelimport/keras/examples/imdb_fasttext/imdb_fasttext_tf_keras_2_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/imdb_fasttext/imdb_fasttext_tf_keras_2_inputs_and_outputs.h5";
        importEndModelTest(tempDir,modelPath, inputsOutputPath, true, false, false, false);
    }

    /**
     * Simple LSTM (return sequences = false) into Dense layer test
     */
    @Test
    @DisplayName("Import Simple Lstm Tf Keras 1")
    void importSimpleLstmTfKeras1(@TempDir Path tempDir) throws Exception {
        String modelPath = "modelimport/keras/examples/simple_lstm/simple_lstm_tf_keras_1_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/simple_lstm/simple_lstm_tf_keras_1_inputs_and_outputs.h5";
        importEndModelTest(tempDir,modelPath, inputsOutputPath, true, true, false, false);
    }

    @Test
    @DisplayName("Import Simple Lstm Th Keras 1")
    void importSimpleLstmThKeras1(@TempDir Path tempDir) throws Exception {
        String modelPath = "modelimport/keras/examples/simple_lstm/simple_lstm_th_keras_1_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/simple_lstm/simple_lstm_th_keras_1_inputs_and_outputs.h5";
        importEndModelTest(tempDir,modelPath, inputsOutputPath, true, true, false, false);
    }

    @Test
    @DisplayName("Import Simple Lstm Tf Keras 2")
    void importSimpleLstmTfKeras2(@TempDir Path tempDir) throws Exception {
        String modelPath = "modelimport/keras/examples/simple_lstm/simple_lstm_tf_keras_2_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/simple_lstm/simple_lstm_tf_keras_2_inputs_and_outputs.h5";
        importEndModelTest(tempDir,modelPath, inputsOutputPath, true, false, false, false);
    }

    /**
     * Simple LSTM (return sequences = true) into flatten into Dense layer test
     */
    @Test
    @DisplayName("Import Simple Flatten Lstm Tf Keras 2")
    void importSimpleFlattenLstmTfKeras2(@TempDir Path tempDir) throws Exception {
        String modelPath = "modelimport/keras/examples/simple_flatten_lstm/simple_flatten_lstm_tf_keras_2_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/simple_flatten_lstm/" + "simple_flatten_lstm_tf_keras_2_inputs_and_outputs.h5";
        importEndModelTest(tempDir,modelPath, inputsOutputPath, true, true, false, false);
    }

    /**
     * Simple RNN (return sequences = true) into flatten into Dense layer test
     */
    @Test
    @DisplayName("Import Simple Flatten Rnn Tf Keras 2")
    void importSimpleFlattenRnnTfKeras2(@TempDir Path tempDir) throws Exception {
        String modelPath = "modelimport/keras/examples/simple_flatten_rnn/simple_flatten_rnn_tf_keras_2_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/simple_flatten_rnn/" + "simple_flatten_rnn_tf_keras_2_inputs_and_outputs.h5";
        importEndModelTest(tempDir,modelPath, inputsOutputPath, true, true, false, false, true, null, null);
    }

    /**
     * Simple RNN (return sequences = false) into Dense layer test
     */
    @Test
    @DisplayName("Import Simple Rnn Tf Keras 2")
    void importSimpleRnnTfKeras2(@TempDir Path tempDir) throws Exception {
        String modelPath = "modelimport/keras/examples/simple_rnn/simple_rnn_tf_keras_2_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/simple_rnn/" + "simple_rnn_tf_keras_2_inputs_and_outputs.h5";
        importEndModelTest(tempDir,modelPath, inputsOutputPath, true, true, false, false);
    }

    /**
     * CNN without bias test
     */
    @Test
    @DisplayName("Import Cnn No Bias Tf Keras 2")
    void importCnnNoBiasTfKeras2(@TempDir Path tempDir) throws Exception {
        String modelPath = "modelimport/keras/examples/cnn_no_bias/mnist_cnn_no_bias_tf_keras_2_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/cnn_no_bias/mnist_cnn_no_bias_tf_keras_2_inputs_and_outputs.h5";
        importEndModelTest(tempDir,modelPath, inputsOutputPath, true, true, true, false);
    }

    @Test
    @DisplayName("Import Sparse Xent")
    void importSparseXent(@TempDir Path tempDir) throws Exception {
        String modelPath = "modelimport/keras/examples/simple_sparse_xent/simple_sparse_xent_mlp_keras_2_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/simple_sparse_xent/simple_sparse_xent_mlp_keras_2_inputs_and_outputs.h5";
        MultiLayerNetwork net = importEndModelTest(tempDir,modelPath, inputsOutputPath, true, true, true, true);
        Layer outLayer = net.getOutputLayer();
        assertTrue(outLayer instanceof org.deeplearning4j.nn.layers.LossLayer);
        LossLayer llConf = (LossLayer) outLayer.getConfig();
        assertEquals(new LossSparseMCXENT(), llConf.getLossFn());
    }

    /**
     * GAN import tests
     */
    @Test
    @DisplayName("Import Dcgan Mnist Discriminator")
    void importDcganMnistDiscriminator(@TempDir Path tempDir) throws Exception {
        importSequentialModelH5Test(tempDir,"modelimport/keras/examples/mnist_dcgan/dcgan_discriminator_epoch_50.h5");
    }

    @Test
    @Disabled("Neither keras or tfkeras can load this.")
    @DisplayName("Import Dcgan Mnist Generator")
    void importDcganMnistGenerator(@TempDir Path tempDir) throws Exception {
        importSequentialModelH5Test(tempDir,"modelimport/keras/examples/mnist_dcgan/dcgan_generator_epoch_50.h5");
    }

    /**
     * Auxillary classifier GAN import test
     */
    @Test
    @DisplayName("Import Acgan Discriminator")
    void importAcganDiscriminator(@TempDir Path tempDir) throws Exception {
        ComputationGraph model = importFunctionalModelH5Test(tempDir,"modelimport/keras/examples/acgan/acgan_discriminator_1_epochs.h5");
        // NHWC
        INDArray input = Nd4j.create(10, 28, 28, 1);
        INDArray[] output = model.output(input);
    }

    // AB 2020/04/22 Ignored until Keras model import updated to use NHWC support
    @Test
    @DisplayName("Import Acgan Generator")
    void importAcganGenerator(@TempDir Path tempDir) throws Exception {
        ComputationGraph model = importFunctionalModelH5Test(tempDir,"modelimport/keras/examples/acgan/acgan_generator_1_epochs.h5");
        // System.out.println(model.summary()) ;
        INDArray latent = Nd4j.create(10, 100);
        INDArray label = Nd4j.create(10, 1);
        INDArray[] output = model.output(latent, label);
    }

    @Test
    @DisplayName("Import Acgan Combined")
    void importAcganCombined(@TempDir Path tempDir) throws Exception {
        ComputationGraph model = importFunctionalModelH5Test(tempDir,"modelimport/keras/examples/acgan/acgan_combined_1_epochs.h5");
        // TODO: imports, but incorrectly. Has only one input, should have two.
    }

    /**
     * Deep convolutional GAN import test
     */
    @Test
    @DisplayName("Import Dcgan Discriminator")
    void importDcganDiscriminator(@TempDir Path tempDir) throws Exception {
        importSequentialModelH5Test(tempDir,"modelimport/keras/examples/gans/dcgan_discriminator.h5");
    }

    @Test
    @DisplayName("Import Dcgan Generator")
    void importDcganGenerator(@TempDir Path tempDir) throws Exception {
        importSequentialModelH5Test(tempDir,"modelimport/keras/examples/gans/dcgan_generator.h5");
    }

    /**
     * Wasserstein GAN import test
     */
    @Test
    @DisplayName("Import Wgan Discriminator")
    void importWganDiscriminator(@TempDir Path tempDir) throws Exception {
        for (int i = 0; i < 100; i++) {
            // run a few times to make sure HDF5 doesn't crash
            importSequentialModelH5Test(tempDir,"modelimport/keras/examples/gans/wgan_discriminator.h5");
        }
    }

    @Test
    @DisplayName("Import Wgan Generator")
    void importWganGenerator(@TempDir Path tempDir) throws Exception {
        importSequentialModelH5Test(tempDir,"modelimport/keras/examples/gans/wgan_generator.h5");
    }

    @Test
    @DisplayName("Import Cnn 1 d")
    void importCnn1d(@TempDir Path tempDir) throws Exception {
        importSequentialModelH5Test(tempDir,"modelimport/keras/examples/cnn1d/cnn1d_flatten_tf_keras2.h5");
    }

    /**
     * DGA classifier test
     */
    @Test
    @DisplayName("Import Dga Classifier")
    void importDgaClassifier(@TempDir Path tempDir) throws Exception {
        importSequentialModelH5Test(tempDir,"modelimport/keras/examples/dga_classifier/keras2_dga_classifier_tf_model.h5");
    }

    /**
     * Reshape flat input into 3D to fit into an LSTM model
     */
    @Test
    @DisplayName("Import Flat Into LSTM")
    void importFlatIntoLSTM(@TempDir Path tempDir) throws Exception {
        importFunctionalModelH5Test(tempDir,"modelimport/keras/examples/reshape_to_rnn/reshape_model.h5");
    }

    /**
     * Functional LSTM test
     */
    @Test
    @DisplayName("Import Functional Lstm Tf Keras 2")
    void importFunctionalLstmTfKeras2(@TempDir Path tempDir) throws Exception {
        String modelPath = "modelimport/keras/examples/functional_lstm/lstm_functional_tf_keras_2.h5";
        // No training enabled
        ComputationGraph graphNoTrain = importFunctionalModelH5Test(tempDir,modelPath, null, false);
        System.out.println(graphNoTrain.summary());
        // Training enabled
        ComputationGraph graph = importFunctionalModelH5Test(tempDir,modelPath, null, true);
        System.out.println(graph.summary());
        // Make predictions
        int miniBatch = 32;
        // NWC format - with nIn=4, seqLength = 10
        INDArray input = Nd4j.ones(miniBatch, 10, 4);
        INDArray[] out = graph.output(input);
        // Fit model
        graph.fit(new INDArray[] { input }, out);
    }

    /**
     * U-Net
     */
    @Test
    @DisplayName("Import Unet Tf Keras 2")
    void importUnetTfKeras2(@TempDir Path tempDir) throws Exception {
        importFunctionalModelH5Test(tempDir,"modelimport/keras/examples/unet/unet_keras_2_tf.h5", null, true);
    }

    /**
     * ResNet50
     */
    @Test
    @DisplayName("Import Resnet 50")
    void importResnet50(@TempDir Path tempDir) throws Exception {
        importFunctionalModelH5Test(tempDir,"modelimport/keras/examples/resnet/resnet50_weights_tf_dim_ordering_tf_kernels.h5");
    }

    /**
     * DenseNet
     */
    @Test
    @DisplayName("Import Dense Net")
    void importDenseNet(@TempDir Path tempDir) throws Exception {
        importFunctionalModelH5Test(tempDir,"modelimport/keras/examples/densenet/densenet121_tf_keras_2.h5");
    }

    /**
     * SqueezeNet
     */
    @Test
    @DisplayName("Import Squeeze Net")
    void importSqueezeNet(@TempDir Path tempDir) throws Exception {
        importFunctionalModelH5Test(tempDir,"modelimport/keras/examples/squeezenet/squeezenet.h5");
    }

    /**
     * MobileNet
     */
    @Test
    @DisplayName("Import Mobile Net")
    void importMobileNet(@TempDir Path tempDir) throws Exception {
        ComputationGraph graph = importFunctionalModelH5Test(tempDir,"modelimport/keras/examples/mobilenet/alternative.hdf5");
        INDArray input = Nd4j.ones(10, 299, 299, 3);
        graph.output(input);
    }

    /**
     * InceptionV3 Keras 2 no top
     */
    @Test
    @DisplayName("Import Inception Keras 2")
    void importInceptionKeras2(@TempDir Path tempDir) throws Exception {
        int[] inputShape = new int[] { 299, 299, 3 };
        ComputationGraph graph = importFunctionalModelH5Test(tempDir,"modelimport/keras/examples/inception/inception_tf_keras_2.h5", inputShape, false);
        // TF = channels last = NHWC
        INDArray input = Nd4j.ones(10, 299, 299, 3);
        graph.output(input);
        System.out.println(graph.summary());
    }

    /**
     * InceptionV3
     */
    @Test
    @DisplayName("Import Inception")
    // note this is actually keras 1 and its input dimension ordering is channels first
    // Takes unreasonably long, but works
    void importInception(@TempDir Path tempDir) throws Exception {
        ComputationGraph graph = importFunctionalModelH5Test(tempDir,"modelimport/keras/examples/inception/inception_v3_complete.h5");
        // TH = channels first = NCHW
        INDArray input = Nd4j.ones(10, 3, 299, 299);
        graph.output(input);
        System.out.println(graph.summary());
    }

    /**
     * Inception V4
     */
    @Test
    @Disabled
    @DisplayName("Import Inception V 4")
    // Model and weights have about 170mb, too large for test resources and also too excessive to enable as unit test
    void importInceptionV4(@TempDir Path testDir) throws Exception {
        String modelUrl = DL4JResources.getURLString("models/inceptionv4_keras_imagenet_weightsandconfig.h5");
        File kerasFile = testDir.resolve("inceptionv4_keras_imagenet_weightsandconfig.h5").toFile();
        if (!kerasFile.exists()) {
            FileUtils.copyURLToFile(new URL(modelUrl), kerasFile);
            kerasFile.deleteOnExit();
        }
        int[] inputShape = new int[] { 299, 299, 3 };
        ComputationGraph graph = importFunctionalModelH5Test(testDir,kerasFile.getAbsolutePath(), inputShape, false);
        // System.out.println(graph.summary());
    }

    /**
     * Xception
     */
    @Test
    @DisplayName("Import Xception")
    void importXception(@TempDir Path tempDir) throws Exception {
        int[] inputShape = new int[] { 299, 299, 3 };
        ComputationGraph graph = importFunctionalModelH5Test(tempDir,"modelimport/keras/examples/xception/xception_tf_keras_2.h5", inputShape, false);
    }

    /**
     * Seq2seq model
     */
    @Test
    @DisplayName("Import Seq 2 Seq")
    // does not work yet, needs DL4J enhancements
    void importSeq2Seq(@TempDir Path tempDir) throws Exception {
        importFunctionalModelH5Test(tempDir,"modelimport/keras/examples/seq2seq/full_model_seq2seq_5549.h5");
    }

    /**
     * Import all AlphaGo Zero model variants, i.e.
     * - Dual residual architecture
     * - Dual convolutional architecture
     * - Separate (policy and value) residual architecture
     * - Separate (policy and value) convolutional architecture
     */
    // AB 20200427 Bad keras model - Keras JSON has input shape [null, 10, 19, 19] (i.e., NCHW) but all layers are set to channels_last
    @Test
    @Disabled("Data and channel layout mismatch. We don't support permuting the weights yet.")
    @DisplayName("Import Sep Conv Policy")
    void importSepConvPolicy(@TempDir Path tempDir) throws Exception {
        ComputationGraph model = importFunctionalModelH5Test(tempDir,"modelimport/keras/examples/agz/sep_conv_policy.h5");
        INDArray input = Nd4j.create(32, 19, 19, 10);
        model.output(input);
    }

    // AB 20200427 Bad keras model - Keras JSON has input shape [null, 10, 19, 19] (i.e., NCHW) but all layers are set to channels_last
    @Test
    @Disabled("Data and channel layout mismatch. We don't support permuting the weights yet.")
    @DisplayName("Import Sep Res Policy")
    void importSepResPolicy(@TempDir Path tempDir) throws Exception {
        ComputationGraph model = importFunctionalModelH5Test(tempDir,"modelimport/keras/examples/agz/sep_res_policy.h5");
        INDArray input = Nd4j.create(32, 19, 19, 10);
        model.output(input);
    }

    // AB 20200427 Bad keras model - Keras JSON has input shape [null, 10, 19, 19] (i.e., NCHW) but all layers are set to channels_last
    @Test
    @Disabled("Data and channel layout mismatch. We don't support permuting the weights yet.")
    @DisplayName("Import Sep Conv Value")
    void importSepConvValue(@TempDir Path tempDir) throws Exception {
        ComputationGraph model = importFunctionalModelH5Test(tempDir,"modelimport/keras/examples/agz/sep_conv_value.h5");
        INDArray input = Nd4j.create(32, 19, 19, 10);
        model.output(input);
    }

    @Test
    @Disabled("Data and channel layout mismatch. We don't support permuting the weights yet.")
    @DisplayName("Import Sep Res Value")
    void importSepResValue(@TempDir Path tempDir) throws Exception {
        String filePath = "C:\\Users\\agibs\\Documents\\GitHub\\keras1-import-test\\sep_res_value.h5";
        KerasModelBuilder builder = new KerasModel().modelBuilder().modelHdf5Filename(filePath).enforceTrainingConfig(false);
        KerasModel model = builder.buildModel();
        ComputationGraph compGraph = model.getComputationGraph();
        // ComputationGraph model = importFunctionalModelH5Test("modelimport/keras/examples/agz/sep_res_value.h5");
        INDArray input = Nd4j.create(32, 19, 19, 10);
        compGraph.output(input);
    }

    // AB 20200427 Bad keras model - Keras JSON has input shape [null, 10, 19, 19] (i.e., NCHW) but all layers are set to channels_last
    @Test
    @Disabled("Data and channel layout mismatch. We don't support permuting the weights yet.")
    @DisplayName("Import Dual Res")
    void importDualRes(@TempDir Path tempDir) throws Exception {
        ComputationGraph model = importFunctionalModelH5Test(tempDir,"modelimport/keras/examples/agz/dual_res.h5");
        INDArray input = Nd4j.create(32, 19, 19, 10);
        model.output(input);
    }

    @Test
    @Disabled("Data and channel layout mismatch. We don't support permuting the weights yet.")
    @DisplayName("Import Dual Conv")
    void importDualConv(@TempDir Path tempDir) throws Exception {
        ComputationGraph model = importFunctionalModelH5Test(tempDir,"modelimport/keras/examples/agz/dual_conv.h5");
        INDArray input = Nd4j.create(32, 19, 19, 10);
        model.output(input);
    }

    /**
     * MTCNN
     */
    @Test
    @DisplayName("Import MTCNN")
    void importMTCNN(@TempDir Path tempDir) throws Exception {
        ComputationGraph model = importFunctionalModelH5Test(tempDir,"modelimport/keras/examples/48net_complete.h5");
    }

    @Test
    @Disabled("Data and channel layout mismatch. We don't support permuting the weights yet.")
    @DisplayName("Test NCHWNWHC Change Import Model")
    void testNCHWNWHCChangeImportModel(@TempDir Path tempDir) throws Exception {
        ComputationGraph computationGraph = importFunctionalModelH5Test(tempDir,"modelimport/keras/weights/simpleconv2d_model.hdf5");
        computationGraph.output(Nd4j.zeros(1, 1, 28, 28));
    }

    @Test
    @DisplayName("Import MTCNN 2 D")
    // TODO: fails, since we can't use OldSoftMax on >2D data (here: convolution layer)
    // TODO: also related to #6339, fix this together
    void importMTCNN2D(@TempDir Path tempDir) throws Exception {
        ComputationGraph model = importFunctionalModelH5Test(tempDir,"modelimport/keras/examples/12net.h5", new int[] { 24, 24, 3 }, false);
        INDArray input = Nd4j.create(10, 24, 24, 3);
        model.output(input);
        // System.out.println(model.summary());
    }

    /**
     * Masking layers (simple Masking into LSTM)
     */
    @Test
    @DisplayName("Test Masking Zero Value")
    void testMaskingZeroValue(@TempDir Path tempDir) throws Exception {
        MultiLayerNetwork model = importSequentialModelH5Test(tempDir,"modelimport/keras/examples/masking/masking_zero_lstm.h5");
        model.summary();
    }

    @Test
    @DisplayName("Test Masking Two Value")
    void testMaskingTwoValue(@TempDir Path tempDir) throws Exception {
        MultiLayerNetwork model = importSequentialModelH5Test(tempDir,"modelimport/keras/examples/masking/masking_two_lstm.h5");
        model.summary();
    }

    @Test
    @DisplayName("Test Causal Conv 1 D")
    void testCausalConv1D(@TempDir Path tempDir) throws Exception {
        String[] names = new String[] { "causal_conv1d_k2_s1_d1_cl_model.h5", "causal_conv1d_k2_s1_d2_cl_model.h5", "causal_conv1d_k2_s2_d1_cl_model.h5", "causal_conv1d_k2_s3_d1_cl_model.h5", "causal_conv1d_k3_s1_d1_cl_model.h5", "causal_conv1d_k3_s1_d2_cl_model.h5", "causal_conv1d_k3_s2_d1_cl_model.h5", "causal_conv1d_k3_s3_d1_cl_model.h5", "causal_conv1d_k4_s1_d1_cl_model.h5", "causal_conv1d_k4_s1_d2_cl_model.h5", "causal_conv1d_k4_s2_d1_cl_model.h5", "causal_conv1d_k4_s3_d1_cl_model.h5" };
        for (String name : names) {
            System.out.println("Starting test: " + name);
            String modelPath = "modelimport/keras/examples/causal_conv1d/" + name;
            String inputsOutputPath = "modelimport/keras/examples/causal_conv1d/" + (name.substring(0, name.length() - "model.h5".length()) + "inputs_and_outputs.h5");
            // TODO:
            /**
             * Difference in weights. Same elements, but loaded differently. Likely acceptable difference. Need to confirm though.
             */
            MultiLayerNetwork net = importEndModelTest(tempDir,modelPath, inputsOutputPath, true, true, true, true, false, null, null);
            Layer l = net.getLayer(0);
            Convolution1DLayer c1d = (Convolution1DLayer) l.getConfig();
            assertEquals(ConvolutionMode.Causal, c1d.getConvolutionMode());
        }
    }

    @Test
    @DisplayName("Test Conv 1 D")
    void testConv1D(@TempDir Path tempDir) throws Exception {
        String[] names = new String[] { "conv1d_k2_s1_d1_cf_same_model.h5", "conv1d_k2_s1_d1_cf_valid_model.h5", "conv1d_k2_s1_d1_cl_same_model.h5", "conv1d_k2_s1_d1_cl_valid_model.h5", "conv1d_k2_s1_d2_cf_same_model.h5", "conv1d_k2_s1_d2_cf_valid_model.h5", "conv1d_k2_s1_d2_cl_same_model.h5", "conv1d_k2_s1_d2_cl_valid_model.h5", "conv1d_k2_s2_d1_cf_same_model.h5", "conv1d_k2_s2_d1_cf_valid_model.h5", "conv1d_k2_s2_d1_cl_same_model.h5", "conv1d_k2_s2_d1_cl_valid_model.h5", "conv1d_k2_s3_d1_cf_same_model.h5", "conv1d_k2_s3_d1_cf_valid_model.h5", "conv1d_k2_s3_d1_cl_same_model.h5", "conv1d_k2_s3_d1_cl_valid_model.h5", "conv1d_k3_s1_d1_cf_same_model.h5", "conv1d_k3_s1_d1_cf_valid_model.h5", "conv1d_k3_s1_d1_cl_same_model.h5", "conv1d_k3_s1_d1_cl_valid_model.h5", "conv1d_k3_s1_d2_cf_same_model.h5", "conv1d_k3_s1_d2_cf_valid_model.h5", "conv1d_k3_s1_d2_cl_same_model.h5", "conv1d_k3_s1_d2_cl_valid_model.h5", "conv1d_k3_s2_d1_cf_same_model.h5", "conv1d_k3_s2_d1_cf_valid_model.h5", "conv1d_k3_s2_d1_cl_same_model.h5", "conv1d_k3_s2_d1_cl_valid_model.h5", "conv1d_k3_s3_d1_cf_same_model.h5", "conv1d_k3_s3_d1_cf_valid_model.h5", "conv1d_k3_s3_d1_cl_same_model.h5", "conv1d_k3_s3_d1_cl_valid_model.h5", "conv1d_k4_s1_d1_cf_same_model.h5", "conv1d_k4_s1_d1_cf_valid_model.h5", "conv1d_k4_s1_d1_cl_same_model.h5", "conv1d_k4_s1_d1_cl_valid_model.h5", "conv1d_k4_s1_d2_cf_same_model.h5", "conv1d_k4_s1_d2_cf_valid_model.h5", "conv1d_k4_s1_d2_cl_same_model.h5", "conv1d_k4_s1_d2_cl_valid_model.h5", "conv1d_k4_s2_d1_cf_same_model.h5", "conv1d_k4_s2_d1_cf_valid_model.h5", "conv1d_k4_s2_d1_cl_same_model.h5", "conv1d_k4_s2_d1_cl_valid_model.h5", "conv1d_k4_s3_d1_cf_same_model.h5", "conv1d_k4_s3_d1_cf_valid_model.h5", "conv1d_k4_s3_d1_cl_same_model.h5", "conv1d_k4_s3_d1_cl_valid_model.h5" };
        for (String name : names) {
            System.out.println("Starting test: " + name);
            String modelPath = "modelimport/keras/examples/conv1d/" + name;
            String inputsOutputPath = "modelimport/keras/examples/conv1d/" + (name.substring(0, name.length() - "model.h5".length()) + "inputs_and_outputs.h5");
            importEndModelTest(tempDir,modelPath, inputsOutputPath, true, true, true, true, false, null, // f, f2);
            null);
        }
    }

    @Test
    @DisplayName("Test Activation Layers")
    void testActivationLayers(@TempDir Path tempDir) throws Exception {
        String[] names = new String[] { "ELU_0_model.h5", "LeakyReLU_0_model.h5", "ReLU_0_model.h5", "ReLU_1_model.h5", "ReLU_2_model.h5", "ReLU_3_model.h5", "Softmax_0_model.h5", "ThresholdReLU_0_model.h5" };
        for (String name : names) {
            System.out.println("Starting test: " + name);
            String modelPath = "modelimport/keras/examples/activations/" + name;
            String inputsOutputPath = "modelimport/keras/examples/activations/" + (name.substring(0, name.length() - "model.h5".length()) + "inputs_and_outputs.h5");
            importEndModelTest(tempDir,modelPath, inputsOutputPath, true, true, true, true, false, null, null);
        }
    }

    private ComputationGraph importFunctionalModelH5Test(Path tempDir,String modelPath) throws Exception {
        return importFunctionalModelH5Test(tempDir,modelPath, null, false);
    }

    private ComputationGraph importFunctionalModelH5Test(Path tempDir,String modelPath, int[] inputShape, boolean train) throws Exception {
        File modelFile;
        try (InputStream is = Resources.asStream(modelPath)) {
            modelFile = createTempFile(tempDir,TEMP_MODEL_FILENAME, H5_EXTENSION);
            Files.copy(is, modelFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        }
        KerasModelBuilder builder = new KerasModel().modelBuilder().modelHdf5Filename(modelFile.getAbsolutePath()).enforceTrainingConfig(train);
        if (inputShape != null) {
            builder.inputShape(inputShape);
        }
        KerasModel model = builder.buildModel();
        return model.getComputationGraph();
    }

    private MultiLayerNetwork importSequentialModelH5Test(Path tempDir,String modelPath) throws Exception {
        return importSequentialModelH5Test(tempDir,modelPath, null);
    }

    private MultiLayerNetwork importSequentialModelH5Test(Path tempDir,String modelPath, int[] inputShape) throws Exception {
        try (InputStream is = Resources.asStream(modelPath)) {
            File modelFile = createTempFile(tempDir,TEMP_MODEL_FILENAME, H5_EXTENSION);
            Files.copy(is, modelFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
            KerasModelBuilder builder = new KerasModel().modelBuilder().modelHdf5Filename(modelFile.getAbsolutePath()).enforceTrainingConfig(false);
            if (inputShape != null) {
                builder.inputShape(inputShape);
            }
            KerasSequentialModel model = builder.buildSequential();
            return model.getMultiLayerNetwork();
        }
    }

    public MultiLayerNetwork importEndModelTest(Path tempDir,String modelPath, String inputsOutputsPath, boolean tfOrdering, boolean checkPredictions, boolean checkGradients, boolean enforceTrainingConfig) throws Exception {
        return importEndModelTest(tempDir,modelPath, inputsOutputsPath, tfOrdering, checkPredictions, checkGradients, true, enforceTrainingConfig, null, null);
    }

    public MultiLayerNetwork importEndModelTest(Path tempDir,String modelPath, String inputsOutputsPath, boolean tfOrdering, boolean checkPredictions, boolean checkGradients, boolean enforceTrainingConfig, boolean checkAuc, Function<INDArray, INDArray> inputPreProc, BiFunction<String, INDArray, INDArray> expectedPreProc) throws Exception {
        MultiLayerNetwork model;
        try (InputStream is = Resources.asStream(modelPath)) {
            File modelFile = createTempFile(tempDir,TEMP_MODEL_FILENAME, H5_EXTENSION);
            Files.copy(is, modelFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
            KerasSequentialModel kerasModel = new KerasModel().modelBuilder().modelHdf5Filename(modelFile.getAbsolutePath()).enforceTrainingConfig(enforceTrainingConfig).buildSequential();
            model = kerasModel.getMultiLayerNetwork();
        }
        File outputsFile = createTempFile(tempDir,TEMP_OUTPUTS_FILENAME, H5_EXTENSION);
        try (InputStream is = Resources.asStream(inputsOutputsPath)) {
            Files.copy(is, outputsFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        }
        try (Hdf5Archive outputsArchive = new Hdf5Archive(outputsFile.getAbsolutePath())) {
            if (checkPredictions) {
                INDArray input = getInputs(outputsArchive, tfOrdering)[0];
                if (inputPreProc != null)
                    input = inputPreProc.apply(input);
                Map<String, INDArray> activationsKeras = getActivations(outputsArchive, tfOrdering);
                for (int i = 0; i < model.getLayers().length; i++) {
                    String layerName = model.getLayerNames().get(i);
                    if (activationsKeras.containsKey(layerName)) {
                        INDArray activationsDl4j = model.feedForwardToLayer(i, input, false).get(i + 1);
                        long[] shape = activationsDl4j.shape();
                        INDArray exp = activationsKeras.get(layerName);
                        Nd4j.getExecutioner().enableDebugMode(true);
                        Nd4j.getExecutioner().enableVerboseMode(true);
                        if (expectedPreProc != null)
                            exp = expectedPreProc.apply(layerName, exp);
                        compareINDArrays(layerName, exp, activationsDl4j, EPS);
                    }
                }
                INDArray predictionsKeras = getPredictions(outputsArchive, tfOrdering)[0];
                INDArray predictionsDl4j = model.output(input, false);
                if (expectedPreProc != null)
                    predictionsKeras = expectedPreProc.apply("output", predictionsKeras);
                compareINDArrays("predictions", predictionsKeras, predictionsDl4j, EPS);
                INDArray outputs = getOutputs(outputsArchive, true)[0];
                if (outputs.rank() == 1) {
                    outputs = outputs.reshape(outputs.length(), 1);
                }
                val nOut = (int) outputs.size(-1);
                if (checkAuc)
                    compareMulticlassAUC("predictions", outputs, predictionsKeras, predictionsDl4j, nOut, EPS);
            }
            if (checkGradients && !SKIP_GRAD_CHECKS) {
                Random r = new Random(12345);
                INDArray input = getInputs(outputsArchive, tfOrdering)[0];
                INDArray predictionsDl4j = model.output(input, false);
                // Infer one-hot labels... this probably won't work for all
                INDArray testLabels = Nd4j.create(predictionsDl4j.shape());
                if (testLabels.rank() == 2) {
                    for (int i = 0; i < testLabels.size(0); i++) {
                        testLabels.putScalar(i, r.nextInt((int) testLabels.size(1)), 1.0);
                    }
                } else if (testLabels.rank() == 3) {
                    for (int i = 0; i < testLabels.size(0); i++) {
                        for (int j = 0; j < testLabels.size(1); j++) {
                            testLabels.putScalar(i, j, r.nextInt((int) testLabels.size(1)), 1.0);
                        }
                    }
                } else {
                    throw new RuntimeException("Cannot gradient check 4d output array");
                }
                checkGradients(model, input, testLabels);
            }
        }
        return model;
    }

    private static INDArray[] getInputs(Hdf5Archive archive, boolean tensorFlowImageDimOrdering) throws Exception {
        List<String> inputNames = (List<String>) KerasModelUtils.parseJsonString(archive.readAttributeAsJson(GROUP_ATTR_INPUTS)).get(GROUP_ATTR_INPUTS);
        INDArray[] inputs = new INDArray[inputNames.size()];
        for (int i = 0; i < inputNames.size(); i++) {
            inputs[i] = archive.readDataSet(inputNames.get(i), GROUP_ATTR_INPUTS);
        }
        return inputs;
    }

    private static Map<String, INDArray> getActivations(Hdf5Archive archive, boolean tensorFlowImageDimOrdering) throws Exception {
        Map<String, INDArray> activations = new HashMap<>();
        for (String layerName : archive.getDataSets(GROUP_ACTIVATIONS)) {
            INDArray activation = archive.readDataSet(layerName, GROUP_ACTIVATIONS);
            activations.put(layerName, activation);
        }
        return activations;
    }

    private static INDArray[] getOutputs(Hdf5Archive archive, boolean tensorFlowImageDimOrdering) throws Exception {
        List<String> outputNames = (List<String>) KerasModelUtils.parseJsonString(archive.readAttributeAsJson(GROUP_ATTR_OUTPUTS)).get(GROUP_ATTR_OUTPUTS);
        INDArray[] outputs = new INDArray[outputNames.size()];
        for (int i = 0; i < outputNames.size(); i++) {
            outputs[i] = archive.readDataSet(outputNames.get(i), GROUP_ATTR_OUTPUTS);
        }
        return outputs;
    }

    private static INDArray[] getPredictions(Hdf5Archive archive, boolean tensorFlowImageDimOrdering) throws Exception {
        List<String> outputNames = (List<String>) KerasModelUtils.parseJsonString(archive.readAttributeAsJson(GROUP_ATTR_OUTPUTS)).get(GROUP_ATTR_OUTPUTS);
        INDArray[] predictions = new INDArray[outputNames.size()];
        for (int i = 0; i < outputNames.size(); i++) {
            predictions[i] = archive.readDataSet(outputNames.get(i), GROUP_PREDICTIONS);
        }
        return predictions;
    }

    private static void compareINDArrays(String label, INDArray expected, INDArray actual, double eps) {
        if (!expected.equalShapes(actual)) {
            throw new IllegalStateException("Shapes do not match for \"" + label + "\": got " + Arrays.toString(expected.shape()) + " vs " + Arrays.toString(actual.shape()));
        }
        INDArray diff = expected.sub(actual.castTo(expected.dataType()));
        double min = diff.minNumber().doubleValue();
        double max = diff.maxNumber().doubleValue();
        log.info(label + ": " + expected.equalsWithEps(actual, eps) + ", " + min + ", " + max);
        double threshold = 1e-7;
        double aAbsMax = Math.max(Math.abs(expected.minNumber().doubleValue()), Math.abs(expected.maxNumber().doubleValue()));
        double bAbsMax = Math.max(Math.abs(actual.minNumber().doubleValue()), Math.abs(actual.maxNumber().doubleValue()));
        // skip too small absolute inputs
        if (Math.abs(aAbsMax) > threshold && Math.abs(bAbsMax) > threshold) {
            boolean eq = expected.equalsWithEps(actual.castTo(expected.dataType()), eps);
            if (!eq) {
                System.out.println("Expected: " + Arrays.toString(expected.shape()) + ", actual: " + Arrays.toString(actual.shape()));
                System.out.println("Expected:\n" + expected);
                System.out.println("Actual: \n" + actual);
            }
            assertTrue(eq,"Output differs: " + label);
        }
    }

    private static void compareMulticlassAUC(String label, INDArray target, INDArray a, INDArray b, int nbClasses, double eps) {
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
                l = new LossLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).build();
            } else {
                // Rank 3
                l = new RnnOutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nIn(labels.size(1)).nOut(labels.size(1)).build();
            }
            netToTest = new TransferLearning.Builder(net).fineTuneConfiguration(new FineTuneConfiguration.Builder().updater(new NoOp()).dropOut(0.0).build()).addLayer(l).build();
        }
        log.info("Num params: " + net.numParams());
        for (Layer l : netToTest.getLayers()) {
            // Remove any dropout manually - until this is fixed:
            // https://github.com/eclipse/deeplearning4j/issues/4368
            l.conf().getLayer().setIDropout(null);
            // Also swap out activation functions... this is a bit of a hack, but should make the net gradient checkable...
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
        Nd4j.setDataType(DataType.DOUBLE);
        boolean passed = GradientCheckUtil.checkGradients(new GradientCheckUtil.MLNConfig().net(netToTest).input(input).labels(labels).subset(true).maxPerParam(9));
        assertTrue(passed, "Gradient check failed");
    }

    private File createTempFile(Path testDir,String prefix, String suffix) throws IOException {
        File ret = new File(testDir.toFile(),prefix + "-" + System.nanoTime() + suffix);
        ret.createNewFile();
        ret.deleteOnExit();
        return ret;
    }
}

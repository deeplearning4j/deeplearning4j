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
package org.deeplearning4j.nn.modelimport.keras.configurations;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasModel;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasSpaceToDepth;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.resources.Resources;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.extension.ExtendWith;

import static org.junit.jupiter.api.Assertions.*;

@Slf4j
@DisplayName("Keras 2 Model Configuration Test")
@Tag(TagNames.FILE_IO)
@Tag(TagNames.KERAS)
@NativeTag
class Keras2ModelConfigurationTest extends BaseDL4JTest {

    ClassLoader classLoader = getClass().getClassLoader();

    @Test
    @DisplayName("File Not Found Test")
    void fileNotFoundTest() {
        assertThrows(IllegalStateException.class, () -> {
            runModelConfigTest("modelimport/keras/foo/bar.json");
        });
    }

    @Test
    @DisplayName("Not A File Test")
    void notAFileTest() {
        assertThrows(IOException.class, () -> {
            runModelConfigTest("modelimport/keras/");
        });
    }

    @Test
    @DisplayName("Simple 222 Config Test")
    void simple222ConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras2/model_2_2_2.json");
    }

    @Test
    @DisplayName("Simple 224 Config Test")
    void simple224ConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras2/model_2_2_4.json");
    }

    @Test
    @DisplayName("Yolo 9000 Config Test")
    void yolo9000ConfigTest() throws Exception {
        KerasLayer.registerCustomLayer("Lambda", KerasSpaceToDepth.class);
        runModelConfigTest("modelimport/keras/configs/keras2/yolo9000_tf_keras_2.json");
    }

    @Test
    @DisplayName("L 1 l 2 Regularizer Dense Tf Config Test")
    void l1l2RegularizerDenseTfConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras2/l1l2_regularizer_dense_tf_keras_2_config.json");
    }

    @Test
    @DisplayName("Dga Classifier Tf Config Test")
    void dgaClassifierTfConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras2/keras2_dga_classifier_tf_config.json");
    }

    @Test
    @DisplayName("Conv Pooling 1 d Tf Config Test")
    void convPooling1dTfConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras2/keras2_conv1d_pooling1d_tf_config.json");
    }

    @Test
    @DisplayName("Bidirectional Lstm Config Test")
    void bidirectionalLstmConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras2/bidirectional_lstm_tf_keras_2_config.json");
    }

    @Test
    @DisplayName("Imdb Lstm Tf Sequential Config Test")
    void imdbLstmTfSequentialConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras2/imdb_lstm_tf_keras_2_config.json");
    }

    @Test
    @DisplayName("Imdb Lstm Th Sequential Config Test")
    void imdbLstmThSequentialConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras2/imdb_lstm_th_keras_2_config.json");
    }

    @Test
    @DisplayName("Simple Rnn Config Test")
    void simpleRnnConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras2/simple_rnn_tf_keras_2_config.json");
    }

    @Test
    @DisplayName("Simple Prelu Config Test")
    void simplePreluConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras2/prelu_config_tf_keras_2.json");
    }

    @Test
    @DisplayName("Mnist Mlp Tf Sequential Config Test")
    void mnistMlpTfSequentialConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras2/mnist_mlp_tf_keras_2_config.json");
    }

    @Test
    @DisplayName("Mnist Mlp Th Sequential Config Test")
    void mnistMlpThSequentialConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras2/mnist_mlp_th_keras_2_config.json");
    }

    @Test
    @DisplayName("Mnist Cnn Tf Sequential Config Test")
    void mnistCnnTfSequentialConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras2/mnist_cnn_tf_keras_2_config.json");
    }

    @Test
    @DisplayName("Mnist Cnn Th Sequential Config Test")
    void mnistCnnThSequentialConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras2/mnist_cnn_th_keras_2_config.json");
    }

    @Test
    @DisplayName("Mnist Cnn No Bias Tf Sequential Config Test")
    void mnistCnnNoBiasTfSequentialConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras2/keras2_mnist_cnn_no_bias_tf_config.json");
    }

    @Test
    @DisplayName("Mlp Sequential Config Test")
    void mlpSequentialConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras2/keras2_mlp_config.json");
    }

    @Test
    @DisplayName("Mlp Constraints Config Test")
    void mlpConstraintsConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras2/mnist_mlp_constraint_tf_keras_2_config.json");
    }

    @Test
    @DisplayName("Embedding Flatten Th Test")
    void embeddingFlattenThTest() throws Exception {
        runModelConfigTest("modelimport/keras/configs/keras2/embedding_flatten_graph_th_keras_2.json");
    }

    @Test
    @DisplayName("Mlp Fapi Config Test")
    void mlpFapiConfigTest() throws Exception {
        runModelConfigTest("modelimport/keras/configs/keras2/keras2_mlp_fapi_config.json");
    }

    @Test
    @DisplayName("Mlp Fapi Multi Loss Config Test")
    void mlpFapiMultiLossConfigTest() throws Exception {
        runModelConfigTest("modelimport/keras/configs/keras2/keras2_mlp_fapi_multiloss_config.json");
    }

    @Test
    @DisplayName("Cnn Tf Test")
    void cnnTfTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras2/keras2_cnn_tf_config.json");
    }

    @Test
    @DisplayName("Cnn Th Test")
    void cnnThTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras2/keras2_cnn_th_config.json");
    }

    @Test
    @DisplayName("Mnist Cnn Tf Test")
    void mnistCnnTfTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras2/keras2_mnist_cnn_tf_config.json");
    }

    @Test
    @DisplayName("Mnist Mlp Tf Test")
    void mnistMlpTfTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras2/keras2_mnist_mlp_tf_config.json");
    }

    @Test
    @DisplayName("Embedding Conv 1 D Tf Test")
    void embeddingConv1DTfTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras2/keras2_tf_embedding_conv1d_config.json");
    }

    @Test
    @DisplayName("Flatten Conv 1 D Tf Test")
    void flattenConv1DTfTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras2/flatten_conv1d_tf_keras_2.json");
    }

    @Test
    @DisplayName("Embedding LSTM Mask Zero Test")
    void embeddingLSTMMaskZeroTest() throws Exception {
        String path = "modelimport/keras/configs/keras2/embedding_lstm_calculator.json";
        try (InputStream is = Resources.asStream(path)) {
            ComputationGraphConfiguration config = new KerasModel().modelBuilder().modelJsonInputStream(is).enforceTrainingConfig(false).buildModel().getComputationGraphConfiguration();
            ComputationGraph model = new ComputationGraph(config);
            model.init();
            INDArray output = model.outputSingle(Nd4j.zeros(1, 3));
            System.out.println(output.shapeInfoToString());
        }
    }

    @Test
    @DisplayName("Permute Retina Unet")
    void permuteRetinaUnet() throws Exception {
        runModelConfigTest("modelimport/keras/configs/keras2/permute_retina_unet.json");
    }

    @Test
    @DisplayName("Simple Add Layer Test")
    void simpleAddLayerTest() throws Exception {
        runModelConfigTest("modelimport/keras/configs/keras2/simple_add_tf_keras_2.json");
    }

    @Override
    public long getTimeoutMilliseconds() {
        return 999999999L;
    }

    @Test
    @DisplayName("Embedding Concat Test")
    void embeddingConcatTest() throws Exception {
        runModelConfigTest("/modelimport/keras/configs/keras2/model_concat_embedding_sequences_tf_keras_2.json");
    }

    @Test
    @DisplayName("Conv 1 d Dilation Test")
    void conv1dDilationTest() throws Exception {
        runModelConfigTest("/modelimport/keras/configs/keras2/conv1d_dilation_tf_keras_2_config.json");
    }

    @Test
    @DisplayName("Test 5982")
    void test5982() throws Exception {
        File jsonFile = Resources.asFile("modelimport/keras/configs/bidirectional_last_timeStep.json");
        val modelGraphConf = KerasModelImport.importKerasSequentialConfiguration(jsonFile.getAbsolutePath());
        MultiLayerNetwork model = new MultiLayerNetwork(modelGraphConf);
        INDArray features = Nd4j.create(new double[] { 1, 3, 1, 2, 2, 1, 82, 2, 10, 1, 3, 1, 2, 1, 82, 3, 1, 10, 1, 2, 1, 3, 1, 10, 82, 2, 1, 1, 10, 82, 2, 3, 1, 2, 1, 10, 1, 2, 3, 82, 2, 1, 10, 3, 82, 1, 2, 1, 10, 1 }, new int[] { 1, 1, 50 });
        model.init();
        INDArray out = model.output(features);
        assertArrayEquals(new long[] { 1, 14 }, out.shape());
    }

    @Test
    @DisplayName("One Lstm Layer Test")
    @Tag(TagNames.LARGE_RESOURCES)
    @Tag(TagNames.LONG_TEST)
    void oneLstmLayerTest() throws Exception {
        try (InputStream is = Resources.asStream("/modelimport/keras/configs/keras2/one_lstm_no_sequences_tf_keras_2.json")) {
            MultiLayerConfiguration config = new KerasModel().modelBuilder().modelJsonInputStream(is).enforceTrainingConfig(false).buildSequential().getMultiLayerConfiguration();
            MultiLayerNetwork model = new MultiLayerNetwork(config);
            model.init();
            // NWC format - [Minibatch, seqLength, channels]
            INDArray input = Nd4j.create(DataType.FLOAT, 50, 1500, 500);
            INDArray out = model.output(input);
            assertTrue(Arrays.equals(out.shape(), new long[] { 50, 64 }));
        }
    }

    @Test
    @DisplayName("Reshape Embedding Concat Test")
    // @Disabled("AB 2019/11/23 - known issue - see https://github.com/eclipse/deeplearning4j/issues/8373 and https://github.com/eclipse/deeplearning4j/issues/8441")
    void ReshapeEmbeddingConcatTest() throws Exception {
        try (InputStream is = Resources.asStream("/modelimport/keras/configs/keras2/reshape_embedding_concat.json")) {
            ComputationGraphConfiguration config = new KerasModel().modelBuilder().modelJsonInputStream(is).enforceTrainingConfig(false).buildModel().getComputationGraphConfiguration();
            ComputationGraph model = new ComputationGraph(config);
            model.init();
            // System.out.println(model.summary());
            model.outputSingle(Nd4j.zeros(1, 1), Nd4j.zeros(1, 1), Nd4j.zeros(1, 1));
        }
    }

    private void runSequentialConfigTest(String path) throws Exception {
        try (InputStream is = Resources.asStream(path)) {
            MultiLayerConfiguration config = new KerasModel().modelBuilder().modelJsonInputStream(is).enforceTrainingConfig(false).buildSequential().getMultiLayerConfiguration();
            MultiLayerNetwork model = new MultiLayerNetwork(config);
            model.init();
        }
    }

    private void runModelConfigTest(String path) throws Exception {
        try (InputStream is = Resources.asStream(path)) {
            ComputationGraphConfiguration config = new KerasModel().modelBuilder().modelJsonInputStream(is).enforceTrainingConfig(false).buildModel().getComputationGraphConfiguration();
            ComputationGraph model = new ComputationGraph(config);
            model.init();
        }
    }
}

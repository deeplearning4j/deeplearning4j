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
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.modelimport.keras.KerasModel;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.resources.Resources;
import java.io.InputStream;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.extension.ExtendWith;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;

@Slf4j
@DisplayName("Keras 1 Model Configuration Test")
@Tag(TagNames.FILE_IO)
@Tag(TagNames.KERAS)
@NativeTag
class Keras1ModelConfigurationTest extends BaseDL4JTest {

    private ClassLoader classLoader = getClass().getClassLoader();

    @Test
    @DisplayName("Imdb Lstm Tf Sequential Config Test")
    void imdbLstmTfSequentialConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/imdb_lstm_tf_keras_1_config.json", true);
    }

    @Test
    @DisplayName("Imdb Lstm Th Sequential Config Test")
    void imdbLstmThSequentialConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/imdb_lstm_th_keras_1_config.json", true);
    }

    @Test
    @DisplayName("Mnist Mlp Tf Sequential Config Test")
    void mnistMlpTfSequentialConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/mnist_mlp_tf_keras_1_config.json", true);
    }

    @Test
    @DisplayName("Mnist Mlp Th Sequential Config Test")
    void mnistMlpThSequentialConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/mnist_mlp_th_keras_1_config.json", true);
    }

    @Test
    @DisplayName("Mnist Cnn Tf Sequential Config Test")
    void mnistCnnTfSequentialConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/mnist_cnn_tf_keras_1_config.json", true);
    }

    @Test
    @DisplayName("Mnist Cnn No Bias Tf Sequential Config Test")
    void mnistCnnNoBiasTfSequentialConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/mnist_cnn_no_bias_tf_config.json", true);
    }

    @Test
    @DisplayName("Mnist Cnn Th Sequential Config Test")
    void mnistCnnThSequentialConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/mnist_cnn_th_keras_1_config.json", true);
    }

    @Test
    @DisplayName("Mlp Sequential Config Test")
    void mlpSequentialConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/mlp_config.json", true);
    }

    @Test
    @DisplayName("Mlp Constraints Config Test")
    void mlpConstraintsConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/mnist_mlp_constraint_tf_keras_1_config.json", true);
    }

    @Test
    @DisplayName("Reshape Mlp Config Test")
    void reshapeMlpConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/mnist_mlp_reshape_tf_keras_1_config.json", true);
    }

    @Test
    @DisplayName("Reshape Cnn Config Test")
    void reshapeCnnConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/mnist_cnn_reshape_tf_keras_1_config.json", true);
    }

    @Test
    @DisplayName("Mlp Fapi Config Test")
    void mlpFapiConfigTest() throws Exception {
        runModelConfigTest("modelimport/keras/configs/keras1/mlp_fapi_config.json");
    }

    @Test
    @DisplayName("Mlp Fapi Multi Loss Config Test")
    void mlpFapiMultiLossConfigTest() throws Exception {
        runModelConfigTest("modelimport/keras/configs/keras1/mlp_fapi_multiloss_config.json");
    }

    @Test
    @DisplayName("Yolo Config Test")
    void yoloConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/yolo_model.json", true);
    }

    @Test
    @DisplayName("Cnn Tf Test")
    void cnnTfTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/cnn_tf_config.json", true);
    }

    @Test
    @DisplayName("Cnn Th Test")
    void cnnThTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/cnn_th_config.json", true);
    }

    @Test
    @DisplayName("Lstm Fixed Len Test")
    void lstmFixedLenTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/lstm_tddense_config.json", false);
    }

    @Test
    @DisplayName("Mnist Cnn Tf Test")
    void mnistCnnTfTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/mnist_cnn_tf_config.json", true);
    }

    @Test
    @DisplayName("Mnist Mlp Tf Test")
    void mnistMlpTfTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/mnist_mlp_tf_config.json", true);
    }

    @Test
    @DisplayName("Embedding Conv 1 D Tf Test")
    void embeddingConv1DTfTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/keras1_tf_embedding_conv1d_config.json", true);
    }

    private void runSequentialConfigTest(String path, boolean training) throws Exception {
        try (InputStream is = Resources.asStream(path)) {
            MultiLayerConfiguration config = new KerasModel().modelBuilder().modelJsonInputStream(is).enforceTrainingConfig(training).buildSequential().getMultiLayerConfiguration();
            MultiLayerNetwork model = new MultiLayerNetwork(config);
            model.init();
        }
    }

    private void runModelConfigTest(String path) throws Exception {
        try (InputStream is = Resources.asStream(path)) {
            ComputationGraphConfiguration config = new KerasModel().modelBuilder().modelJsonInputStream(is).enforceTrainingConfig(true).buildModel().getComputationGraphConfiguration();
            ComputationGraph model = new ComputationGraph(config);
            model.init();
        }
    }
}

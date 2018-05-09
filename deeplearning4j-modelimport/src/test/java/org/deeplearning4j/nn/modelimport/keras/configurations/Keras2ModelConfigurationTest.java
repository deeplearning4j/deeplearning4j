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
package org.deeplearning4j.nn.modelimport.keras.configurations;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasModel;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasSpaceToDepth;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;


/**
 * Unit tests for Keras2 model configuration import.
 *
 * @author Max Pumperla
 */

@Slf4j
public class Keras2ModelConfigurationTest {

    ClassLoader classLoader = getClass().getClassLoader();

    @Test
    public void yolo9000ConfigTest() throws Exception {
        KerasLayer.registerCustomLayer("Lambda", KerasSpaceToDepth.class);
        runModelConfigTest("configs/keras2/yolo9000_tf_keras_2.json");
    }

    @Test
    public void l1l2RegularizerDenseTfConfigTest() throws Exception {
        runSequentialConfigTest("configs/keras2/l1l2_regularizer_dense_tf_keras_2_config.json");
    }

    @Test
    public void dgaClassifierTfConfigTest() throws Exception {
        runSequentialConfigTest("configs/keras2/keras2_dga_classifier_tf_config.json");
    }

    @Test
    public void convPooling1dTfConfigTest() throws Exception {
        runSequentialConfigTest("configs/keras2/keras2_conv1d_pooling1d_tf_config.json");
    }

    @Test
    public void bidirectionalLstmConfigTest() throws Exception {
        runSequentialConfigTest("configs/keras2/bidirectional_lstm_tf_keras_2_config.json");
    }

    @Test
    public void imdbLstmTfSequentialConfigTest() throws Exception {
        runSequentialConfigTest("configs/keras2/imdb_lstm_tf_keras_2_config.json");
    }

    @Test
    public void imdbLstmThSequentialConfigTest() throws Exception {
        runSequentialConfigTest("configs/keras2/imdb_lstm_th_keras_2_config.json");
    }

    @Test
    public void simpleRnnConfigTest() throws Exception {
        runSequentialConfigTest("configs/keras2/simple_rnn_tf_keras_2_config.json");
    }


    @Test
    public void mnistMlpTfSequentialConfigTest() throws Exception {
        runSequentialConfigTest("configs/keras2/mnist_mlp_tf_keras_2_config.json");
    }

    @Test
    public void mnistMlpThSequentialConfigTest() throws Exception {
        runSequentialConfigTest("configs/keras2/mnist_mlp_th_keras_2_config.json");
    }

    @Test
    public void mnistCnnTfSequentialConfigTest() throws Exception {
        runSequentialConfigTest("configs/keras2/mnist_cnn_tf_keras_2_config.json");
    }

    @Test
    public void mnistCnnThSequentialConfigTest() throws Exception {
        runSequentialConfigTest("configs/keras2/mnist_cnn_th_keras_2_config.json");
    }

    @Test
    public void mnistCnnNoBiasTfSequentialConfigTest() throws Exception {
        runSequentialConfigTest("configs/keras2/keras2_mnist_cnn_no_bias_tf_config.json");
    }


    @Test
    public void mlpSequentialConfigTest() throws Exception {
        runSequentialConfigTest("configs/keras2/keras2_mlp_config.json");
    }


    @Test
    public void mlpConstraintsConfigTest() throws Exception {
        runSequentialConfigTest("configs/keras2/mnist_mlp_constraint_tf_keras_2_config.json");
    }

    @Test
    public void embeddingFlattenThTest() throws Exception {
        runModelConfigTest("configs/keras2/embedding_flatten_graph_th_keras_2.json");
    }

    @Test
    public void mlpFapiConfigTest() throws Exception {
        runModelConfigTest("configs/keras2/keras2_mlp_fapi_config.json");
    }

    @Test
    public void mlpFapiMultiLossConfigTest() throws Exception {
        runModelConfigTest("configs/keras2/keras2_mlp_fapi_multiloss_config.json");
    }

    @Test
    public void cnnTfTest() throws Exception {
        runSequentialConfigTest("configs/keras2/keras2_cnn_tf_config.json");
    }

    @Test
    public void cnnThTest() throws Exception {
        runSequentialConfigTest("configs/keras2/keras2_cnn_th_config.json");
    }

    @Test
    public void mnistCnnTfTest() throws Exception {
        runSequentialConfigTest("configs/keras2/keras2_mnist_cnn_tf_config.json");
    }

    @Test
    public void mnistMlpTfTest() throws Exception {
        runSequentialConfigTest("configs/keras2/keras2_mnist_mlp_tf_config.json");
    }

    @Test
    public void embeddingConv1DTfTest() throws Exception {
        runSequentialConfigTest("configs/keras2/keras2_tf_embedding_conv1d_config.json");
    }

    @Test
    public void flattenConv1DTfTest() throws Exception {
        runSequentialConfigTest("configs/keras2/flatten_conv1d_tf_keras_2.json");
    }

    @Test
    public void embeddingLSTMMaskZeroTest() throws Exception {
        runModelConfigTest("configs/keras2/embedding_lstm_calculator.json");
    }

    @Test
    public void permuteRetinaUnet() throws Exception {
        runModelConfigTest("configs/keras2/permute_retina_unet.json");
    }


    @Test
    public void simpleAddLayerTest() throws Exception {
        runModelConfigTest("configs/keras2/simple_add_tf_keras_2.json");
    }

    @Test
    public void embeddingConcatTest() throws Exception {
        runModelConfigTest("/configs/keras2/model_concat_embedding_sequences_tf_keras_2.json");
    }

    @Test
    public void conv1dDilationTest() throws Exception {
        runModelConfigTest("/configs/keras2/conv1d_dilation_tf_keras_2_config.json");
    }


    private void runSequentialConfigTest(String path) throws Exception {
        ClassPathResource configResource = new ClassPathResource(path, classLoader);
        MultiLayerConfiguration config =
                new KerasModel().modelBuilder().modelJsonInputStream(configResource.getInputStream())
                        .enforceTrainingConfig(true).buildSequential().getMultiLayerConfiguration();
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
    }

    private void runModelConfigTest(String path) throws Exception {
        ClassPathResource configResource = new ClassPathResource(path, classLoader);
        ComputationGraphConfiguration config =
                new KerasModel().modelBuilder().modelJsonInputStream(configResource.getInputStream())
                        .enforceTrainingConfig(true).buildModel().getComputationGraphConfiguration();
        ComputationGraph model = new ComputationGraph(config);
        model.init();
    }
}

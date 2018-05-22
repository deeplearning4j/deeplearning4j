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
import org.deeplearning4j.nn.modelimport.keras.KerasModel;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;


/**
 * Unit tests for Keras1 model configuration import.
 *
 * @author Max Pumperla
 */

@Slf4j
public class Keras1ModelConfigurationTest {

    private ClassLoader classLoader = getClass().getClassLoader();

    @Test
    public void imdbLstmTfSequentialConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/imdb_lstm_tf_keras_1_config.json", true);
    }

    @Test
    public void imdbLstmThSequentialConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/imdb_lstm_th_keras_1_config.json", true);
    }

    @Test
    public void mnistMlpTfSequentialConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/mnist_mlp_tf_keras_1_config.json", true);
    }

    @Test
    public void mnistMlpThSequentialConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/mnist_mlp_th_keras_1_config.json", true);
    }

    @Test
    public void mnistCnnTfSequentialConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/mnist_cnn_tf_keras_1_config.json", true);
    }

    @Test
    public void mnistCnnNoBiasTfSequentialConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/mnist_cnn_no_bias_tf_config.json", true);
    }

    @Test
    public void mnistCnnThSequentialConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/mnist_cnn_th_keras_1_config.json", true);
    }

    @Test
    public void mlpSequentialConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/mlp_config.json", true);
    }

    @Test
    public void mlpConstraintsConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/mnist_mlp_constraint_tf_keras_1_config.json", true);
    }

    @Test
    public void reshapeMlpConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/mnist_mlp_reshape_tf_keras_1_config.json", true);
    }

    @Test
    public void reshapeCnnConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/mnist_cnn_reshape_tf_keras_1_config.json", true);
    }

    @Test
    public void mlpFapiConfigTest() throws Exception {
        runModelConfigTest("modelimport/keras/configs/keras1/mlp_fapi_config.json");
    }

    @Test
    public void mlpFapiMultiLossConfigTest() throws Exception {
        runModelConfigTest("modelimport/keras/configs/keras1/mlp_fapi_multiloss_config.json");
    }

    @Test
    public void yoloConfigTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/yolo_model.json", true);
    }

    @Test
    public void cnnTfTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/cnn_tf_config.json", true);
    }

    @Test
    public void cnnThTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/cnn_th_config.json", true);
    }

    @Test
    public void lstmFixedLenTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/lstm_tddense_config.json", false);
    }

    @Test
    public void mnistCnnTfTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/mnist_cnn_tf_config.json", true);
    }

    @Test
    public void mnistMlpTfTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/mnist_mlp_tf_config.json", true);
    }

    @Test
    public void embeddingConv1DTfTest() throws Exception {
        runSequentialConfigTest("modelimport/keras/configs/keras1/keras1_tf_embedding_conv1d_config.json", true);
    }

    private void runSequentialConfigTest(String path, boolean training) throws Exception {
        ClassPathResource configResource = new ClassPathResource(path, classLoader);
        MultiLayerConfiguration config =
                new KerasModel().modelBuilder().modelJsonInputStream(configResource.getInputStream())
                        .enforceTrainingConfig(training).buildSequential().getMultiLayerConfiguration();
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

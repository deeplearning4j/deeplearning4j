package org.deeplearning4j.nn.modelimport.keras;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
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

//    @Test
//    public void imdbLstmTfSequentialConfigTest() throws Exception {
//        runSequentialConfigTest("configs/keras2/imdb_lstm_tf_keras_2_config.json");
//    }
//
//    @Test
//    public void imdbLstmThSequentialConfigTest() throws Exception {
//        runSequentialConfigTest("configs/keras2/imdb_lstm_th_keras_2_config.json");
//    }

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

    void runSequentialConfigTest(String path) throws Exception {
        ClassPathResource configResource = new ClassPathResource(path, classLoader);
        MultiLayerConfiguration config =
                new KerasModel().modelBuilder().modelJsonInputStream(configResource.getInputStream())
                        .enforceTrainingConfig(true).buildSequential().getMultiLayerConfiguration();
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
    }

    void runModelConfigTest(String path) throws Exception {
        ClassPathResource configResource = new ClassPathResource(path, classLoader);
        ComputationGraphConfiguration config =
                new KerasModel().modelBuilder().modelJsonInputStream(configResource.getInputStream())
                        .enforceTrainingConfig(true).buildModel().getComputationGraphConfiguration();
        ComputationGraph model = new ComputationGraph(config);
        model.init();
    }
}

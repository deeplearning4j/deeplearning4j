package org.deeplearning4j.nn.modelimport.keras;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
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

//    @Test
//    public void imdbLstmTfSequentialConfigTest() throws Exception {
//        runSequentialConfigTest("configs/keras1/imdb_lstm_tf_keras_1_config.json");
//    }
//
//    @Test
//    public void imdbLstmThSequentialConfigTest() throws Exception {
//        runSequentialConfigTest("configs/keras1/imdb_lstm_th_keras_1_config.json");
//    }

    @Test
    public void mnistMlpTfSequentialConfigTest() throws Exception {
        runSequentialConfigTest("configs/keras1/mnist_mlp_tf_keras_1_config.json");
    }

    @Test
    public void mnistMlpThSequentialConfigTest() throws Exception {
        runSequentialConfigTest("configs/keras1/mnist_mlp_th_keras_1_config.json");
    }

    @Test
    public void mnistCnnTfSequentialConfigTest() throws Exception {
        runSequentialConfigTest("configs/keras1/mnist_cnn_tf_keras_1_config.json");
    }

    @Test
    public void mnistCnnThSequentialConfigTest() throws Exception {
        runSequentialConfigTest("configs/keras1/mnist_cnn_th_keras_1_config.json");
    }

    @Test
    public void mlpSequentialConfigTest() throws Exception {
        runSequentialConfigTest("configs/keras1/mlp_config.json");
    }

    @Test
    public void reshapeMlpConfigTest() throws Exception {
        runSequentialConfigTest("configs/keras1/mnist_mlp_reshape_tf_keras_1_config.json");
    }

    @Test
    public void reshapeCnnConfigTest() throws Exception {
        runSequentialConfigTest("configs/keras1/mnist_cnn_reshape_tf_keras_1_config.json");
    }

    @Test
    public void mlpFapiConfigTest() throws Exception {
        runModelConfigTest("configs/keras1/mlp_fapi_config.json");
    }

    @Test
    public void mlpFapiMultiLossConfigTest() throws Exception {
        runModelConfigTest("configs/keras1/mlp_fapi_multiloss_config.json");
    }

    @Test
    public void yoloConfigTest() throws Exception {
        runSequentialConfigTest("configs/keras1/yolo_model.json");
    }

    @Test
    public void cnnTfTest() throws Exception {
        runSequentialConfigTest("configs/keras1/cnn_tf_config.json");
    }

    @Test
    public void cnnThTest() throws Exception {
        runSequentialConfigTest("configs/keras1/cnn_th_config.json");
    }

    @Test
    public void lstmFixedLenTest() throws Exception {
        runSequentialConfigTest("configs/keras1/lstm_tddense_config.json");
    }

    @Test
    public void mnistCnnTfTest() throws Exception {
        runSequentialConfigTest("configs/keras1/mnist_cnn_tf_config.json");
    }

    @Test
    public void mnistMlpTfTest() throws Exception {
        runSequentialConfigTest("configs/keras1/mnist_mlp_tf_config.json");
    }

    void runSequentialConfigTest(String path) throws Exception {
        ClassPathResource configResource = new ClassPathResource(path, classLoader);
        MultiLayerConfiguration config =
                new KerasModel.ModelBuilder().modelJsonInputStream(configResource.getInputStream())
                        .enforceTrainingConfig(true).buildSequential().getMultiLayerConfiguration();
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
    }

    void runModelConfigTest(String path) throws Exception {
        ClassPathResource configResource = new ClassPathResource(path, classLoader);
        ComputationGraphConfiguration config =
                new KerasModel.ModelBuilder().modelJsonInputStream(configResource.getInputStream())
                        .enforceTrainingConfig(true).buildModel().getComputationGraphConfiguration();
        ComputationGraph model = new ComputationGraph(config);
        model.init();
    }
}

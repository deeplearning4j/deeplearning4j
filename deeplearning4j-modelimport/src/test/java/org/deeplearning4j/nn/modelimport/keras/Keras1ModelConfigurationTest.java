package org.deeplearning4j.nn.modelimport.keras;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;


/**
 * Unit tests for Keras model configuration import.
 *
 * @author dave@skymind.io
 */

@Slf4j
public class KerasModelConfigurationTest {

    ClassLoader classLoader = getClass().getClassLoader();

    @Test
    public void importKerasMlpSequentialConfigTest() throws Exception {
        ClassPathResource configResource = new ClassPathResource(
                "configs/keras1/mlp_config.json", classLoader);
        MultiLayerConfiguration config =
                new KerasModel.ModelBuilder().modelJsonInputStream(configResource.getInputStream())
                        .enforceTrainingConfig(true).buildSequential().getMultiLayerConfiguration();
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
    }

    @Test
    public void importKerasMlpModelConfigTest() throws Exception {
        ClassPathResource configResource = new ClassPathResource(
                "configs/keras1/mlp_fapi_config.json", classLoader);
        ComputationGraphConfiguration config =
                new KerasModel.ModelBuilder().modelJsonInputStream(configResource.getInputStream())
                        .enforceTrainingConfig(true).buildModel().getComputationGraphConfiguration();
        ComputationGraph model = new ComputationGraph(config);
        model.init();
    }

    @Test
    public void importKerasMlpModelMultilossConfigTest() throws Exception {
        ClassPathResource configResource = new ClassPathResource(
                "configs/keras1/mlp_fapi_multiloss_config.json", classLoader);
        ComputationGraphConfiguration config =
                new KerasModel.ModelBuilder().modelJsonInputStream(configResource.getInputStream())
                        .enforceTrainingConfig(true).buildModel().getComputationGraphConfiguration();
        ComputationGraph model = new ComputationGraph(config);
        model.init();
    }

    @Test
    public void importKerasConvnetTensorflowConfigTest() throws Exception {
        ClassPathResource configResource = new ClassPathResource(
                "configs/keras1/cnn_tf_config.json", classLoader);
        MultiLayerConfiguration config =
                new KerasModel.ModelBuilder().modelJsonInputStream(configResource.getInputStream())
                        .enforceTrainingConfig(true).buildSequential().getMultiLayerConfiguration();
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
    }

    @Test
    public void importKerasConvnetTheanoConfigTest() throws Exception {
        ClassPathResource configResource = new ClassPathResource(
                "configs/keras1/cnn_th_config.json", classLoader);
        MultiLayerConfiguration config =
                new KerasModel.ModelBuilder().modelJsonInputStream(configResource.getInputStream())
                        .enforceTrainingConfig(true).buildSequential().getMultiLayerConfiguration();
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
    }

    @Test
    public void importKerasLstmFixedLenConfigTest() throws Exception {
        ClassPathResource configResource = new ClassPathResource(
                "configs/keras1/lstm_tddense_config.json", classLoader);
        MultiLayerConfiguration config =
                new KerasModel.ModelBuilder().modelJsonInputStream(configResource.getInputStream())
                        .enforceTrainingConfig(true).buildSequential().getMultiLayerConfiguration();
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
    }

    @Test
    public void importMnistCnnTensorFlowConfigurationTest() throws Exception {
        ClassPathResource configResource = new ClassPathResource(
                "configs/keras1/mnist_cnn_tf_config.json", classLoader);
        MultiLayerConfiguration config =
                new KerasModel.ModelBuilder().modelJsonInputStream(configResource.getInputStream())
                        .enforceTrainingConfig(true).buildSequential().getMultiLayerConfiguration();
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
    }

    @Test
    public void importMnistMlpTensorFlowConfigurationTest() throws Exception {
        ClassPathResource configResource = new ClassPathResource(
                "configs/keras1/mnist_mlp_tf_config.json", classLoader);
        MultiLayerConfiguration config =
                new KerasModel.ModelBuilder().modelJsonInputStream(configResource.getInputStream())
                        .enforceTrainingConfig(true).buildSequential().getMultiLayerConfiguration();
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
    }
}

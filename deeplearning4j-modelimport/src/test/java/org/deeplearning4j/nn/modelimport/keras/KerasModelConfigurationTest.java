package org.deeplearning4j.nn.modelimport.keras;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

/**
 * Unit tests for Keras model configuration import.
 *
 * @author dave@skymind.io
 */

@Slf4j
public class KerasModelConfigurationTest {

    @Test
    public void importKerasMlpSequentialConfigTest() throws Exception {
        ClassPathResource configResource = new ClassPathResource("modelimport/keras/configs/mlp_config.json",
                        KerasModelConfigurationTest.class.getClassLoader());
        MultiLayerConfiguration config =
                        new KerasModel.ModelBuilder().modelJsonInputStream(configResource.getInputStream())
                                        .enforceTrainingConfig(true).buildSequential().getMultiLayerConfiguration();
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
    }

    @Test
    public void importKerasMlpModelConfigTest() throws Exception {
        ClassPathResource configResource = new ClassPathResource("modelimport/keras/configs/mlp_fapi_config.json",
                        KerasModelConfigurationTest.class.getClassLoader());
        ComputationGraphConfiguration config =
                        new KerasModel.ModelBuilder().modelJsonInputStream(configResource.getInputStream())
                                        .enforceTrainingConfig(true).buildModel().getComputationGraphConfiguration();
        ComputationGraph model = new ComputationGraph(config);
        model.init();
    }

    @Test
    public void importKerasMlpModelMultilossConfigTest() throws Exception {
        ClassPathResource configResource =
                        new ClassPathResource("modelimport/keras/configs/mlp_fapi_multiloss_config.json",
                                        KerasModelConfigurationTest.class.getClassLoader());
        ComputationGraphConfiguration config =
                        new KerasModel.ModelBuilder().modelJsonInputStream(configResource.getInputStream())
                                        .enforceTrainingConfig(true).buildModel().getComputationGraphConfiguration();
        ComputationGraph model = new ComputationGraph(config);
        model.init();
    }

    @Test
    public void importKerasConvnetTensorflowConfigTest() throws Exception {
        ClassPathResource configResource = new ClassPathResource("modelimport/keras/configs/cnn_tf_config.json",
                        KerasModelConfigurationTest.class.getClassLoader());
        MultiLayerConfiguration config =
                        new KerasModel.ModelBuilder().modelJsonInputStream(configResource.getInputStream())
                                        .enforceTrainingConfig(true).buildSequential().getMultiLayerConfiguration();
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
    }

    @Test
    public void importKerasConvnetTheanoConfigTest() throws Exception {
        ClassPathResource configResource = new ClassPathResource("modelimport/keras/configs/cnn_th_config.json",
                        KerasModelConfigurationTest.class.getClassLoader());
        MultiLayerConfiguration config =
                        new KerasModel.ModelBuilder().modelJsonInputStream(configResource.getInputStream())
                                        .enforceTrainingConfig(true).buildSequential().getMultiLayerConfiguration();
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
    }

    @Test
    public void importKerasLstmFixedLenConfigTest() throws Exception {
        ClassPathResource configResource = new ClassPathResource("modelimport/keras/configs/lstm_fixed_config.json",
                        KerasModelConfigurationTest.class.getClassLoader());
        MultiLayerConfiguration config =
                        new KerasModel.ModelBuilder().modelJsonInputStream(configResource.getInputStream())
                                        .enforceTrainingConfig(true).buildSequential().getMultiLayerConfiguration();
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
    }

    @Test
    public void importMnistCnnTensorFlowConfigurationTest() throws Exception {
        ClassPathResource configResource =
                        new ClassPathResource("modelimport/keras/examples/mnist_cnn/mnist_cnn_tf_config.json",
                                        KerasModelConfigurationTest.class.getClassLoader());
        MultiLayerConfiguration config =
                        new KerasModel.ModelBuilder().modelJsonInputStream(configResource.getInputStream())
                                        .enforceTrainingConfig(true).buildSequential().getMultiLayerConfiguration();
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
    }
}

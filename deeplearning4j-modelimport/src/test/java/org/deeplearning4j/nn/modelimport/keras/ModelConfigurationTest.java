package org.deeplearning4j.nn.modelimport.keras;

import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;

import static org.deeplearning4j.nn.modelimport.keras.ModelConfiguration.importSequentialModelConfigFromFile;

/**
 * Unit tests for end-to-end Keras model configuration import.
 *
 * @author davekale
 */
public class ModelConfigurationTest {

    @Test
    public void importKerasMlpConfigTest() throws Exception {
        ClassPathResource resource = new ClassPathResource("keras/config/mlp_config.json",
                ModelConfigurationTest.class.getClassLoader());
        String configFilename = resource.getFile().getAbsolutePath();
        importKerasConfigTest(configFilename);
    }

    @Test
    public void importKerasConvnetTensorflowConfigTest() throws Exception {
        ClassPathResource resource = new ClassPathResource("keras/config/cnn_tf_config.json",
                ModelConfigurationTest.class.getClassLoader());
        String configFilename = resource.getFile().getAbsolutePath();
        importKerasConfigTest(configFilename);
    }

    @Test
    public void importKerasConvnetTheanoConfigTest() throws Exception {
        ClassPathResource resource = new ClassPathResource("keras/config/cnn_th_config.json",
                ModelConfigurationTest.class.getClassLoader());
        String configFilename = resource.getFile().getAbsolutePath();
        importKerasConfigTest(configFilename);
    }

    @Test
    public void importKerasLstmFixedLenConfigTest() throws Exception {
        ClassPathResource resource = new ClassPathResource("keras/config/lstm_fixed_config.json",
                ModelConfigurationTest.class.getClassLoader());
        String configFilename = resource.getFile().getAbsolutePath();
        importKerasConfigTest(configFilename);
    }

    public static void importKerasConfigTest(String configFilename) throws Exception {
        importSequentialModelConfigFromFile(configFilename);
    }
}

package org.deeplearning4j.nn.modelimport.keras;

import org.junit.Test;

import static org.deeplearning4j.nn.modelimport.keras.ModelConfiguration.importSequentialModelConfigFromFile;

/**
 * Created by davekale on 11/11/16.
 */
public class ModelConfigurationTest {

    @Test
    public void importKerasMlpConfigTest() throws Exception {
        ClassLoader classLoader = getClass().getClassLoader();
        String configFilename = classLoader.getResource("keras/config/mlp_config.json").getPath();
        importKerasConfigTest(configFilename);
    }

    @Test
    public void importKerasConvnetTensorflowConfigTest() throws Exception {
        ClassLoader classLoader = getClass().getClassLoader();
        String configFilename = classLoader.getResource("keras/config/cnn_tf_config.json").getPath();
        importKerasConfigTest(configFilename);
    }

    @Test
    public void importKerasConvnetTheanoConfigTest() throws Exception {
        ClassLoader classLoader = getClass().getClassLoader();
        String configFilename = classLoader.getResource("keras/config/cnn_th_config.json").getPath();
        importKerasConfigTest(configFilename);
    }

    @Test
    public void importKerasLstmFixedLenConfigTest() throws Exception {
        ClassLoader classLoader = getClass().getClassLoader();
        String configFilename = classLoader.getResource("keras/config/lstm_fixed_config.json").getPath();
        importKerasConfigTest(configFilename);
    }

    public static void importKerasConfigTest(String configFilename) throws Exception {
        importSequentialModelConfigFromFile(configFilename);
    }
}

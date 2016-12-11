package org.deeplearning4j.nn.modelimport.keras;

import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;

import static org.deeplearning4j.nn.modelimport.keras.ModelConfiguration.importSequentialModelConfigFromJsonFile;

/**
 * Unit tests for end-to-end Keras model configuration import.
 *
 * @author davekale
 */
public class ModelConfigurationTest {

    @Test
    public void importKerasMlpConfigTest() throws Exception {
        ClassPathResource resource = new ClassPathResource("keras/simple/mlp_config.json",
                ModelConfigurationTest.class.getClassLoader());
        String configFilename = resource.getFile().getAbsolutePath();
        importKerasConfigTest(configFilename);
    }

    @Test
    public void importKerasConvnetTensorflowConfigTest() throws Exception {
        ClassPathResource resource = new ClassPathResource("keras/simple/cnn_tf_config.json",
                ModelConfigurationTest.class.getClassLoader());
        String configFilename = resource.getFile().getAbsolutePath();
        importKerasConfigTest(configFilename);
    }

//    @Test
//    public void importKerasConvnetTheanoConfigTest() throws Exception {
//        ClassPathResource resource = new ClassPathResource("keras/simple/cnn_th_config.json",
//                ModelConfigurationTest.class.getClassLoader());
//        String configFilename = resource.getFile().getAbsolutePath();
//        importKerasConfigTest(configFilename);
//    }

    @Test
    public void importKerasLstmFixedLenConfigTest() throws Exception {
        ClassPathResource resource = new ClassPathResource("keras/simple/lstm_fixed_config.json",
                ModelConfigurationTest.class.getClassLoader());
        String configFilename = resource.getFile().getAbsolutePath();
        importKerasConfigTest(configFilename);
    }

//    @Test
//    public void dudeFixed() throws Exception {
//        ClassPathResource resource = new ClassPathResource("keras/simple/dude.json",
//                ModelConfigurationTest.class.getClassLoader());
//        String configFilename = resource.getFile().getAbsolutePath();
//        importKerasConfigTest(configFilename);
//    }

    public static void importKerasConfigTest(String configFilename) throws Exception {
        importSequentialModelConfigFromJsonFile(configFilename);
    }
//
//    @Test
//    public void importKerasInceptionV3ConfigTest() throws Exception {
//        ClassPathResource resource = new ClassPathResource("keras/simple/inception_v3_config.json",
//                ModelConfigurationTest.class.getClassLoader());
//        String configFilename = resource.getFile().getAbsolutePath();
//        importFunctionalApiConfigFromFile(configFilename);
//    }
}

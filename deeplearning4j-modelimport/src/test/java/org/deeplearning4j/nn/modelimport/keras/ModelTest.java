package org.deeplearning4j.nn.modelimport.keras;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.FileInputStream;

import static org.junit.Assert.*;
import static org.deeplearning4j.nn.modelimport.keras.Model.importSequentialModel;
import static org.deeplearning4j.nn.modelimport.keras.Model.importSequentialModelInputStream;

/**
 * Unit tests for Keras model configuration import.
 *
 * TODO: Replace deprecated stuff and rename to something like KerasModelTest
 * TODO: Move test resources to dl4j-test-resources
 * TODO: Reorganize test resources
 * TODO: Add more extensive tests including exotic Functional API architectures
 *
 * @author dave@skymind.io
 */
@Slf4j
public class ModelTest {

    @Test(expected=UnsupportedOperationException.class)
    public void CnnModelImportInputStreamTest() throws Exception {
        String modelPath = new ClassPathResource("keras/simple/cnn_tf_model.h5",
                ModelConfigurationTest.class.getClassLoader()).getFile().getAbsolutePath();
        FileInputStream inputStream = new FileInputStream(modelPath);
        MultiLayerNetwork model = importSequentialModelInputStream(inputStream);
        CnnModelTest(model);
    }

    @Test
    public void CnnModelImportTest() throws Exception {
        String modelPath = new ClassPathResource("keras/simple/cnn_tf_model.h5",
                ModelConfigurationTest.class.getClassLoader()).getFile().getAbsolutePath();
        MultiLayerNetwork model = importSequentialModel(modelPath);
        CnnModelTest(model);
    }

    @Test
    public void CnnConfigAndWeightsImportTest() throws Exception {
        String configPath = new ClassPathResource("keras/simple/cnn_tf_config.json",
                ModelConfigurationTest.class.getClassLoader()).getFile().getAbsolutePath();
        String weightsPath = new ClassPathResource("keras/simple/cnn_tf_weights.h5",
                ModelConfigurationTest.class.getClassLoader()).getFile().getAbsolutePath();
        MultiLayerNetwork model = importSequentialModel(configPath, weightsPath);
        CnnModelTest(model);
    }

    public void CnnModelTest(MultiLayerNetwork model) throws Exception {
        // Test input
        String inputPath = new ClassPathResource("keras/simple/cnn_tf_input.txt",
                ModelConfigurationTest.class.getClassLoader()).getFile().getAbsolutePath();
        INDArray input = Nd4j.readNumpy(inputPath, " ").reshape(1, 3, 8, 8);

        // Test outputs
        String outputPath = new ClassPathResource("keras/simple/cnn_tf_output.txt",
                ModelConfigurationTest.class.getClassLoader()).getFile().getAbsolutePath();
        INDArray outputTrue = Nd4j.readNumpy(outputPath, " ");

        // Make predictions
        INDArray outputPredicted = model.output(input, false);

        // Compare predictions to outputs
        assertEquals(outputTrue, outputPredicted);
    }

    @Test
    public void LstmModelImportTest() throws Exception {
        String modelPath = new ClassPathResource("keras/simple/lstm_tf_model.h5",
                ModelConfigurationTest.class.getClassLoader()).getFile().getAbsolutePath();
        MultiLayerNetwork model = importSequentialModel(modelPath);
    }
}

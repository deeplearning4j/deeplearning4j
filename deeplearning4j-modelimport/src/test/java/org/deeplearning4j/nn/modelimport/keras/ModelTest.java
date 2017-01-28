package org.deeplearning4j.nn.modelimport.keras;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.io.Resource;

import java.io.File;
import java.io.FileInputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

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
        ClassPathResource resource = new ClassPathResource("keras/simple/cnn_tf_model.h5",
                ModelConfigurationTest.class.getClassLoader());
        MultiLayerNetwork model = importSequentialModelInputStream(resource.getInputStream());
        CnnModelTest(model);
    }

    @Test
    public void CnnModelImportTest() throws Exception {
        ClassPathResource resource = new ClassPathResource("keras/simple/cnn_tf_model.h5",
                ModelConfigurationTest.class.getClassLoader());
        File file = File.createTempFile("tempModel", ".h5");
        Files.copy(resource.getInputStream(), file.toPath(),  StandardCopyOption.REPLACE_EXISTING);
        String s = file.getAbsolutePath();
        MultiLayerNetwork model = importSequentialModel(file.getAbsolutePath());
        CnnModelTest(model);
    }

    @Test
    public void CnnConfigAndWeightsImportTest() throws Exception {
        File configFile = new ClassPathResource("keras/simple/cnn_tf_config.json",
                ModelConfigurationTest.class.getClassLoader()).getTempFileFromArchive();
        configFile.deleteOnExit();
        File weightsFile = new ClassPathResource("keras/simple/cnn_tf_weights.h5",
                ModelConfigurationTest.class.getClassLoader()).getTempFileFromArchive();
        weightsFile.deleteOnExit();
        MultiLayerNetwork model = importSequentialModel(configFile.getAbsolutePath(), weightsFile.getAbsolutePath());
        CnnModelTest(model);
    }

    public void CnnModelTest(MultiLayerNetwork model) throws Exception {
        // Test input
        Resource resource = new ClassPathResource("keras/simple/cnn_tf_input.txt",
                ModelConfigurationTest.class.getClassLoader());
        INDArray input = Nd4j.readNumpy(resource.getInputStream(), " ").reshape(1, 3, 8, 8);

        // Test outputs
        resource = new ClassPathResource("keras/simple/cnn_tf_output.txt",
                ModelConfigurationTest.class.getClassLoader());
        INDArray outputTrue = Nd4j.readNumpy(resource.getInputStream(), " ");

        // Make predictions
        INDArray outputPredicted = model.output(input, false);

        // Compare predictions to outputs
        assertEquals(outputTrue, outputPredicted);
    }
}

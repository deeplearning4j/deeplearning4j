package org.deeplearning4j.nn.modelimport.keras;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import java.util.List;

import static org.junit.Assert.*;
import static org.deeplearning4j.nn.modelimport.keras.Model.importModel;
import static org.deeplearning4j.nn.modelimport.keras.Model.importSequentialModel;

/**
 * Created by davekale on 11/14/16.
 */
public class ModelTest {
    private static Logger log = LoggerFactory.getLogger(ModelTest.class);

    @Test
    public void SimpleModelImportTest() throws Exception {
        String modelFn = "/Users/davekale/skymind/models/keras/mlp_model.h5";
        MultiLayerNetwork model = importSequentialModel(modelFn);
        System.out.println("DONE!");
    }

    @Test
    public void CnnModelImportTest() throws Exception {
        String modelPath = new ClassPathResource("keras/simple/cnn_tf_model.h5",
                ModelConfigurationTest.class.getClassLoader()).getFile().getAbsolutePath();
        MultiLayerNetwork model = (MultiLayerNetwork)importModel(modelPath);
        CnnModelTest(model);
    }

    @Test
    public void CnnConfigAndWeightsImportTest() throws Exception {
        String configPath = new ClassPathResource("keras/simple/cnn_tf_config.json",
                ModelConfigurationTest.class.getClassLoader()).getFile().getAbsolutePath();
        String weightsPath = new ClassPathResource("keras/simple/cnn_tf_weights.h5",
                ModelConfigurationTest.class.getClassLoader()).getFile().getAbsolutePath();
        MultiLayerNetwork model = (MultiLayerNetwork)importModel(configPath, weightsPath);
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
        List<INDArray> activations = model.feedForward(input, false);
        INDArray outputPredicted = activations.get(activations.size()-2);
        /* TODO: fix the above once we better understand the proper way to get
         * a prediction from the last layer (since model.output(*) gives the loss
         * when the last layer is a LossLayer.
         */
//        INDArray outputPredicted = model.output(input, false);

        // Compare predictions to outputs
        assertEquals(outputTrue, outputPredicted);
    }
}

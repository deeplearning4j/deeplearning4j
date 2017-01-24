package org.deeplearning4j.keras.model;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.keras.model.KerasModelRef;
import org.deeplearning4j.keras.model.KerasModelType;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.io.IOException;

/**
 * Reads the neural network model from Keras, specified by the parameters. Reuses the -modelimport code.
 *
 * @author pkoperek@gmail.com
 */
@Slf4j
public class KerasModelSerializer {

    public MultiLayerNetwork read(String modelFilePath, String modelType)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {

        MultiLayerNetwork multiLayerNetwork;
        if (KerasModelType.SEQUENTIAL.equals(modelType)) {
            multiLayerNetwork = KerasModelImport.importKerasSequentialModelAndWeights(modelFilePath);
            multiLayerNetwork.init();
        } else {
            throw new RuntimeException("Model type unsupported! (" + modelType + ")");
        }

        return multiLayerNetwork;
    }

    // TODO write method that writes back to keras format
}

package org.deeplearning4j.keras.model;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;

import java.io.IOException;

/**
 * Reads the neural network model from Keras, specified by the parameters. Reuses the -modelimport code.
 *
 * @author pkoperek@gmail.com
 */
@Slf4j
public class KerasModelSerializer {

    public ComputationGraph read(String modelFilePath, KerasModelType modelType)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {

        ComputationGraph model;
        if (KerasModelType.SEQUENTIAL.equals(modelType)) {
            model = KerasModelImport.importKerasModelAndWeights(modelFilePath);
            model.init();
        } else {
            throw new RuntimeException("Model type unsupported! (" + modelType + ")");
        }

        return model;
    }

    // TODO write method that writes back to keras format
}

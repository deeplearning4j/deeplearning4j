package org.deeplearning4j.keras;

import lombok.extern.slf4j.Slf4j;
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
public class NeuralNetworkReader {

    public MultiLayerNetwork readNeuralNetwork(EntryPointFitParameters entryPointFitParameters)
                    throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {

        MultiLayerNetwork multiLayerNetwork;
        if (KerasModelType.SEQUENTIAL.equals(entryPointFitParameters.getType())) {
            multiLayerNetwork = KerasModelImport
                            .importKerasSequentialModelAndWeights(entryPointFitParameters.getModelFilePath());
            multiLayerNetwork.init();
        } else {
            throw new RuntimeException("Model type unsupported! (" + entryPointFitParameters.getType() + ")");
        }

        return multiLayerNetwork;
    }
}

package org.deeplearning4j.keras.model;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.keras.model.KerasModelType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;

/**
 * Reads the neural network model from Keras, specified by the parameters. Reuses the -modelimport code.
 *
 * @author pkoperek@gmail.com
 */
@Slf4j
public class KerasModelSerializer {

    public MultiLayerNetwork readSequential(String modelFilePath, KerasModelType modelType)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {

        MultiLayerNetwork model;
        if (KerasModelType.SEQUENTIAL.equals(modelType)) {
            model = KerasModelImport.importKerasSequentialModelAndWeights(modelFilePath);
            model.init();
        } else {
            throw new RuntimeException("Model type unsupported! (" + modelType + ") Did you mean to use .readFunctional()?");
        }

        return model;
    }

    public ComputationGraph readFunctional(String modelFilePath, KerasModelType modelType)
        throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {

        ComputationGraph model;
        if (KerasModelType.FUNCTIONAL.equals(modelType)) {
            model = KerasModelImport.importKerasModelAndWeights(modelFilePath);
            model.init();
        } else {
            throw new RuntimeException("Model type unsupported! (" + modelType + ") Did you mean to use .readSequential()?");
        }

        return model;
    }

    // TODO write method that writes back to keras format
}

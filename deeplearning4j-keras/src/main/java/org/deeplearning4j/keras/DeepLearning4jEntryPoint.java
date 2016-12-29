package org.deeplearning4j.keras;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * API exposed to the Python side. This class contains methods which are used by the python wrapper.
 * It is instantiated directly in the server code.
 */
@Slf4j
public class DeepLearning4jEntryPoint {

    private final NeuralNetworkReader neuralNetworkReader = new NeuralNetworkReader();
    private final NDArrayHDF5Reader ndArrayHDF5Reader = new NDArrayHDF5Reader();
    private final NeuralNetworkBatchLearner neuralNetworkBatchLearner = new NeuralNetworkBatchLearner();

    /**
     * Performs fitting of the model which is referenced in the parameters according to learning parameters specified.
     *
     * @param entryPointFitParameters Definition of the model and learning process
     */
    public void fit(EntryPointFitParameters entryPointFitParameters) throws Exception {

        try {
            MultiLayerNetwork multiLayerNetwork = neuralNetworkReader.readNeuralNetwork(entryPointFitParameters);

            INDArray features = ndArrayHDF5Reader.readFromPath(entryPointFitParameters.getTrainFeaturesFile());
            INDArray labels = ndArrayHDF5Reader.readFromPath(entryPointFitParameters.getTrainLabelsFile());
            int batchSize = entryPointFitParameters.getBatchSize();

            for (int i = 0; i < entryPointFitParameters.getNbEpoch(); i++) {
                log.info("Fitting: " + i);

                neuralNetworkBatchLearner.fitInBatches(multiLayerNetwork, features, labels, batchSize);
            }

            log.info("Learning model finished");
        } catch (Throwable e) {
            log.error("Error while handling request!", e);
            throw e;
        }
    }

}

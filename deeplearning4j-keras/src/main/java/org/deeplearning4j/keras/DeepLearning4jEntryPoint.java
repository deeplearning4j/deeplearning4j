package org.deeplearning4j.keras;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * API exposed to the Python side. This class contains methods which are used by the python wrapper.
 * It is instantiated directly in the server code.
 */
@Slf4j
public class DeepLearning4jEntryPoint {

    private final NeuralNetworkReader neuralNetworkReader = new NeuralNetworkReader();

    /**
     * Performs fitting of the model which is referenced in the parameters according to learning parameters specified.
     *
     * @param entryPointFitParameters Definition of the model and learning process
     */
    public void fit(EntryPointFitParameters entryPointFitParameters) throws Exception {

        try {
            MultiLayerNetwork multiLayerNetwork = neuralNetworkReader.readNeuralNetwork(entryPointFitParameters);

            DataSetIterator dataSetIterator = new HDF5MiniBatchDataSetIterator(
                    entryPointFitParameters.getTrainFeaturesDirectory(),
                    entryPointFitParameters.getTrainLabelsDirectory()
            );

            for (int i = 0; i < entryPointFitParameters.getNbEpoch(); i++) {
                log.info("Fitting: " + i);

                multiLayerNetwork.fit(dataSetIterator);
            }

            log.info("Learning model finished");
        } catch (Throwable e) {
            log.error("Error while handling request!", e);
            throw e;
        }
    }

}

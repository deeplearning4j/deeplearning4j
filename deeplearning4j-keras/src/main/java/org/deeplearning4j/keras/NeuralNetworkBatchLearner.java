package org.deeplearning4j.keras;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

@Slf4j
public class NeuralNetworkBatchLearner {

    private INDArrayIndex[] createSlicingIndexes(int length) {
        INDArrayIndex[] ndIndexes = new INDArrayIndex[length];
        for (int i = 0; i < ndIndexes.length; i++) {
            ndIndexes[i] = NDArrayIndex.all();
        }
        return ndIndexes;
    }

    public void fitInBatches(MultiLayerNetwork multiLayerNetwork, INDArray features, INDArray labels, int batchSize) {
        final INDArrayIndex[] ndIndexes = createSlicingIndexes(features.shape().length);

        int begin = 0;
        int samplesCount = features.size(0);

        while (begin < samplesCount) {
            int end = begin + batchSize;

            if (log.isTraceEnabled()) {
                log.trace("Processing batch: " + begin + " " + end);
            }

            ndIndexes[0] = NDArrayIndex.interval(begin, end);
            INDArray featuresBatch = features.get(ndIndexes);
            INDArray labelsBatch = labels.get(NDArrayIndex.interval(begin, end));
            multiLayerNetwork.fit(featuresBatch, labelsBatch);

            begin += batchSize;
        }
    }
}

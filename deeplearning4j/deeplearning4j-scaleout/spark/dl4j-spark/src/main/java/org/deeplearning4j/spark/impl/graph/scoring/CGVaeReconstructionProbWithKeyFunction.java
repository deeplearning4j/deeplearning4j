package org.deeplearning4j.spark.impl.graph.scoring;

import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.spark.impl.common.score.BaseVaeReconstructionProbWithKeyFunctionAdapter;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Function to calculate the reconstruction probability for a variational autoencoder, that is the first layer in a
 * ComputationGraph.<br>
 * Note that scoring is batched for computational efficiency.<br>
 *
 * @author Alex Black
 */
public class CGVaeReconstructionProbWithKeyFunction<K> extends BaseVaeReconstructionProbWithKeyFunctionAdapter<K> {


    /**
     * @param params            MultiLayerNetwork parameters
     * @param jsonConfig        MultiLayerConfiguration, as json
     * @param useLogProbability If true: use log probability. False: use raw probability.
     * @param batchSize         Batch size to use when scoring
     * @param numSamples        Number of samples to use when calling {@link VariationalAutoencoder#reconstructionLogProbability(INDArray, int)}
     */
    public CGVaeReconstructionProbWithKeyFunction(Broadcast<INDArray> params, Broadcast<String> jsonConfig,
                    boolean useLogProbability, int batchSize, int numSamples) {
        super(params, jsonConfig, useLogProbability, batchSize, numSamples);
    }

    @Override
    public VariationalAutoencoder getVaeLayer() {
        ComputationGraph network =
                        new ComputationGraph(ComputationGraphConfiguration.fromJson((String) jsonConfig.getValue()));
        network.init();
        INDArray val = ((INDArray) params.value()).unsafeDuplication();
        if (val.length() != network.numParams(false))
            throw new IllegalStateException(
                            "Network did not have same number of parameters as the broadcasted set parameters");
        network.setParams(val);

        Layer l = network.getLayer(0);
        if (!(l instanceof VariationalAutoencoder)) {
            throw new RuntimeException(
                            "Cannot use CGVaeReconstructionProbWithKeyFunction on network that doesn't have a VAE "
                                            + "layer as layer 0. Layer type: " + l.getClass());
        }
        return (VariationalAutoencoder) l;
    }
}

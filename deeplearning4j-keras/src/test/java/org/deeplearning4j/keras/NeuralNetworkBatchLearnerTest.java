package org.deeplearning4j.keras;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

public class NeuralNetworkBatchLearnerTest {

    @Rule
    public ExpectedException thrown = ExpectedException.none();

    private final double[] sampleOnes = new double[]{1.0, 1.0, 1.0, 1.0};
    private final double[] sampleTwos = new double[]{2.0, 2.0, 2.0, 2.0};
    private final double[] one = new double[]{1.0};
    private final double[] two = new double[]{2.0};
    private final double[] oneAndTwo = new double[]{1.0, 2.0};

    private NeuralNetworkBatchLearner neuralNetworkBatchLearner = new NeuralNetworkBatchLearner();

    @Test
    public void shouldNotFailToFitTheModelWithSingleSample() throws Exception {
        // Given
        MultiLayerNetwork network = prepareNetwork();

        INDArray features = Nd4j.create(sampleOnes);
        INDArray labels = Nd4j.create(one);
        int batchSize = 1;

        // When
        neuralNetworkBatchLearner.fitInBatches(network, features, labels, batchSize);

        // Then
        // fall through if no exception thrown
    }

    @Test
    public void shouldFitTheModelUsingBatches() throws Exception {
        // Given
        MultiLayerNetwork network = mock(MultiLayerNetwork.class);

        INDArray features = Nd4j.create(new double[][]{sampleOnes, sampleTwos});
        INDArray labels = Nd4j.create(oneAndTwo);
        int batchSize = 1;

        // When
        neuralNetworkBatchLearner.fitInBatches(network, features, labels, batchSize);

        // Then
        verify(network).fit(Nd4j.create(sampleOnes), Nd4j.create(one));
        verify(network).fit(Nd4j.create(sampleTwos), Nd4j.create(two));
    }

    private MultiLayerNetwork prepareNetwork() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(4).nOut(1)
                        .build())
                .layer(1, new LossLayer.Builder().build())
                .build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        return network;
    }

}
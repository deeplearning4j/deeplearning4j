package org.deeplearning4j.ui.flow;

import org.canova.image.loader.LFWLoader;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.flow.beans.LayerInfo;
import org.deeplearning4j.ui.flow.beans.ModelInfo;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

/**
 * This set of tests addresses different stages of model state serialization for later visualization
 *
 * @author raver119@gmail.com
 */
public class FlowIterationListenerTest {
    private static ComputationGraph graph;
    private static MultiLayerNetwork network;

    private static Logger log = LoggerFactory.getLogger(FlowIterationListenerTest.class);

    @Before
    public void setUp() throws Exception {
        if (graph == null) {

        }

        if (network == null) {
            final int numRows = 40;
            final int numColumns = 40;
            int nChannels = 3;
            int outputNum = LFWLoader.NUM_LABELS;
            int numSamples = LFWLoader.NUM_IMAGES;
            boolean useSubset = false;
            int batchSize = 200;// numSamples/10;
            int iterations = 5;
            int splitTrainNum = (int) (batchSize*.8);
            int seed = 123;
            int listenerFreq = iterations/5;
            DataSet lfwNext;
            SplitTestAndTrain trainTest;
            DataSet trainInput;
            List<INDArray> testInput = new ArrayList<>();
            List<INDArray> testLabels = new ArrayList<>();

            MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .iterations(iterations)
                    .activation("relu")
                    .weightInit(WeightInit.XAVIER)
                    .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .learningRate(0.01)
                    .momentum(0.9)
                    .regularization(true)
                    .updater(Updater.ADAGRAD)
                    .useDropConnect(true)
                    .list(9)
                    .layer(0, new ConvolutionLayer.Builder(4, 4)
                            .name("cnn1")
                            .nIn(nChannels)
                            .stride(1, 1)
                            .nOut(20)
                            .build())
                    .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                            .name("pool1")
                            .build())
                    .layer(2, new ConvolutionLayer.Builder(3, 3)
                            .name("cnn2")
                            .stride(1,1)
                            .nOut(40)
                            .build())
                    .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                            .name("pool2")
                            .build())
                    .layer(4, new ConvolutionLayer.Builder(3, 3)
                            .name("cnn3")
                            .stride(1,1)
                            .nOut(60)
                            .build())
                    .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                            .name("pool3")
                            .build())
                    .layer(6, new ConvolutionLayer.Builder(2, 2)
                            .name("cnn4")
                            .stride(1,1)
                            .nOut(80)
                            .build())
                    .layer(7, new DenseLayer.Builder()
                            .name("ffn1")
                            .nOut(160)
                            .dropOut(0.5)
                            .build())
                    .layer(8, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .nOut(outputNum)
                            .activation("softmax")
                            .build())
                    .backprop(true).pretrain(false);
            new ConvolutionLayerSetup(builder,numRows,numColumns,nChannels);

            network = new MultiLayerNetwork(builder.build());
            network.init();
        }
    }

    @Test
    public void testMLNModelInfo1() throws Exception {
        FlowIterationListener listener = new FlowIterationListener();

        ModelInfo info = listener.buildModelInfo(network);

        for (LayerInfo layerInfo: info.getLayers().values()) {
            log.info("Layer: " + layerInfo);
        }

        // checking total number of layers
        assertEquals(9, info.size());

        // checking, if all named layers exist
        assertNotEquals(null, info.getLayerInfoByName("cnn1"));
        assertNotEquals(null, info.getLayerInfoByName("cnn2"));
        assertNotEquals(null, info.getLayerInfoByName("cnn3"));
        assertNotEquals(null, info.getLayerInfoByName("cnn4"));
        assertNotEquals(null, info.getLayerInfoByName("pool1"));
        assertNotEquals(null, info.getLayerInfoByName("pool2"));
        assertNotEquals(null, info.getLayerInfoByName("pool3"));
        assertNotEquals(null, info.getLayerInfoByName("ffn1"));

        // checking if output layer has no outgoing connections
        assertEquals(0, info.getLayerInfoByCoords(0,9).getConnections().size());

        // check description for cnn
        assertNotEquals(null, info.getLayerInfoByName("cnn1").getDescription().getMainLine());
    }
}
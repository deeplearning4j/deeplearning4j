package org.deeplearning4j.nn.transferlearning;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.FrozenLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Map;

import static org.junit.Assert.*;

/**
 * Created by Alex on 10/07/2017.
 */
public class TestTransferLearningModelSerializer extends BaseDL4JTest {

    @Test
    public void testModelSerializerFrozenLayers() throws Exception {

        FineTuneConfiguration finetune = new FineTuneConfiguration.Builder().updater(new Sgd(0.1)).build();

        int nIn = 6;
        int nOut = 3;

        MultiLayerConfiguration origConf = new NeuralNetConfiguration.Builder().updater(new Sgd(0.1))
                        .activation(Activation.TANH).dropOut(0.5).list()
                        .layer(0, new DenseLayer.Builder().nIn(nIn).nOut(5).build())
                        .layer(1, new DenseLayer.Builder().nIn(5).nOut(4).build())
                        .layer(2, new DenseLayer.Builder().nIn(4).nOut(3).build())
                        .layer(3, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nIn(3)
                                                        .nOut(nOut).build())
                        .build();
        MultiLayerNetwork origModel = new MultiLayerNetwork(origConf);
        origModel.init();

        MultiLayerNetwork withFrozen = new TransferLearning.Builder(origModel).fineTuneConfiguration(finetune)
                        .setFeatureExtractor(1).build();

        assertTrue(withFrozen.getLayer(0) instanceof FrozenLayer);
        assertTrue(withFrozen.getLayer(1) instanceof FrozenLayer);

        assertTrue(withFrozen.getLayerWiseConfigurations().getConf(0)
                        .getLayer() instanceof org.deeplearning4j.nn.conf.layers.misc.FrozenLayer);
        assertTrue(withFrozen.getLayerWiseConfigurations().getConf(1)
                        .getLayer() instanceof org.deeplearning4j.nn.conf.layers.misc.FrozenLayer);

        MultiLayerNetwork restored = TestUtils.testModelSerialization(withFrozen);

        assertTrue(restored.getLayer(0) instanceof FrozenLayer);
        assertTrue(restored.getLayer(1) instanceof FrozenLayer);
        assertFalse(restored.getLayer(2) instanceof FrozenLayer);
        assertFalse(restored.getLayer(3) instanceof FrozenLayer);

        INDArray in = Nd4j.rand(3, nIn);
        INDArray out = withFrozen.output(in);
        INDArray out2 = restored.output(in);

        assertEquals(out, out2);

        //Sanity check on train mode:
        out = withFrozen.output(in, true);
        out2 = restored.output(in, true);
    }


    @Test
    public void testModelSerializerFrozenLayersCompGraph() throws Exception {
        FineTuneConfiguration finetune = new FineTuneConfiguration.Builder().updater(new Sgd(0.1)).build();

        int nIn = 6;
        int nOut = 3;

        ComputationGraphConfiguration origConf = new NeuralNetConfiguration.Builder().activation(Activation.TANH).graphBuilder().addInputs("in")
                        .addLayer("0", new DenseLayer.Builder().nIn(nIn).nOut(5).build(), "in")
                        .addLayer("1", new DenseLayer.Builder().nIn(5).nOut(4).build(), "0")
                        .addLayer("2", new DenseLayer.Builder().nIn(4).nOut(3).build(), "1")
                        .addLayer("3", new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nIn(3)
                                                        .nOut(nOut).build(),
                                        "2")
                        .setOutputs("3").build();
        ComputationGraph origModel = new ComputationGraph(origConf);
        origModel.init();

        ComputationGraph withFrozen = new TransferLearning.GraphBuilder(origModel).fineTuneConfiguration(finetune)
                        .setFeatureExtractor("1").build();

        assertTrue(withFrozen.getLayer(0) instanceof FrozenLayer);
        assertTrue(withFrozen.getLayer(1) instanceof FrozenLayer);

        Map<String, GraphVertex> m = withFrozen.getConfiguration().getVertices();
        Layer l0 = ((LayerVertex) m.get("0")).getLayerConf().getLayer();
        Layer l1 = ((LayerVertex) m.get("1")).getLayerConf().getLayer();
        assertTrue(l0 instanceof org.deeplearning4j.nn.conf.layers.misc.FrozenLayer);
        assertTrue(l1 instanceof org.deeplearning4j.nn.conf.layers.misc.FrozenLayer);

        ComputationGraph restored = TestUtils.testModelSerialization(withFrozen);

        assertTrue(restored.getLayer(0) instanceof FrozenLayer);
        assertTrue(restored.getLayer(1) instanceof FrozenLayer);
        assertFalse(restored.getLayer(2) instanceof FrozenLayer);
        assertFalse(restored.getLayer(3) instanceof FrozenLayer);

        INDArray in = Nd4j.rand(3, nIn);
        INDArray out = withFrozen.outputSingle(in);
        INDArray out2 = restored.outputSingle(in);

        assertEquals(out, out2);

        //Sanity check on train mode:
        out = withFrozen.outputSingle(true, in);
        out2 = restored.outputSingle(true, in);
    }
}

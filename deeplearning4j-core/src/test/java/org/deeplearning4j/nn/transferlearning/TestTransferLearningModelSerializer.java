package org.deeplearning4j.nn.transferlearning;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.layers.FrozenLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Created by Alex on 10/07/2017.
 */
public class TestTransferLearningModelSerializer {

    @Test
    public void testModelSerializerFrozenLayers() throws Exception {

        NeuralNetConfiguration.Builder overallConf = new NeuralNetConfiguration.Builder()
                .learningRate(0.1)
                .updater(Updater.SGD)
                .activation(Activation.TANH);

        FineTuneConfiguration finetune = new FineTuneConfiguration.Builder().learningRate(0.1).build();

        int nIn = 6;
        int nOut = 3;

        MultiLayerNetwork origModel = new MultiLayerNetwork(overallConf.clone().list()
                .layer(0, new DenseLayer.Builder().nIn(6).nOut(5).build())
                .layer(1, new DenseLayer.Builder().nIn(5).nOut(4).build())
                .layer(2, new DenseLayer.Builder().nIn(4).nOut(3).build())
                .layer(3, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                        LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nIn(3).nOut(3)
                        .build())
                .build());
        origModel.init();

        MultiLayerNetwork withFrozen = new TransferLearning.Builder(origModel)
                .fineTuneConfiguration(finetune)
                .setFeatureExtractor(1)
                .build();

        assertTrue(withFrozen.getLayer(0) instanceof FrozenLayer);
        assertTrue(withFrozen.getLayer(1) instanceof FrozenLayer);

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ModelSerializer.writeModel(withFrozen, baos, false);
        baos.close();

        byte[] asBytes = baos.toByteArray();

        ByteArrayInputStream bais = new ByteArrayInputStream(asBytes);
        MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(bais);

        assertTrue(restored.getLayer(0) instanceof FrozenLayer);
        assertTrue(restored.getLayer(1) instanceof FrozenLayer);
        assertFalse(restored.getLayer(0) instanceof FrozenLayer);
        assertFalse(restored.getLayer(0) instanceof FrozenLayer);

    }

}

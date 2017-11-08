package org.deeplearning4j.nn.updater.custom;

import org.deeplearning4j.nn.api.activations.ActivationsFactory;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by Alex on 09/05/2017.
 */
public class TestCustomUpdater {

    private static final ActivationsFactory af = ActivationsFactory.getInstance();

    @Test
    public void testCustomUpdater() {

        //Create a simple custom updater, equivalent to SGD updater

        double lr = 0.03;

        Nd4j.getRandom().setSeed(12345);
        MultiLayerConfiguration conf1 = new NeuralNetConfiguration.Builder().seed(12345)
                        .activation(Activation.TANH).updater(new CustomIUpdater(lr)) //Specify custom IUpdater
                        .list().layer(0, new DenseLayer.Builder().nIn(10).nOut(10).build())
                        .layer(1, new OutputLayer.Builder().nIn(10).nOut(10)
                                        .lossFunction(LossFunctions.LossFunction.MSE).build())
                        .build();

        Nd4j.getRandom().setSeed(12345);
        MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder().seed(12345)
                        .activation(Activation.TANH).updater(new Sgd(lr)).list()
                        .layer(0, new DenseLayer.Builder().nIn(10).nOut(10).build()).layer(1, new OutputLayer.Builder()
                                        .nIn(10).nOut(10).lossFunction(LossFunctions.LossFunction.MSE).build())
                        .build();

        //First: Check updater config
        assertTrue(((BaseLayer) conf1.getConf(0)).getIUpdater() instanceof CustomIUpdater);
        assertTrue(((BaseLayer) conf1.getConf(1)).getIUpdater() instanceof CustomIUpdater);
        assertTrue(((BaseLayer) conf2.getConf(0)).getIUpdater() instanceof Sgd);
        assertTrue(((BaseLayer) conf2.getConf(1)).getIUpdater() instanceof Sgd);

        CustomIUpdater u0_0 = (CustomIUpdater) ((BaseLayer) conf1.getConf(0)).getIUpdater();
        CustomIUpdater u0_1 = (CustomIUpdater) ((BaseLayer) conf1.getConf(1)).getIUpdater();
        assertEquals(lr, u0_0.getLearningRate(), 1e-6);
        assertEquals(lr, u0_1.getLearningRate(), 1e-6);

        Sgd u1_0 = (Sgd) ((BaseLayer) conf2.getConf(0)).getIUpdater();
        Sgd u1_1 = (Sgd) ((BaseLayer) conf2.getConf(1)).getIUpdater();
        assertEquals(lr, u1_0.getLearningRate(), 1e-6);
        assertEquals(lr, u1_1.getLearningRate(), 1e-6);


        //Second: check JSON
        String asJson = conf1.toJson();
        MultiLayerConfiguration fromJson = MultiLayerConfiguration.fromJson(asJson);
        assertEquals(conf1, fromJson);

        Nd4j.getRandom().setSeed(12345);
        MultiLayerNetwork net1 = new MultiLayerNetwork(conf1);
        net1.init();

        Nd4j.getRandom().setSeed(12345);
        MultiLayerNetwork net2 = new MultiLayerNetwork(conf2);
        net2.init();


        //Third: check gradients are equal
        INDArray in = Nd4j.rand(5, 10);
        INDArray labels = Nd4j.rand(5, 10);

        net1.setInput(in);
        net2.setInput(in);

        net1.setLabels(labels);
        net2.setLabels(labels);

        net1.computeGradientAndScore(af.create(in), af.create(labels));
        net2.computeGradientAndScore(af.create(in), af.create(labels));

        assertEquals(net1.getFlattenedGradients(), net2.getFlattenedGradients());
    }

}

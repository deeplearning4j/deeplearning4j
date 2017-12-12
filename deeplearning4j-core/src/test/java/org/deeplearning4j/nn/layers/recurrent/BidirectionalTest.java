package org.deeplearning4j.nn.layers.recurrent;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.MultiLayerUpdater;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.Assert.assertEquals;

public class BidirectionalTest {


    @Test
    public void compareImplementations(){

        //Bidirectional(GravesLSTM) and GravesBidirectionalLSTM should be equivalent, given equivalent params
        //Note that GravesBidirectionalLSTM implements ADD mode only

        MultiLayerConfiguration conf1 = new NeuralNetConfiguration.Builder()
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam())
                .list()
                .layer(new Bidirectional(Bidirectional.Mode.ADD, new GravesLSTM.Builder().nIn(10).nOut(10).build()))
                .layer(new Bidirectional(Bidirectional.Mode.ADD, new GravesLSTM.Builder().nIn(10).nOut(10).build()))
                .layer(new RnnOutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE)
                        .nIn(10).nOut(10).build())
                .build();

        MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam())
                .list()
                .layer(new GravesBidirectionalLSTM.Builder().nIn(10).nOut(10).build())
                .layer(new GravesBidirectionalLSTM.Builder().nIn(10).nOut(10).build())
                .layer(new RnnOutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE)
                        .nIn(10).nOut(10).build())
                .build();

        MultiLayerNetwork net1 = new MultiLayerNetwork(conf1);
        net1.init();

        MultiLayerNetwork net2 = new MultiLayerNetwork(conf2);
        net2.init();

        assertEquals(net1.numParams(), net2.numParams());
        for(int i=0; i<3; i++ ){
            int n1 = net1.getLayer(i).numParams();
            int n2 = net2.getLayer(i).numParams();
            assertEquals(n1, n2);
        }

        net2.setParams(net1.params());  //Assuming exact same layout here...

        INDArray in = Nd4j.rand(new int[]{3, 10, 5});

        INDArray out1 = net1.output(in);
        INDArray out2 = net2.output(in);

        assertEquals(out1, out2);

        INDArray labels = Nd4j.rand(new int[]{3, 10, 5});

        net1.setInput(in);
        net1.setLabels(labels);

        net2.setInput(in);
        net2.setLabels(labels);

        net1.computeGradientAndScore();
        net2.computeGradientAndScore();

        //Ensure scores are equal:
        assertEquals(net1.score(), net2.score(), 1e-6);

        //Ensure gradients are equal:
        Gradient g1 = net1.gradient();
        Gradient g2 = net2.gradient();
        assertEquals(g1.gradient(), g2.gradient());

        //Ensure updates are equal:
        MultiLayerUpdater u1 = (MultiLayerUpdater) net1.getUpdater();
        MultiLayerUpdater u2 = (MultiLayerUpdater) net2.getUpdater();
        assertEquals(u1.getUpdaterStateViewArray(), u2.getUpdaterStateViewArray());
        u1.update(net1, g1, 0, 0, 3);
        u2.update(net2, g2, 0, 0, 3);
        assertEquals(g1.gradient(), g2.gradient());
        assertEquals(u1.getUpdaterStateViewArray(), u2.getUpdaterStateViewArray());

        //Ensure params are equal, after fitting
        net1.fit(in, labels);
        net2.fit(in, labels);

        INDArray p1 = net1.params();
        INDArray p2 = net2.params();
        assertEquals(p1, p2);
    }

}

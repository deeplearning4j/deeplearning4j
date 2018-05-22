package org.deeplearning4j.nn.rl;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

/**
 * Testing calculating a Gradient object in one model, and updating/applying it on another.
 * This is used for example in RL4J
 *
 * @author Alex Black
 */
public class TestMultiModelGradientApplication extends BaseDL4JTest {

    @Test
    public void testGradientApplyMultiLayerNetwork() {
        int minibatch = 7;
        int nIn = 10;
        int nOut = 10;

        for (boolean regularization : new boolean[] {false, true}) {
            for (IUpdater u : new IUpdater[] {new Sgd(0.1), new Nesterovs(0.1), new Adam(0.1)}) {

                MultiLayerConfiguration conf =
                                new NeuralNetConfiguration.Builder().seed(12345).activation(Activation.TANH)
                                                .weightInit(WeightInit.XAVIER).updater(u)
                                                .l1(regularization ? 0.2 : 0.0)
                                                .l2(regularization ? 0.3 : 0.0).list()
                                                .layer(0, new DenseLayer.Builder().nIn(nIn).nOut(10).build())
                                                .layer(1, new DenseLayer.Builder().nIn(10).nOut(10).build()).layer(2,
                                                                new OutputLayer.Builder(
                                                                                LossFunctions.LossFunction.MCXENT)
                                                                                                .activation(Activation.SOFTMAX)
                                                                                                .nIn(10).nOut(nOut)
                                                                                                .build())
                                                .build();


                Nd4j.getRandom().setSeed(12345);
                MultiLayerNetwork net1GradCalc = new MultiLayerNetwork(conf);
                net1GradCalc.init();

                Nd4j.getRandom().setSeed(12345);
                MultiLayerNetwork net2GradUpd = new MultiLayerNetwork(conf.clone());
                net2GradUpd.init();

                assertEquals(net1GradCalc.params(), net2GradUpd.params());

                INDArray f = Nd4j.rand(minibatch, nIn);
                INDArray l = Nd4j.create(minibatch, nOut);
                for (int i = 0; i < minibatch; i++) {
                    l.putScalar(i, i % nOut, 1.0);
                }
                net1GradCalc.setInput(f);
                net1GradCalc.setLabels(l);

                net2GradUpd.setInput(f);
                net2GradUpd.setLabels(l);

                //Calculate gradient in first net, update and apply it in the second
                //Also: calculate gradient in the second net, just to be sure it isn't modified while doing updating on
                // the other net's gradient
                net1GradCalc.computeGradientAndScore();
                net2GradUpd.computeGradientAndScore();

                Gradient g = net1GradCalc.gradient();
                INDArray gBefore = g.gradient().dup(); //Net 1 gradient should be modified
                INDArray net2GradBefore = net2GradUpd.gradient().gradient().dup(); //But net 2 gradient should not be
                net2GradUpd.getUpdater().update(net2GradUpd, g, 0, 0, minibatch, LayerWorkspaceMgr.noWorkspaces());
                INDArray gAfter = g.gradient().dup();
                INDArray net2GradAfter = net2GradUpd.gradient().gradient().dup();

                assertNotEquals(gBefore, gAfter); //Net 1 gradient should be modified
                assertEquals(net2GradBefore, net2GradAfter); //But net 2 gradient should not be


                //Also: if we apply the gradient using a subi op, we should get the same final params as if we did a fit op
                // on the original network
                net2GradUpd.params().subi(g.gradient());

                net1GradCalc.fit(f, l);
                assertEquals(net1GradCalc.params(), net2GradUpd.params());


                //=============================
                if (!(u instanceof Sgd)) {
                    net2GradUpd.getUpdater().getStateViewArray().assign(net1GradCalc.getUpdater().getStateViewArray());
                }
                assertEquals(net1GradCalc.params(), net2GradUpd.params());
                assertEquals(net1GradCalc.getUpdater().getStateViewArray(),
                                net2GradUpd.getUpdater().getStateViewArray());

                //Remove the next 2 lines: fails - as net 1 is 1 iteration ahead
                net1GradCalc.getLayerWiseConfigurations().setIterationCount(0);
                net2GradUpd.getLayerWiseConfigurations().setIterationCount(0);

                for (int i = 0; i < 100; i++) {
                    net1GradCalc.fit(f, l);
                    net2GradUpd.fit(f, l);
                    assertEquals(net1GradCalc.params(), net2GradUpd.params());
                }
            }
        }
    }


    @Test
    public void testGradientApplyComputationGraph() {
        int minibatch = 7;
        int nIn = 10;
        int nOut = 10;

        for (boolean regularization : new boolean[] {false, true}) {
            for (IUpdater u : new IUpdater[] {new Sgd(0.1), new Adam(0.1)}) {

                ComputationGraphConfiguration conf =
                                new NeuralNetConfiguration.Builder().seed(12345).activation(Activation.TANH)
                                                .weightInit(WeightInit.XAVIER).updater(u)
                                                .l1(regularization ? 0.2 : 0.0)
                                                .l2(regularization ? 0.3 : 0.0).graphBuilder().addInputs("in")
                                                .addLayer("0", new DenseLayer.Builder().nIn(nIn).nOut(10).build(), "in")
                                                .addLayer("1", new DenseLayer.Builder().nIn(10).nOut(10).build(), "0")
                                                .addLayer("2", new OutputLayer.Builder(
                                                                LossFunctions.LossFunction.MCXENT)
                                                                                .activation(Activation.SOFTMAX).nIn(10)
                                                                                .nOut(nOut).build(),
                                                                "1")
                                                .setOutputs("2").build();


                Nd4j.getRandom().setSeed(12345);
                ComputationGraph net1GradCalc = new ComputationGraph(conf);
                net1GradCalc.init();

                Nd4j.getRandom().setSeed(12345);
                ComputationGraph net2GradUpd = new ComputationGraph(conf.clone());
                net2GradUpd.init();

                assertEquals(net1GradCalc.params(), net2GradUpd.params());

                INDArray f = Nd4j.rand(minibatch, nIn);
                INDArray l = Nd4j.create(minibatch, nOut);
                for (int i = 0; i < minibatch; i++) {
                    l.putScalar(i, i % nOut, 1.0);
                }
                net1GradCalc.setInputs(f);
                net1GradCalc.setLabels(l);

                net2GradUpd.setInputs(f);
                net2GradUpd.setLabels(l);

                //Calculate gradient in first net, update and apply it in the second
                //Also: calculate gradient in the second net, just to be sure it isn't modified while doing updating on
                // the other net's gradient
                net1GradCalc.computeGradientAndScore();
                net2GradUpd.computeGradientAndScore();

                Gradient g = net1GradCalc.gradient();
                INDArray gBefore = g.gradient().dup(); //Net 1 gradient should be modified
                INDArray net2GradBefore = net2GradUpd.gradient().gradient().dup(); //But net 2 gradient should not be
                net2GradUpd.getUpdater().update(g, 0, 0, minibatch, LayerWorkspaceMgr.noWorkspaces());
                INDArray gAfter = g.gradient().dup();
                INDArray net2GradAfter = net2GradUpd.gradient().gradient().dup();

                assertNotEquals(gBefore, gAfter); //Net 1 gradient should be modified
                assertEquals(net2GradBefore, net2GradAfter); //But net 2 gradient should not be


                //Also: if we apply the gradient using a subi op, we should get the same final params as if we did a fit op
                // on the original network
                net2GradUpd.params().subi(g.gradient());

                net1GradCalc.fit(new INDArray[] {f}, new INDArray[] {l});
                assertEquals(net1GradCalc.params(), net2GradUpd.params());

                //=============================
                if (!(u instanceof Sgd)) {
                    net2GradUpd.getUpdater().getStateViewArray().assign(net1GradCalc.getUpdater().getStateViewArray());
                }
                assertEquals(net1GradCalc.params(), net2GradUpd.params());
                assertEquals(net1GradCalc.getUpdater().getStateViewArray(),
                                net2GradUpd.getUpdater().getStateViewArray());

                //Remove the next 2 lines: fails - as net 1 is 1 iteration ahead
                net1GradCalc.getConfiguration().setIterationCount(0);
                net2GradUpd.getConfiguration().setIterationCount(0);


                for (int i = 0; i < 100; i++) {
                    net1GradCalc.fit(new INDArray[] {f}, new INDArray[] {l});
                    net2GradUpd.fit(new INDArray[] {f}, new INDArray[] {l});
                    assertEquals(net1GradCalc.params(), net2GradUpd.params());
                }
            }
        }
    }

}

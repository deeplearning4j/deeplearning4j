package org.deeplearning4j.nn.layers.recurrent;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.dropout.TestDropout;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

public class TestRnnLayers extends BaseDL4JTest {

    @Test
    public void testTimeStepIs3Dimensional() {

        int nIn = 12;
        int nOut = 3;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new NoOp())
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new SimpleRnn.Builder().nIn(nIn).nOut(3).build())
                .layer(new LSTM.Builder().nIn(3).nOut(5).build())
                .layer(new RnnOutputLayer.Builder().nOut(nOut).build())
                .build();


        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        org.deeplearning4j.nn.layers.recurrent.SimpleRnn simpleRnn =
                (org.deeplearning4j.nn.layers.recurrent.SimpleRnn) net.getLayer(0);

        INDArray rnnInput3d = Nd4j.create(10, 12, 1);
        INDArray simpleOut = simpleRnn.rnnTimeStep(rnnInput3d, LayerWorkspaceMgr.noWorkspaces());
        assertTrue(Arrays.equals(simpleOut.shape(), new long[] {10, 3, 1}));

        INDArray rnnInput2d = Nd4j.create(10, 12);
        try {
            simpleRnn.rnnTimeStep(rnnInput2d, LayerWorkspaceMgr.noWorkspaces());
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().equals("3D input expected to RNN layer expected, got 2"));
        }

        org.deeplearning4j.nn.layers.recurrent.LSTM lstm =
                (org.deeplearning4j.nn.layers.recurrent.LSTM) net.getLayer(1);

        INDArray lstmInput3d = Nd4j.create(10, 3, 1);
        INDArray lstmOut = lstm.rnnTimeStep(lstmInput3d, LayerWorkspaceMgr.noWorkspaces());
        assertTrue(Arrays.equals(lstmOut.shape(), new long[] {10, 5, 1}));

        INDArray lstmInput2d = Nd4j.create(10, 3);
        try {
            lstm.rnnTimeStep(lstmInput2d, LayerWorkspaceMgr.noWorkspaces());
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().equals("3D input expected to RNN layer expected, got 2"));
        }


    }

    @Test
    public void testDropoutRecurrentLayers(){
        Nd4j.getRandom().setSeed(12345);

        String[] layerTypes = new String[]{"graves", "lstm", "simple"};

        for(String s : layerTypes){

            Layer layer;
            Layer layerD;
            Layer layerD2;
            TestDropout.CustomDropout cd = new TestDropout.CustomDropout();
            switch (s){
                case "graves":
                    layer = new GravesLSTM.Builder().activation(Activation.TANH).nIn(10).nOut(10).build();
                    layerD = new GravesLSTM.Builder().dropOut(0.5).activation(Activation.TANH).nIn(10).nOut(10).build();
                    layerD2 = new GravesLSTM.Builder().dropOut(cd).activation(Activation.TANH).nIn(10).nOut(10).build();
                    break;
                case "lstm":
                    layer = new org.deeplearning4j.nn.conf.layers.LSTM.Builder().activation(Activation.TANH).nIn(10).nOut(10).build();
                    layerD = new org.deeplearning4j.nn.conf.layers.LSTM.Builder().dropOut(0.5).activation(Activation.TANH).nIn(10).nOut(10).build();
                    layerD2 = new org.deeplearning4j.nn.conf.layers.LSTM.Builder().dropOut(cd).activation(Activation.TANH).nIn(10).nOut(10).build();
                    break;
                case "simple":
                    layer = new SimpleRnn.Builder().activation(Activation.TANH).nIn(10).nOut(10).build();
                    layerD = new SimpleRnn.Builder().dropOut(0.5).activation(Activation.TANH).nIn(10).nOut(10).build();
                    layerD2 = new SimpleRnn.Builder().dropOut(cd).activation(Activation.TANH).nIn(10).nOut(10).build();
                    break;
                default:
                    throw new RuntimeException(s);
            }

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(12345)
                    .list()
                    .layer(layer)
                    .layer(new RnnOutputLayer.Builder().activation(Activation.TANH).nIn(10).nOut(10).build())
                    .build();

            MultiLayerConfiguration confD = new NeuralNetConfiguration.Builder()
                    .seed(12345)
                    .list()
                    .layer(layerD)
                    .layer(new RnnOutputLayer.Builder().activation(Activation.TANH).nIn(10).nOut(10).build())
                    .build();

            MultiLayerConfiguration confD2 = new NeuralNetConfiguration.Builder()
                    .seed(12345)
                    .list()
                    .layer(layerD2)
                    .layer(new RnnOutputLayer.Builder().activation(Activation.TANH).nIn(10).nOut(10).build())
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            MultiLayerNetwork netD = new MultiLayerNetwork(confD);
            netD.init();

            MultiLayerNetwork netD2 = new MultiLayerNetwork(confD2);
            netD2.init();

            assertEquals(s, net.params(), netD.params());
            assertEquals(s, net.params(), netD2.params());

            INDArray f = Nd4j.rand(new int[]{3, 10, 10});

            //Output: test mode -> no dropout
            INDArray out1 = net.output(f);
            INDArray out1D = netD.output(f);
            INDArray out1D2 = netD2.output(f);
            assertEquals(s, out1, out1D);
            assertEquals(s, out1, out1D2);


            INDArray out2 = net.output(f, true);
            INDArray out2D = netD.output(f, true);
            assertNotEquals(s, out2, out2D);

            INDArray l = TestUtils.randomOneHotTimeSeries(3, 10, 10, 12345);

            net.fit(f.dup(), l);
            netD.fit(f.dup(), l);
            assertNotEquals(s, net.params(), netD.params());

            netD2.fit(f.dup(), l);
            netD2.fit(f.dup(), l);
            netD2.fit(f.dup(), l);


            List<Pair<Integer,Integer>> expected = Arrays.asList(
                    new Pair<>(0, 0),
                    new Pair<>(1, 0),
                    new Pair<>(2, 0));

            assertEquals(s, expected, cd.getAllCalls());
        }
    }

}

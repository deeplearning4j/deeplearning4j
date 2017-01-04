package org.deeplearning4j.optimizer.listener;

import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.api.storage.listener.RoutingIterationListener;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.AutoEncoder;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by Alex on 01/01/2017.
 */
public class TestListenerSetting {

    @Test
    public void testSettingListenersUnsupervised(){
        //Pretrain layers should get copies of the listeners, in addition to the

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(0, new RBM.Builder().nIn(10).nOut(10).build())
                .layer(1, new AutoEncoder.Builder().nIn(10).nOut(10).build())
                .layer(2, new VariationalAutoencoder.Builder().nIn(10).nOut(10).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.setListeners(new ScoreIterationListener(), new TestRoutingListener());

        for(Layer l : net.getLayers()){
            Collection<IterationListener> layerListeners = l.getListeners();
            assertEquals(l.getClass().toString(),2, layerListeners.size());
            IterationListener[] lArr = layerListeners.toArray(new IterationListener[2]);
            assertTrue(lArr[0] instanceof ScoreIterationListener);
            assertTrue(lArr[1] instanceof TestRoutingListener);
        }

        Collection<IterationListener> netListeners = net.getListeners();
        assertEquals(2, netListeners.size());
        IterationListener[] lArr = netListeners.toArray(new IterationListener[2]);
        assertTrue(lArr[0] instanceof ScoreIterationListener);
        assertTrue(lArr[1] instanceof TestRoutingListener);


        ComputationGraphConfiguration gConf = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("in")
                .addLayer("0", new RBM.Builder().nIn(10).nOut(10).build(), "in")
                .addLayer("1", new AutoEncoder.Builder().nIn(10).nOut(10).build(), "0")
                .addLayer("2", new VariationalAutoencoder.Builder().nIn(10).nOut(10).build(), "1")
                .setOutputs("2")
                .build();
        ComputationGraph cg = new ComputationGraph(gConf);
        cg.init();

        cg.setListeners(new ScoreIterationListener(), new TestRoutingListener());

        for(Layer l : cg.getLayers()){
            Collection<IterationListener> layerListeners = l.getListeners();
            assertEquals(2, layerListeners.size());
            lArr = layerListeners.toArray(new IterationListener[2]);
            assertTrue(lArr[0] instanceof ScoreIterationListener);
            assertTrue(lArr[1] instanceof TestRoutingListener);
        }

        netListeners = cg.getListeners();
        assertEquals(2, netListeners.size());
        lArr = netListeners.toArray(new IterationListener[2]);
        assertTrue(lArr[0] instanceof ScoreIterationListener);
        assertTrue(lArr[1] instanceof TestRoutingListener);
    }

    private static class TestRoutingListener implements RoutingIterationListener {

        @Override
        public void onEpochStart(Model model) { }

        @Override
        public void onEpochEnd(Model model) { }

        @Override
        public void onForwardPass(Model model, List<INDArray> activations) { }

        @Override
        public void onForwardPass(Model model, Map<String, INDArray> activations) { }

        @Override
        public void onGradientCalculation(Model model) { }

        @Override
        public void onBackwardPass(Model model) { }

        @Override
        public void setStorageRouter(StatsStorageRouter router) { }

        @Override
        public StatsStorageRouter getStorageRouter() { return null; }

        @Override
        public void setWorkerID(String workerID) { }

        @Override
        public String getWorkerID() { return null; }

        @Override
        public void setSessionID(String sessionID) { }

        @Override
        public String getSessionID() { return null; }

        @Override
        public RoutingIterationListener clone() { return null; }

        @Override
        public boolean invoked() { return false; }

        @Override
        public void invoke() { }

        @Override
        public void iterationDone(Model model, int iteration) { }
    }

}

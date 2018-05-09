package org.deeplearning4j.optimizer.listener;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.api.storage.listener.RoutingIterationListener;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.AutoEncoder;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.ComposableIterationListener;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.optimize.listeners.TimeIterationListener;
import org.deeplearning4j.optimize.listeners.checkpoint.CheckpointListener;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by Alex on 01/01/2017.
 */
public class TestListeners extends BaseDL4JTest {

    @Rule
    public TemporaryFolder tempDir = new TemporaryFolder();

    @Test
    public void testSettingListenersUnsupervised() {
        //Pretrain layers should get copies of the listeners, in addition to the

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list()
                        .layer(0, new AutoEncoder.Builder().nIn(10).nOut(10).build())
                        .layer(1, new VariationalAutoencoder.Builder().nIn(10).nOut(10).build()).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.setListeners(new ScoreIterationListener(), new TestRoutingListener());

        for (Layer l : net.getLayers()) {
            Collection<TrainingListener> layerListeners = l.getListeners();
            assertEquals(l.getClass().toString(), 2, layerListeners.size());
            TrainingListener[] lArr = layerListeners.toArray(new TrainingListener[2]);
            assertTrue(lArr[0] instanceof ScoreIterationListener);
            assertTrue(lArr[1] instanceof TestRoutingListener);
        }

        Collection<TrainingListener> netListeners = net.getListeners();
        assertEquals(2, netListeners.size());
        TrainingListener[] lArr = netListeners.toArray(new TrainingListener[2]);
        assertTrue(lArr[0] instanceof ScoreIterationListener);
        assertTrue(lArr[1] instanceof TestRoutingListener);


        ComputationGraphConfiguration gConf = new NeuralNetConfiguration.Builder().graphBuilder().addInputs("in")
                        .addLayer("0", new AutoEncoder.Builder().nIn(10).nOut(10).build(), "in")
                        .addLayer("1", new VariationalAutoencoder.Builder().nIn(10).nOut(10).build(), "0")
                        .setOutputs("1").build();
        ComputationGraph cg = new ComputationGraph(gConf);
        cg.init();

        cg.setListeners(new ScoreIterationListener(), new TestRoutingListener());

        for (Layer l : cg.getLayers()) {
            Collection<TrainingListener> layerListeners = l.getListeners();
            assertEquals(2, layerListeners.size());
            lArr = layerListeners.toArray(new TrainingListener[2]);
            assertTrue(lArr[0] instanceof ScoreIterationListener);
            assertTrue(lArr[1] instanceof TestRoutingListener);
        }

        netListeners = cg.getListeners();
        assertEquals(2, netListeners.size());
        lArr = netListeners.toArray(new TrainingListener[2]);
        assertTrue(lArr[0] instanceof ScoreIterationListener);
        assertTrue(lArr[1] instanceof TestRoutingListener);
    }

    private static class TestRoutingListener extends BaseTrainingListener implements RoutingIterationListener {

        @Override
        public void setStorageRouter(StatsStorageRouter router) {}

        @Override
        public StatsStorageRouter getStorageRouter() {
            return null;
        }

        @Override
        public void setWorkerID(String workerID) {}

        @Override
        public String getWorkerID() {
            return null;
        }

        @Override
        public void setSessionID(String sessionID) {}

        @Override
        public String getSessionID() {
            return null;
        }

        @Override
        public RoutingIterationListener clone() {
            return null;
        }

        @Override
        public void iterationDone(Model model, int iteration, int epoch) {}
    }





    @Test
    public void testListenerSerialization() throws Exception {
        //Note: not all listeners are (or should be) serializable. But some should be - for Spark etc

        List<TrainingListener> listeners = new ArrayList<>();
        listeners.add(new ScoreIterationListener());
        listeners.add(new PerformanceListener(1));
        listeners.add(new TimeIterationListener(10000));
        listeners.add(new ComposableIterationListener(new ScoreIterationListener(), new PerformanceListener(1)));
        listeners.add(new CheckpointListener.Builder(tempDir.newFolder()).keepAll().saveEveryNIterations(3).build());   //Doesn't usually need to be serialized, but no reason it can't be...


        DataSetIterator iter = new IrisDataSetIterator(10, 150);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(new OutputLayer.Builder().nIn(4).nOut(3)
                        .activation(Activation.TANH)
                        .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(listeners);

        net.fit(iter);

        List<TrainingListener> listeners2 = new ArrayList<>();
        for(TrainingListener il : listeners){
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(baos);
            oos.writeObject(il);
            byte[] bytes = baos.toByteArray();

            ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(bytes));
            TrainingListener il2 = (TrainingListener) ois.readObject();

            listeners2.add(il2);
        }

        net.setListeners(listeners2);
        net.fit(iter);
    }

}

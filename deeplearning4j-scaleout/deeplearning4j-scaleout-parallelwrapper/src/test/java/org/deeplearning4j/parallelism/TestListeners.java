package org.deeplearning4j.parallelism;

import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.api.storage.listener.RoutingIterationListener;
import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
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
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 23/03/2017.
 */
public class TestListeners {

    @Test
    public void testListeners(){
        TestListener.clearCounts();

        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .list()
                .layer(0, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(10).nOut(10).activation(Activation.TANH).build());

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        testListenersForModel(model);
    }

    @Test
    public void testListenersGraph() {
        TestListener.clearCounts();

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("in")
                .addLayer("0", new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(10).nOut(10).activation(Activation.TANH).build(), "in")
                .setOutputs("0")
                .build();

        ComputationGraph model = new ComputationGraph(conf);
        model.init();

        testListenersForModel(model);
    }

    private static void testListenersForModel(Model model){

        int nWorkers = 2;
        ParallelWrapper wrapper = new ParallelWrapper.Builder(model)
                .workers(nWorkers)
                .averagingFrequency(1)
                .reportScoreAfterAveraging(true)
                .useLegacyAveraging(true)
                .build();

        wrapper.setListeners(new TestListener());

        List<DataSet> data = new ArrayList<>();
        for( int i=0; i<nWorkers; i++ ){
            data.add( new DataSet(Nd4j.rand(1, 10), Nd4j.rand(1,10)));
        }

        DataSetIterator iter = new ExistingDataSetIterator(data);

        wrapper.fit(iter);

        assertEquals(nWorkers+1, TestListener.instanceCount.get()); //Original instance + 2 clones
        assertEquals(2, TestListener.workerIDs.size());
        assertEquals(1, TestListener.sessionIDs.size());
        assertEquals(2, TestListener.forwardPassCount.get());
    }


    private static class TestListener implements RoutingIterationListener {

        private static final AtomicInteger forwardPassCount = new AtomicInteger();
        private static final AtomicInteger instanceCount = new AtomicInteger();
        private static final Set<String> workerIDs = new ConcurrentHashSet<>();
        private static final Set<String> sessionIDs = new ConcurrentHashSet<>();

        public static void clearCounts(){
            forwardPassCount.set(0);
            instanceCount.set(0);
            workerIDs.clear();
            sessionIDs.clear();
        }

        public TestListener(){
            instanceCount.incrementAndGet();
        }

        @Override
        public void onEpochStart(Model model) { }

        @Override
        public void onEpochEnd(Model model) { }

        @Override
        public void onForwardPass(Model model, List<INDArray> activations) {
            forwardPassCount.incrementAndGet();
        }

        @Override
        public void onForwardPass(Model model, Map<String, INDArray> activations) {
            forwardPassCount.incrementAndGet();
        }

        @Override
        public void onGradientCalculation(Model model) { }

        @Override
        public void onBackwardPass(Model model) { }

        @Override
        public void setStorageRouter(StatsStorageRouter router) { }

        @Override
        public StatsStorageRouter getStorageRouter() { return null; }

        @Override
        public void setWorkerID(String workerID) {
            workerIDs.add(workerID);
        }

        @Override
        public String getWorkerID() {
            return null;
        }

        @Override
        public void setSessionID(String sessionID) {
            sessionIDs.add(sessionID);
        }

        @Override
        public String getSessionID() {
            return "session_id";
        }

        @Override
        public RoutingIterationListener clone() {
            return new TestListener();
        }

        @Override
        public boolean invoked() { return false; }

        @Override
        public void invoke() { }

        @Override
        public void iterationDone(Model model, int iteration) { }
    }

}

package org.deeplearning4j.nn.conf.weightnoise;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.MatchCondition;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.SigmoidSchedule;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

public class TestWeightNoise extends BaseDL4JTest {

    @Test
    public void testWeightNoiseConfigJson() {
        IWeightNoise[] weightNoises = new IWeightNoise[]{
                new DropConnect(0.5),
                new DropConnect(new SigmoidSchedule(ScheduleType.ITERATION, 0.5, 0.5, 100)),
                new WeightNoise(new NormalDistribution(0, 0.1))
        };

        for (IWeightNoise wn : weightNoises) {
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .weightNoise(wn)
                    .list()
                    .layer(new DenseLayer.Builder().nIn(10).nOut(10).build())
                    .layer(new DenseLayer.Builder().nIn(10).nOut(10).weightNoise(new DropConnect(0.25)).build())
                    .layer(new OutputLayer.Builder().nIn(10).nOut(10).build())
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            assertEquals(wn, ((BaseLayer) net.getLayer(0).conf().getLayer()).getWeightNoise());
            assertEquals(new DropConnect(0.25), ((BaseLayer) net.getLayer(1).conf().getLayer()).getWeightNoise());
            assertEquals(wn, ((BaseLayer) net.getLayer(2).conf().getLayer()).getWeightNoise());

            TestUtils.testModelSerialization(net);


            ComputationGraphConfiguration conf2 = new NeuralNetConfiguration.Builder()
                    .weightNoise(wn)
                    .graphBuilder()
                    .addInputs("in")
                    .layer("0", new DenseLayer.Builder().nIn(10).nOut(10).build(), "in")
                    .layer("1", new DenseLayer.Builder().nIn(10).nOut(10).weightNoise(new DropConnect(0.25)).build(), "0")
                    .layer("2", new OutputLayer.Builder().nIn(10).nOut(10).build(), "1")
                    .setOutputs("2")
                    .build();

            ComputationGraph graph = new ComputationGraph(conf2);
            graph.init();

            assertEquals(wn, ((BaseLayer) graph.getLayer(0).conf().getLayer()).getWeightNoise());
            assertEquals(new DropConnect(0.25), ((BaseLayer) graph.getLayer(1).conf().getLayer()).getWeightNoise());
            assertEquals(wn, ((BaseLayer) graph.getLayer(2).conf().getLayer()).getWeightNoise());

            TestUtils.testModelSerialization(graph);
        }
    }


    @Test
    public void testCalls() {

        List<DataSet> trainData = new ArrayList<>();
        trainData.add(new DataSet(Nd4j.rand(5, 10), Nd4j.rand(5, 10)));
        trainData.add(new DataSet(Nd4j.rand(5, 10), Nd4j.rand(5, 10)));
        trainData.add(new DataSet(Nd4j.rand(5, 10), Nd4j.rand(5, 10)));

        List<List<WeightNoiseCall>> expCalls = new ArrayList<>();
        for (int i = 0; i < 3; i++) {
            List<WeightNoiseCall> expCallsForLayer = new ArrayList<>();
            expCallsForLayer.add(new WeightNoiseCall(i, "W", 0, 0, true));
            expCallsForLayer.add(new WeightNoiseCall(i, "b", 0, 0, true));
            expCallsForLayer.add(new WeightNoiseCall(i, "W", 1, 0, true));
            expCallsForLayer.add(new WeightNoiseCall(i, "b", 1, 0, true));
            expCallsForLayer.add(new WeightNoiseCall(i, "W", 2, 0, true));
            expCallsForLayer.add(new WeightNoiseCall(i, "b", 2, 0, true));
            expCallsForLayer.add(new WeightNoiseCall(i, "W", 3, 1, true));
            expCallsForLayer.add(new WeightNoiseCall(i, "b", 3, 1, true));
            expCallsForLayer.add(new WeightNoiseCall(i, "W", 4, 1, true));
            expCallsForLayer.add(new WeightNoiseCall(i, "b", 4, 1, true));
            expCallsForLayer.add(new WeightNoiseCall(i, "W", 5, 1, true));
            expCallsForLayer.add(new WeightNoiseCall(i, "b", 5, 1, true));

            //2 test calls
            expCallsForLayer.add(new WeightNoiseCall(i, "W", 6, 2, false));
            expCallsForLayer.add(new WeightNoiseCall(i, "b", 6, 2, false));

            expCalls.add(expCallsForLayer);
        }


        CustomWeightNoise wn1 = new CustomWeightNoise();
        CustomWeightNoise wn2 = new CustomWeightNoise();
        CustomWeightNoise wn3 = new CustomWeightNoise();

        List<CustomWeightNoise> list = Arrays.asList(wn1, wn2, wn3);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(new DenseLayer.Builder().nIn(10).nOut(10).weightNoise(wn1).build())
                .layer(new DenseLayer.Builder().nIn(10).nOut(10).weightNoise(wn2).build())
                .layer(new OutputLayer.Builder().nIn(10).nOut(10).weightNoise(wn3).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.fit(new ExistingDataSetIterator(trainData.iterator()));
        net.fit(new ExistingDataSetIterator(trainData.iterator()));
        net.output(trainData.get(0).getFeatures());

        for (int i = 0; i < 3; i++) {
            assertEquals(expCalls.get(i), list.get(i).getAllCalls());
        }


        wn1 = new CustomWeightNoise();
        wn2 = new CustomWeightNoise();
        wn3 = new CustomWeightNoise();
        list = Arrays.asList(wn1, wn2, wn3);

        ComputationGraphConfiguration conf2 = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("in")
                .layer("0", new DenseLayer.Builder().nIn(10).nOut(10).weightNoise(wn1).build(), "in")
                .layer("1", new DenseLayer.Builder().nIn(10).nOut(10).weightNoise(wn2).build(), "0")
                .layer("2", new OutputLayer.Builder().nIn(10).nOut(10).weightNoise(wn3).build(), "1")
                .setOutputs("2")
                .build();

        ComputationGraph graph = new ComputationGraph(conf2);
        graph.init();

        int[] layerIdxs = new int[]{graph.getLayer(0).getIndex(), graph.getLayer(1).getIndex(), graph.getLayer(2).getIndex()};

        expCalls.clear();
        for (int i = 0; i < 3; i++) {
            List<WeightNoiseCall> expCallsForLayer = new ArrayList<>();
            expCallsForLayer.add(new WeightNoiseCall(layerIdxs[i], "W", 0, 0, true));
            expCallsForLayer.add(new WeightNoiseCall(layerIdxs[i], "b", 0, 0, true));
            expCallsForLayer.add(new WeightNoiseCall(layerIdxs[i], "W", 1, 0, true));
            expCallsForLayer.add(new WeightNoiseCall(layerIdxs[i], "b", 1, 0, true));
            expCallsForLayer.add(new WeightNoiseCall(layerIdxs[i], "W", 2, 0, true));
            expCallsForLayer.add(new WeightNoiseCall(layerIdxs[i], "b", 2, 0, true));
            expCallsForLayer.add(new WeightNoiseCall(layerIdxs[i], "W", 3, 1, true));
            expCallsForLayer.add(new WeightNoiseCall(layerIdxs[i], "b", 3, 1, true));
            expCallsForLayer.add(new WeightNoiseCall(layerIdxs[i], "W", 4, 1, true));
            expCallsForLayer.add(new WeightNoiseCall(layerIdxs[i], "b", 4, 1, true));
            expCallsForLayer.add(new WeightNoiseCall(layerIdxs[i], "W", 5, 1, true));
            expCallsForLayer.add(new WeightNoiseCall(layerIdxs[i], "b", 5, 1, true));

            //2 test calls
            expCallsForLayer.add(new WeightNoiseCall(layerIdxs[i], "W", 6, 2, false));
            expCallsForLayer.add(new WeightNoiseCall(layerIdxs[i], "b", 6, 2, false));

            expCalls.add(expCallsForLayer);
        }

        graph.fit(new ExistingDataSetIterator(trainData.iterator()));
        graph.fit(new ExistingDataSetIterator(trainData.iterator()));
        graph.output(trainData.get(0).getFeatures());

        for (int i = 0; i < 3; i++) {
            assertEquals(String.valueOf(i), expCalls.get(i), list.get(i).getAllCalls());
        }

    }

    @Data
    private static class CustomWeightNoise implements IWeightNoise {

        private List<WeightNoiseCall> allCalls = new ArrayList<>();

        @Override
        public INDArray getParameter(Layer layer, String paramKey, int iteration, int epoch, boolean train, LayerWorkspaceMgr workspaceMgr) {
            allCalls.add(new WeightNoiseCall(layer.getIndex(), paramKey, iteration, epoch, train));
            return layer.getParam(paramKey);
        }

        @Override
        public IWeightNoise clone() {
            return new CustomWeightNoise();
        }
    }

    @AllArgsConstructor
    @Data
    private static class WeightNoiseCall {
        private int layerIdx;
        private String paramKey;
        private int iter;
        private int epoch;
        private boolean train;
    }


    @Test
    public void testDropConnectValues() {
        Nd4j.getRandom().setSeed(12345);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.ONES)
                .list()
                .layer(new OutputLayer.Builder().nIn(10).nOut(10).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        Layer l = net.getLayer(0);
        DropConnect d = new DropConnect(0.5);

        INDArray outTest = d.getParameter(l, "W", 0, 0, false, LayerWorkspaceMgr.noWorkspaces());
        assertTrue(l.getParam("W") == outTest);    //Should be same object
        INDArray outTrain = d.getParameter(l, "W", 0, 0, true, LayerWorkspaceMgr.noWorkspaces());
        assertNotEquals(l.getParam("W"), outTrain);

        assertEquals(l.getParam("W"), Nd4j.ones(10, 10));

        int countZeros = Nd4j.getExecutioner().exec(new MatchCondition(outTrain, Conditions.equals(0))).z().getInt(0);
        int countOnes = Nd4j.getExecutioner().exec(new MatchCondition(outTrain, Conditions.equals(1))).z().getInt(0);

        assertEquals(100, countZeros + countOnes);  //Should only be 0 or 2
        //Stochastic, but this should hold for most cases
        assertTrue(countZeros >= 25 && countZeros <= 75);
        assertTrue(countOnes >= 25 && countOnes <= 75);
    }

}

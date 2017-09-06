package org.deeplearning4j.nn.conf.weightnoise;

import org.deeplearning4j.TestUtils;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.SigmoidSchedule;

import static org.junit.Assert.assertEquals;

public class TestWeightNoise {

    @Test
    public void testWeightNoiseConfigJson(){
        IWeightNoise[] weightNoises = new IWeightNoise[]{
                new DropConnect(0.5),
                new DropConnect(new SigmoidSchedule(ScheduleType.ITERATION, 0.5, 0.5, 100)),
                new WeightNoise(new NormalDistribution(0, 0.1))
        };

        for(IWeightNoise wn : weightNoises) {
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .weightNoise(wn)
                    .list()
                    .layer(new DenseLayer.Builder().nIn(10).nOut(10).build())
                    .layer(new DenseLayer.Builder().nIn(10).nOut(10).weightNoise(new DropConnect(0.25)).build())
                    .layer(new OutputLayer.Builder().nIn(10).nOut(10).build())
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            assertEquals(wn, ((BaseLayer)net.getLayer(0).conf().getLayer()).getWeightNoise());
            assertEquals(new DropConnect(0.25), ((BaseLayer)net.getLayer(1).conf().getLayer()).getWeightNoise());
            assertEquals(wn, ((BaseLayer)net.getLayer(2).conf().getLayer()).getWeightNoise());

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

            assertEquals(wn, ((BaseLayer)graph.getLayer(0).conf().getLayer()).getWeightNoise());
            assertEquals(new DropConnect(0.25), ((BaseLayer)graph.getLayer(1).conf().getLayer()).getWeightNoise());
            assertEquals(wn, ((BaseLayer)graph.getLayer(2).conf().getLayer()).getWeightNoise());

            TestUtils.testModelSerialization(graph);
        }
    }

}

package org.deeplearning4j.nn.graph;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.junit.Test;

public class TestComputationGraphNetwork {

    @Test
    public void testConfigurationBasic(){

        ComputationGraphConfiguration configuration = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInput(0, "input")
                .addLayer("firstLayer", new DenseLayer.Builder().nIn(4).nOut(5).build(),"input")
                .addLayer("outputLayer", new OutputLayer.Builder().nIn(5).nOut(3).build(),"firstLayer")
                .build();

        ComputationGraph graph = new ComputationGraph(configuration);
        graph.init();
    }

    @Test
    public void testFitBasicIris(){

        ComputationGraphConfiguration configuration = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInput(0, "input")
                .addLayer("firstLayer", new DenseLayer.Builder().nIn(4).nOut(5).build(),"input")
                .addLayer("outputLayer", new OutputLayer.Builder().nIn(5).nOut(3).build(),"firstLayer")
                .build();

        ComputationGraph graph = new ComputationGraph(configuration);
        graph.init();

        DataSetIterator iris = new IrisDataSetIterator(150,150);

        graph.fit(iris);
    }

}

package org.deeplearning4j.nn.graph;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

public class TestComputationGraphNetwork {

    @Test
    public void testConfigurationBasic(){

        ComputationGraphConfiguration configuration = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("input")
                .addLayer("firstLayer", new DenseLayer.Builder().nIn(4).nOut(5).build(),"input")
                .addLayer("outputLayer", new OutputLayer.Builder().nIn(5).nOut(3).build(),"firstLayer")
                .setOutputs("outputLayer")
                .pretrain(false).backprop(true)
                .build();

        ComputationGraph graph = new ComputationGraph(configuration);
        graph.init();

        //Get topological sort order
        int[] order = graph.topologicalSortOrder();
        int[] expOrder = new int[]{0,1,2};
        assertArrayEquals(expOrder,order);  //Only one valid order: 0 (input) -> 1 (firstlayer) -> 2 (outputlayer)

        INDArray params = graph.params();
        assertNotNull(params);

        int nParams = (4*5+5) + (5*3+3);
        assertEquals(nParams,params.length());

        INDArray arr = Nd4j.linspace(0,nParams,nParams);
        assertEquals(nParams,arr.length());

        graph.setParams(arr);
        params = graph.params();
        assertEquals(arr,params);

        //Number of inputs and outputs:
        assertEquals(1,graph.getNumInputArrays());
        assertEquals(1,graph.getNumOutputArrays());
    }

    @Test
    public void testFitBasicIris(){

        ComputationGraphConfiguration configuration = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("input")
                .addLayer("firstLayer", new DenseLayer.Builder().nIn(4).nOut(5).build(),"input")
                .addLayer("outputLayer", new OutputLayer.Builder().nIn(5).nOut(3).build(),"firstLayer")
                .setOutputs("outputLayer")
                .pretrain(false).backprop(true)
                .build();

        ComputationGraph graph = new ComputationGraph(configuration);
        graph.init();

        DataSetIterator iris = new IrisDataSetIterator(150,150);

        graph.fit(iris);
    }

}

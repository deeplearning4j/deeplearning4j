package org.deeplearning4j.nn.graph;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;

public class TestComputationGraphNetwork {

    private static ComputationGraphConfiguration getIrisGraphConfiguration(){
        return new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .graphBuilder()
                .addInputs("input")
                .addLayer("firstLayer", new DenseLayer.Builder().nIn(4).nOut(5).build(), "input")
                .addLayer("outputLayer", new OutputLayer.Builder().nIn(5).nOut(3).build(), "firstLayer")
                .setOutputs("outputLayer")
                .pretrain(false).backprop(true)
                .build();
    }

    private static MultiLayerConfiguration getIrisMLNConfiguration(){
        return new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list(2)
                .layer(0, new DenseLayer.Builder().nIn(4).nOut(5).build())
                .layer(1, new OutputLayer.Builder().nIn(5).nOut(3).build())
                .pretrain(false).backprop(true)
                .build();
    }

    private static int getNumParams(){
        //Number of parameters for both iris models
        return (4*5+5) + (5*3+3);
    }

    @Test
    public void testConfigurationBasic(){

        ComputationGraphConfiguration configuration = getIrisGraphConfiguration();

        ComputationGraph graph = new ComputationGraph(configuration);
        graph.init();

        //Get topological sort order
        int[] order = graph.topologicalSortOrder();
        int[] expOrder = new int[]{0,1,2};
        assertArrayEquals(expOrder,order);  //Only one valid order: 0 (input) -> 1 (firstlayer) -> 2 (outputlayer)

        INDArray params = graph.params();
        assertNotNull(params);

        int nParams = getNumParams();
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
    public void testForwardBasicIris(){

        ComputationGraphConfiguration configuration = getIrisGraphConfiguration();
        ComputationGraph graph = new ComputationGraph(configuration);
        graph.init();

        MultiLayerConfiguration mlc = getIrisMLNConfiguration();
        MultiLayerNetwork net = new MultiLayerNetwork(mlc);
        net.init();

        DataSetIterator iris = new IrisDataSetIterator(150,150);
        DataSet ds = iris.next();

        graph.setInput(0, ds.getFeatureMatrix());
        Map<String,INDArray> activations = graph.feedForward(false);
        assertEquals(3,activations.size()); //2 layers + 1 input node
        assertTrue(activations.containsKey("input"));
        assertTrue(activations.containsKey("firstLayer"));
        assertTrue(activations.containsKey("outputLayer"));

        //Now: set parameters of both networks to be identical. Then feedforward, and check we get the same outputs
        Nd4j.getRandom().setSeed(12345);
        int nParams = getNumParams();
        INDArray params = Nd4j.rand(1, nParams);
        graph.setParams(params.dup());
        net.setParams(params.dup());

        List<INDArray> mlnAct = net.feedForward(ds.getFeatureMatrix(),false);
        activations = graph.feedForward(ds.getFeatureMatrix(), false);

        assertEquals(mlnAct.get(0),activations.get("input"));
        assertEquals(mlnAct.get(1),activations.get("firstLayer"));
        assertEquals(mlnAct.get(2),activations.get("outputLayer"));
    }

    @Test
    public void testBackwardIrisBasic(){
        ComputationGraphConfiguration configuration = getIrisGraphConfiguration();
        ComputationGraph graph = new ComputationGraph(configuration);
        graph.init();

        MultiLayerConfiguration mlc = getIrisMLNConfiguration();
        MultiLayerNetwork net = new MultiLayerNetwork(mlc);
        net.init();

        DataSetIterator iris = new IrisDataSetIterator(150,150);
        DataSet ds = iris.next();

        //Now: set parameters of both networks to be identical. Then feedforward, and check we get the same outputs
        Nd4j.getRandom().setSeed(12345);
        int nParams = (4*5+5) + (5*3+3);
        INDArray params = Nd4j.rand(1, nParams);
        graph.setParams(params.dup());
        net.setParams(params.dup());

        INDArray input = ds.getFeatureMatrix();
        INDArray labels = ds.getLabels();
        graph.setInput(0, input.dup());
        graph.setLabel(0, labels.dup());

        net.setInput(input.dup());
        net.setLabels(labels.dup());

        //Compute gradients
        net.computeGradientAndScore();
        Pair<Gradient,Double> netGradScore = net.gradientAndScore();

        graph.computeGradientAndScore();
        Pair<Gradient,Double> graphGradScore = graph.gradientAndScore();

        assertEquals(netGradScore.getSecond(), graphGradScore.getSecond(), 1e-3);

        //Compare gradients
        Gradient netGrad = netGradScore.getFirst();
        Gradient graphGrad = graphGradScore.getFirst();

        assertNotNull(graphGrad);
        assertEquals(netGrad.gradientForVariable().size(), graphGrad.gradientForVariable().size());

        assertEquals(netGrad.getGradientFor("0_W"), graphGrad.getGradientFor("firstLayer_W"));
        assertEquals(netGrad.getGradientFor("0_b"),graphGrad.getGradientFor("firstLayer_b"));
        assertEquals(netGrad.getGradientFor("1_W"), graphGrad.getGradientFor("outputLayer_W"));
        assertEquals(netGrad.getGradientFor("1_b"),graphGrad.getGradientFor("outputLayer_b"));
    }

    @Test
    public void testIrisFit(){

        ComputationGraphConfiguration configuration = getIrisGraphConfiguration();
        ComputationGraph graph = new ComputationGraph(configuration);
        graph.init();

        MultiLayerConfiguration mlnConfig = getIrisMLNConfiguration();
        MultiLayerNetwork net = new MultiLayerNetwork(mlnConfig);
        net.init();

        Nd4j.getRandom().setSeed(12345);
        int nParams = getNumParams();
        INDArray params = Nd4j.rand(1,nParams);

        graph.setParams(params.dup());
        net.setParams(params.dup());


        DataSetIterator iris = new IrisDataSetIterator(75,150);

        net.fit(iris);
        iris.reset();

        graph.fit(iris);

        //Check that parameters are equal for both models after fitting:
        INDArray paramsMLN = net.params();
        INDArray paramsGraph = graph.params();

        assertNotEquals(params,paramsGraph);
        assertEquals(paramsMLN,paramsGraph);
    }

}

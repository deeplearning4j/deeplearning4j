package org.deeplearning4j.nn.graph;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.api.Layer;

import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;
import static org.junit.Assert.assertTrue;


/**
 * Created by nyghtowl on 1/15/16.
 */
public class TestCompGraphMulti {

    protected ComputationGraphConfiguration conf;
    protected ComputationGraph graph;
    protected DataSetIterator cifar;
    protected DataSet ds;

    protected static ComputationGraphConfiguration getMultiInputGraphConfig(){
        return new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .graphBuilder()
                .addInputs("input")
                .addLayer("cnn1", new ConvolutionLayer.Builder(4,4).stride(2,2).nIn(1).nOut(3).build(), "input")
                .addLayer("cnn2", new ConvolutionLayer.Builder(4,4).stride(2,2).nIn(1).nOut(3).build(), "input")
                .addLayer("max1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).build(), "cnn1", "cnn2")
                .addLayer("dnn1", new DenseLayer.Builder().nIn(15*15*3).nOut(7).build(), "max1")
                .addLayer("max2", new SubsamplingLayer.Builder().build(), "max1")
                .addLayer("output", new OutputLayer.Builder().nIn(7).nOut(10).build(), "dnn1", "max2")
                .setOutputs("output")
                .inputPreProcessor("cnn1", new FeedForwardToCnnPreProcessor(32, 32, 3))
                .inputPreProcessor("cnn2", new FeedForwardToCnnPreProcessor(32, 32, 3))
                .inputPreProcessor("dnn1", new CnnToFeedForwardPreProcessor(15, 15, 3))
                .pretrain(false).backprop(true)
                .build();
    }

    protected static DataSetIterator getDS() {
        return new CifarDataSetIterator(5,5);
    }

    protected static int getNumParams(){
        return (3*1*4*4+3) + (3*1*4*4+3) + (7*15*15*3+7) + (7*10+10);
    }

    @Before
    public void beforeDo(){
        conf = getMultiInputGraphConfig();
        graph = new ComputationGraph(conf);
        graph.init();

        cifar = getDS();
        ds = cifar.next();

    }

    @Test
    public void testConfigBasic(){

        int[] order = graph.topologicalSortOrder();
        int[] expOrder = new int[]{0,1,3,7,6,4,5,8,2};
        assertArrayEquals(expOrder,order);  //Only one valid order: 0 (input) -> 1 (firstlayer) -> 2 (outputlayer)

        INDArray params = graph.params();
        assertNotNull(params);

        int nParams = getNumParams();
        assertEquals(nParams,params.length());

        INDArray arr = Nd4j.linspace(0, nParams, nParams);
        assertEquals(nParams,arr.length());

        graph.setParams(arr);
        params = graph.params();
        assertEquals(arr,params);

        //Number of inputs and outputs:
        assertEquals(1,graph.getNumInputArrays());
        assertEquals(1,graph.getNumOutputArrays());

    }

    @Test
    public void testForwardBasic(){

        graph.setInput(0, ds.getFeatureMatrix());
        // TODO issue with CNN preOutput z calc in tensorMatrixMul - need to review using weights 5*1*2*2 and col includes 16 x 16 h & w
        Map<String,INDArray> activations = graph.feedForward(true);
        assertEquals(7, activations.size()); //2 layers + 1 input node
        assertTrue(activations.containsKey("input"));
        assertTrue(activations.containsKey("cnn1"));
        assertTrue(activations.containsKey("outputLayer"));

        // Check feedforward activations

    }

    @Test
    public void testBackwardIrisBasic(){

        //Now: set parameters of both networks to be identical. Then feedforward, and check we get the same outputs
        Nd4j.getRandom().setSeed(12345);

        INDArray input = ds.getFeatureMatrix();
        INDArray labels = ds.getLabels();
        graph.setInput(0, input.dup());
        graph.setLabel(0, labels.dup());

        //Compute gradients
        graph.computeGradientAndScore();
        Pair<Gradient,Double> graphGradScore = graph.gradientAndScore();

        // Check gradients
    }

    @Test
    public void testEvaluation(){
        Evaluation evalExpected = new Evaluation();
        // TODO setup graph output evaluation
//        INDArray out = graph.output(ds.getFeatureMatrix(), Layer.TrainingMode.TEST);
//        evalExpected.eval(ds.getLabels(), out);

        // Check evaluation results

    }

}

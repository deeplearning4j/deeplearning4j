package org.deeplearning4j.nn.conf;


import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.graph.nodes.ElementWiseNode;
import org.deeplearning4j.nn.graph.nodes.MergeNode;
import org.deeplearning4j.nn.graph.nodes.SubsetNode;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

public class ComputationGraphConfigurationTest {

    @Test
    public void testJSONBasic(){
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0,1))
                .updater(Updater.NONE).learningRate(1.0)
                .graphBuilder()
                .addInputs("input")
                .addLayer("firstLayer", new DenseLayer.Builder().nIn(4).nOut(5).activation("tanh").build(), "input")
                .addLayer("outputLayer", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation("softmax").nIn(5).nOut(3).build(), "firstLayer")
                .setOutputs("outputLayer")
                .pretrain(false).backprop(true)
                .build();

        String json = conf.toJson();
        ComputationGraphConfiguration conf2 = ComputationGraphConfiguration.fromJson(json);

        assertEquals(json,conf2.toJson());
        assertEquals(conf, conf2);
    }

    @Test
    public void testJSONBasic2(){
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .graphBuilder()
                .addInputs("input")
                .addLayer("cnn1", new ConvolutionLayer.Builder(2,2).stride(2,2).nIn(1).nOut(5).build(), "input")
                .addLayer("cnn2", new ConvolutionLayer.Builder(2,2).stride(2,2).nIn(1).nOut(5).build(), "input")
                .addLayer("max1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).build(), "cnn1", "cnn2")
                .addLayer("dnn1", new DenseLayer.Builder().nOut(7).build(), "max1")
                .addLayer("max2", new SubsamplingLayer.Builder().build(), "max1")
                .addLayer("output", new OutputLayer.Builder().nIn(7).nOut(10).build(), "dnn1", "max2")
                .setOutputs("output")
                .inputPreProcessor("cnn1", new FeedForwardToCnnPreProcessor(32, 32, 3))
                .inputPreProcessor("cnn2", new FeedForwardToCnnPreProcessor(32, 32, 3))
                .inputPreProcessor("dnn1", new CnnToFeedForwardPreProcessor(8, 8, 5))
                .pretrain(false).backprop(true)
                .build();

        String json = conf.toJson();
        ComputationGraphConfiguration conf2 = ComputationGraphConfiguration.fromJson(json);

        assertEquals(json,conf2.toJson());
        assertEquals(conf, conf2);
    }

    @Test
    public void testJSONWithGraphNodes(){

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .graphBuilder()
                .addInputs("input1", "input2")
                .addLayer("cnn1", new ConvolutionLayer.Builder(2, 2).stride(2, 2).nIn(1).nOut(5).build(), "input1")
                .addLayer("cnn2", new ConvolutionLayer.Builder(2,2).stride(2,2).nIn(1).nOut(5).build(), "input2")
                .addNode("merge1", new MergeNode(), "cnn1", "cnn2")
                .addNode("subset1", new SubsetNode(0,1), "merge1")
                .addLayer("dense1", new DenseLayer.Builder().nIn(20).nOut(5).build(), "subset1")
                .addLayer("dense2", new DenseLayer.Builder().nIn(20).nOut(5).build(), "subset1")
                .addNode("add", new ElementWiseNode(ElementWiseNode.Op.Add), "dense1", "dense2")
                .build();

        String json = conf.toJson();
        System.out.println(json);

        ComputationGraphConfiguration conf2 = ComputationGraphConfiguration.fromJson(json);

        assertEquals(json,conf2.toJson());
        assertEquals(conf, conf2);
    }
}

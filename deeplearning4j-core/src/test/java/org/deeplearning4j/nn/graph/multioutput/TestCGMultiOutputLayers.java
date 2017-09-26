package org.deeplearning4j.nn.graph.multioutput;

import org.deeplearning4j.nn.api.activations.Activations;
import org.deeplearning4j.nn.api.activations.ActivationsFactory;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.multioutput.testlayers.SplitDenseLayerConf;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

import static org.junit.Assert.assertEquals;

public class TestCGMultiOutputLayers {

    private static final ActivationsFactory af = ActivationsFactory.getInstance();

    @Test
    public void testMultipleOutputSimple(){

        int nIn = 5;
        int minibatch = 3;

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .graphBuilder()
                .addInputs("in")
                .layer("first", new DenseLayer.Builder().nIn(nIn).nOut(5).build(), "in")
                .layer("second", new SplitDenseLayerConf.Builder().nIn(5).nOut(5).build(), "first")
                .layer("out1", new OutputLayer.Builder().nIn(2).nOut(3).build(), "second/0")
                .layer("out2", new OutputLayer.Builder().nIn(3).nOut(4).build(), "second/1")
                .setOutputs("out1", "out2")
                .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        Map<String,Activations> act = net.feedForward(af.create(Nd4j.create(minibatch, nIn)), true);

        assertEquals(5, act.size());    //Including input

        for( Map.Entry<String,Activations> e : act.entrySet()){
            if(e.getKey().equals("second")){
                assertEquals(2, e.getValue().size());
            } else {
                assertEquals(1, e.getValue().size());
            }
        }
    }

}

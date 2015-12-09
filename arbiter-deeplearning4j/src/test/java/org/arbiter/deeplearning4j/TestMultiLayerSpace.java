package org.arbiter.deeplearning4j;

import org.apache.commons.math3.distribution.UniformIntegerDistribution;
import org.arbiter.optimize.distribution.DegenerateIntegerDistribution;
import org.arbiter.optimize.parameter.FixedValue;
import org.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.arbiter.optimize.parameter.discrete.DiscreteParameterSpace;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class TestMultiLayerSpace {

    @Test
    public void testBasic(){

        MultiLayerConfiguration expected = new NeuralNetConfiguration.Builder()
                .learningRate(0.005)
                .seed(12345)
                .list(3)
                .layer(0, new DenseLayer.Builder().nIn(10).nOut(10).build())
                .layer(1, new DenseLayer.Builder().nIn(10).nOut(10).build())
                .layer(2, new OutputLayer.Builder().nIn(10).nOut(5).build())
                .backprop(true).pretrain(false)
                .build();

        LayerSpace ls1 = new LayerSpace.Builder()
                .layer(DenseLayer.class)
                .add("nIn", new FixedValue<Integer>(10))
                .add("nOut", new FixedValue<Integer>(10))
                .numLayersDistribution(new DegenerateIntegerDistribution(2))
                .build();

        LayerSpace ls2 = new LayerSpace.Builder()
                .layer(OutputLayer.class)
                .add("nOut", new FixedValue<Integer>(5))
                .numLayersDistribution(new DegenerateIntegerDistribution(1))
                .build();

        MultiLayerSpace mls = new MultiLayerSpace.Builder()
                .add("learningRate", new FixedValue<Double>(0.005))
                .add("backprop", new FixedValue<Boolean>(true))
                .add("pretrain", new FixedValue<Boolean>(false))
                .add("seed", new FixedValue<Integer>(12345))
                .addLayer(ls1)
                .addLayer(ls2)
                .build();

        MultiLayerConfiguration conf = mls.randomCandidate();

        assertEquals(expected, conf);
    }

    @Test
    public void testBasic2(){

        LayerSpace ls1 = new LayerSpace.Builder()
                .layer(DenseLayer.class)
                .numLayersDistribution(new UniformIntegerDistribution(1,3))
                .add("nIn", new FixedValue<Integer>(10))
                .add("nOut", new FixedValue<Integer>(10))
                .add("activation", new DiscreteParameterSpace<String>("relu","tanh"))
                .build();

        LayerSpace ls2 = new LayerSpace.Builder()
                .layer(OutputLayer.class)
                .add("nOut", new FixedValue<Integer>(5))
                .add("activation", new FixedValue<Object>("softmax"))
                .build();

        MultiLayerSpace mls = new MultiLayerSpace.Builder()
                .add("pretrain", new FixedValue<>(false))
                .add("backprop", new FixedValue<>(true))
                .add("learningRate", new ContinuousParameterSpace(0.0001,0.1))
                .add("regularization", new FixedValue<Object>(true))
                .add("l2", new ContinuousParameterSpace(0.2, 0.5))
                .addLayer(ls1)
                .addLayer(ls2)
                .build();

        int[] nLayerCounts = new int[3];
        int reluCount = 0;
        int tanhCount = 0;
        for( int i=0; i<50; i++ ){

            MultiLayerConfiguration conf = mls.randomCandidate();
            assertEquals(false, conf.isPretrain());
            assertEquals(true, conf.isBackprop());

            int nLayers = conf.getConfs().size();
            assertTrue(nLayers >= 2 && nLayers <= 4);   //1-3 dense layers + 1 output layer: 2 to 4

            int nLayersExOutputLayer = nLayers - 1;
            nLayerCounts[nLayersExOutputLayer-1]++;

            for( int j=0; j<nLayers; j++ ){
                NeuralNetConfiguration layerConf = conf.getConf(j);

                double lr = layerConf.getLayer().getLearningRate();
                assertTrue(lr >= 0.0001 && lr <= 0.1);
                assertEquals(true, layerConf.isUseRegularization());
                double l2 = layerConf.getLayer().getL2();
                assertTrue( l2 >= 0.2 && l2 <= 0.5);

                if(j == nLayers-1){ //Output layer
                    assertEquals("softmax",layerConf.getLayer().getActivationFunction());
                } else {
                    String actFn = layerConf.getLayer().getActivationFunction();
                    assertTrue("relu".equals(actFn) || "tanh".equals(actFn));
                    if("relu".equals(actFn)) reluCount++;
                    else tanhCount++;
                }
            }
        }

        System.out.println("Number of layers: " + Arrays.toString(nLayerCounts));
        System.out.println("ReLU vs. Tanh: " + reluCount + "\t" + tanhCount);

    }


}

package org.deeplearning4j.nn.layers;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.lang.reflect.Field;

import static junit.framework.TestCase.assertNotNull;
import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

public class TestDropout {

    @Test
    public void testDropoutSimple() throws Exception {
        //Testing dropout with a single layer
        //Layer input: values should be set to either 0.0 or 2.0x original value

        int nIn = 8;
        int nOut = 4;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.SGD).iterations(1)
                .regularization(true).dropOut(0.5)
                .list()
                .layer(0, new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE)
                        .nIn(nIn).nOut(nOut).weightInit(WeightInit.XAVIER).build())
                .backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        Field dropoutMaskField = BaseLayer.class.getDeclaredField("dropoutMask");
        dropoutMaskField.setAccessible(true);

        int nTests = 15;

        Nd4j.getRandom().setSeed(12345);
        int noDropoutCount = 0;
        for( int i=0; i<nTests; i++ ){
            INDArray in = Nd4j.rand(1,nIn);
            INDArray out = Nd4j.rand(1,nOut);
            INDArray inCopy = in.dup();

            net.fit(new DataSet(in,out));

            //Check that original input was not modified (important for multiple iterations, or reuse of input)
            assertEquals(inCopy,in);

            //Check that dropout was actually applied (and applied exactly once)
            INDArray dropoutMask = (INDArray)dropoutMaskField.get(net.getLayer(0));
            assertNotNull(dropoutMask);

            INDArray l0Input = net.getLayer(0).input();
            if(inCopy.equals(l0Input)){ //l0Input should differ from original input (probably)
                int countNonZero = 0;
                for( int j=0; j<dropoutMask.length(); j++ ){
                    if(dropoutMask.getDouble(j) != 0.0) countNonZero++;
                }

                if(countNonZero != nIn){
                    //Some inputs should have been dropped out
                    fail("No dropout occurred, but dropoutMask contains 0.0 elements. mask = " + dropoutMask);
                } else {
                    noDropoutCount++;
                }
            } else {
                //Dropout occurred. Expect inputs to be either scaled 2x original, or set to 0.0 (with dropout = 0.5)
                for( int j=0; j<inCopy.length(); j++ ){
                    double origValue = inCopy.getDouble(j);
                    double doValue = l0Input.getDouble(j);
                    double maskValue = dropoutMask.getDouble(j);
                    assertTrue(maskValue == 0.0 || maskValue == 2.0);
                    if(maskValue == 0.0){
                        //Input was dropped out
                        assertEquals(0.0, doValue, 0.0);
                    } else {
                        //Input was kept -> should be scaled by factor of (1.0/0.5 = 2)
                        assertEquals(origValue*2.0, doValue, 0.0001);
                    }
                }
            }

            //Do forward pass
            //(1) ensure dropout ISN'T being applied for forward pass at test time
            //(2) ensure dropout ISN'T being applied for test time scoring
            //If dropout is applied at test time: outputs + score will differ between passes
            INDArray in2 = Nd4j.rand(1,nIn);
            INDArray out2 = Nd4j.rand(1,nOut);
            INDArray outTest1 = net.output(in2, false);
            INDArray outTest2 = net.output(in2, false);
            INDArray outTest3 = net.output(in2, false);
            assertEquals(outTest1,outTest2);
            assertEquals(outTest1, outTest3);

            double score1 = net.score(new DataSet(in2,out2), false);
            double score2 = net.score(new DataSet(in2,out2), false);
            double score3 = net.score(new DataSet(in2,out2), false);
            assertEquals(score1,score2,0.0);
            assertEquals(score1,score3,0.0);
        }

        if(noDropoutCount >= nTests / 3){
            //at 0.5 dropout ratio and more than a few inputs, expect only a very small number of instances where
            //no dropout occurs, just due to random chance
            fail("Too many instances of dropout not being applied");
        }
    }


    @Test
    public void testDropoutMultiLayer() throws Exception {
        //Testing dropout with multiple layers
        //Layer input: values should be set to either 0.0 or 2.0x original value
        //However: we don't have access to 'original' activations easily
        //Instead: use sigmoid + weight initialization that saturates

        int nIn = 8;
        int layerSize = 10;
        int nOut = 4;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.SGD).iterations(1)
                .regularization(true).dropOut(0.5)
                .learningRate(1e-9)
                .weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(10,11))   //Weight init to cause sigmoid saturation
                .list()
                .layer(0, new DenseLayer.Builder().activation("sigmoid").nIn(nIn).nOut(layerSize).build())
                .layer(1, new DenseLayer.Builder().activation("sigmoid").nIn(layerSize).nOut(layerSize).build())
                .layer(2, new DenseLayer.Builder().activation("sigmoid").nIn(layerSize).nOut(layerSize).build())
                .layer(3, new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE)
                        .nIn(layerSize).nOut(nOut).weightInit(WeightInit.XAVIER).build())
                .backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        Field dropoutMaskField = BaseLayer.class.getDeclaredField("dropoutMask");
        dropoutMaskField.setAccessible(true);

        int nTests = 15;

        Nd4j.getRandom().setSeed(12345);
        int noDropoutCount = 0;
        for( int i=0; i<nTests; i++ ){
            INDArray in = Nd4j.rand(1,nIn);
            INDArray out = Nd4j.rand(1,nOut);
            INDArray inCopy = in.dup();

            net.fit(new DataSet(in,out));

            //Check that original input was not modified (important for multiple iterations, or reuse of input)
            assertEquals(inCopy,in);

            //Check that dropout was actually applied (and applied exactly once)
            INDArray dropoutMask = (INDArray)dropoutMaskField.get(net.getLayer(0));
            assertNotNull(dropoutMask);

            INDArray l0Input = net.getLayer(0).input();
            if(inCopy.equals(l0Input)){ //l0Input should differ from original input (probably)
                int countNonZero = 0;
                for( int j=0; j<dropoutMask.length(); j++ ){
                    if(dropoutMask.getDouble(j) != 0.0) countNonZero++;
                }

                if(countNonZero != nIn){
                    //Some inputs should have been dropped out
                    fail("No dropout occurred, but dropoutMask contains 0.0 elements. mask = " + dropoutMask);
                } else {
                    noDropoutCount++;
                }
            } else {
                //Dropout occurred. Expect inputs to be either scaled 2x original, or set to 0.0 (with dropout = 0.5)
                for( int j=0; j<inCopy.length(); j++ ){
                    double origValue = inCopy.getDouble(j);
                    double doValue = l0Input.getDouble(j);
                    double maskValue = dropoutMask.getDouble(j);
                    assertTrue(maskValue == 0.0 || maskValue == 2.0);
                    if(maskValue == 0.0){
                        //Input was dropped out
                        assertEquals(0.0, doValue, 0.0);
                    } else {
                        //Input was kept -> should be scaled by factor of (1.0/0.5 = 2)
                        assertEquals(origValue*2.0, doValue, 0.0001);
                    }
                }
            }

            //Check other layers. Don't know pre-dropout values in general, but using saturated sigmoids -> inputs should
            //all be ~1.0 before dropout -> either 0 or ~2.0 after dropout
            for( int j=1; j<4; j++ ){
                dropoutMask = (INDArray)dropoutMaskField.get(net.getLayer(j));
                assertNotNull(dropoutMask);

                INDArray ljInput = net.getLayer(j).input();
                for( int k=0; k<ljInput.length(); k++ ){
                    double doValue = ljInput.getDouble(j);
                    double maskValue = dropoutMask.getDouble(j);
                    assertTrue(maskValue == 0.0 || maskValue == 2.0);
                    if(maskValue == 0.0){
                        //Input was dropped out
                        assertEquals(0.0, doValue, 0.0);
                    } else {
                        //Input was kept -> should be scaled by factor of (1.0/0.5 = 2)
                        assertEquals(2.0, doValue, 0.1);    //Sigmoid is saturated -> inputs should be ~1.0 -> 2.0 after dropout
                    }
                }
            }

            //Do forward pass
            //(1) ensure dropout ISN'T being applied for forward pass at test time
            //(2) ensure dropout ISN'T being applied for test time scoring
            //If dropout is applied at test time: outputs + score will differ between passes
            INDArray in2 = Nd4j.rand(1,nIn);
            INDArray out2 = Nd4j.rand(1,nOut);
            INDArray outTest1 = net.output(in2, false);
            INDArray outTest2 = net.output(in2, false);
            INDArray outTest3 = net.output(in2, false);
            assertEquals(outTest1,outTest2);
            assertEquals(outTest1, outTest3);

            double score1 = net.score(new DataSet(in2,out2), false);
            double score2 = net.score(new DataSet(in2,out2), false);
            double score3 = net.score(new DataSet(in2,out2), false);
            assertEquals(score1,score2,0.0);
            assertEquals(score1,score3,0.0);
        }

        if(noDropoutCount >= nTests / 3){
            //at 0.5 dropout ratio and more than a few inputs, expect only a very small number of instances where
            //no dropout occurs, just due to random chance
            fail("Too many instances of dropout not being applied");
        }
    }



}

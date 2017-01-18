package org.deeplearning4j.nn.layers.pooling;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Random;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 18/01/2017.
 */
public class GlobalPoolingMaskingTests {

    @Test
    public void testMaskingRnn(){


        int timeSeriesLength = 5;
        int nIn = 5;
        int layerSize = 4;
        int nOut = 2;
        int miniBatchSize = 1;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .regularization(false)
                .updater(Updater.NONE)
                .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1.0))
                .seed(12345L)
                .list()
                .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(layerSize).activation(Activation.TANH).build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer.Builder().poolingType(PoolingType.AVG).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nIn(layerSize).nOut(nOut).build())
                .pretrain(false).backprop(true)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        Random r = new Random(12345L);
        INDArray input = Nd4j.zeros(miniBatchSize, nIn, timeSeriesLength);
        for (int i = 0; i < miniBatchSize; i++) {
            for (int j = 0; j < nIn; j++) {
                for (int k = 0; k < timeSeriesLength; k++) {
                    input.putScalar(new int[]{i, j, k}, r.nextDouble() - 0.5);
                }
            }
        }

        INDArray mask = Nd4j.create(miniBatchSize, timeSeriesLength-1);
        for( int i=0; i<miniBatchSize; i++ ){
            for( int j=0; j<timeSeriesLength-1; j++ ){
                mask.putScalar(i,j,1.0);
            }
        }

        INDArray labels = Nd4j.zeros(miniBatchSize, nOut);
        for (int i = 0; i < miniBatchSize; i++) {
            int idx = r.nextInt(nOut);
            labels.putScalar(i, idx, 1.0);
        }


        INDArray inputSubset = input.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0,timeSeriesLength-1));

        INDArray outSubset = net.output(inputSubset);

        net.setLayerMaskArrays(mask, null);
        INDArray outMask = net.output(input);

        assertEquals(outSubset, outMask);
    }


}

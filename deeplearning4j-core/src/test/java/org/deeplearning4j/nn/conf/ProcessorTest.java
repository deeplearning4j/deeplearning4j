package org.deeplearning4j.nn.conf;

import static org.junit.Assert.*;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.junit.Test;

/**
 * Created by merlin on 7/31/15.
 */
public class ProcessorTest {


    private static MultiLayerConfiguration getIrisSimpleConfig( int[] hiddenLayerSizes, String activationFunction, int iterations ) {
        MultiLayerConfiguration c = new NeuralNetConfiguration.Builder()
                .nIn(4)
                .nOut(3)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0, 0.1))

                .activationFunction(activationFunction)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)

                .iterations(iterations)
                .batchSize(1)
                .constrainGradientToUnitNorm(false)
                .corruptionLevel(0.0)

                .learningRate(0.1)

                .regularization(false)
                .l1(0.0)
                .l2(0.0)
                .dropOut(0.0)
                .momentum(0.0)
                .applySparsity(false)
                .sparsity(0.0)
                .seed(12345L)
                .useDropConnect(false)

                .list(2)
                .layer(0, new RBM.Builder()
                        .nIn(4)
                        .nOut(3)
                        .build())
                .layer(1, new OutputLayer.Builder()
                        .nIn(3)
                        .nOut(3)
                        .activation("softmax")
                        .build())
                .build();


        return c;
    }


}
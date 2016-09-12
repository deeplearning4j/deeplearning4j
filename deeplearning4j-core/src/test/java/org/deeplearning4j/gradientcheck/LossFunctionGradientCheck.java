package org.deeplearning4j.gradientcheck;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.*;

/**
 * Created by Alex on 12/09/2016.
 */
public class LossFunctionGradientCheck {

    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-10;

    @Test
    public void lossFunctionGradientCheck(){

        ILossFunction[] lossFunctions = new ILossFunction[]{
                new LossBinaryXENT(),
                new LossCosineProximity(),
                new LossHinge(),
                new LossKLD(),
                new LossMAE(),
                new LossMAPE(),
                new LossMCXENT(),
                new LossMSE(),
                new LossMSLE(),
                new LossNegativeLogLikelihood(),
                new LossPoisson(),
                new LossSquaredHinge()
        };

        String[] outputActivationFn = new String[]{
                "sigmoid",  //xent
                "tanh",     //cosine
                "",         //hinge
                "softmax",  //kld
                "identity", //mae
                "identity", //mape
                "softmax",  //mcxent
                "identity", //mse
                "identity", //msle
                "softmax",  //nll
                "",         //poisson
                ""          //squared hinge
        };

        int[] nOut = new int[]{
                1,          //xent
                5,          //cosine
                -1,         //hinge - todo
                -1,         //kld
                3,          //mae
                3,          //mape
                3,          //mcxent
                3,          //mse
                3,          //msle
                3,          //nll
                -1,         //poisson
                -1          //squared hinge
        };


        for( int i=0; i<lossFunctions.length; i++ ){


            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .iterations(1).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(Updater.NONE)
                    .regularization(false)
                    .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0,2))
                    .list()
                    .layer(0, new DenseLayer.Builder().nIn(4).nOut(4).activation("tanh").build())
                    .layer(1, new OutputLayer.Builder()
                            .lossFunction(lossFunctions[i])
                            .activation(outputActivationFn[i])
                            .nIn(4).nOut(nOut[i])
                            .build())
                    .pretrain(false).backprop(true).build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();




        }



    }

    private static INDArray[] getFeaturesAndLabels(ILossFunction l, int nIn, int nOut, long seed){
        Nd4j.getRandom().setSeed(seed);
        INDArray[] ret = new INDArray[2];
        switch(l.getClass().getSimpleName()){
            case "LossBinaryXENT":

                break;
            case "LossCosineProximity":

                break;
            case "LossHinge":

                break;
            case "LossKLD":

                break;
            case "LossMAE":
                break;
            case "LossMAPE":
                break;
            case "LossMCXENT":

                break;
            case "LossMSE":
            case "LossMSLE":

                break;
            case "LossNegativeLogLikelihood":

                break;
            case "LossPoisson":

                break;
            case "LossSquaredHinge":

                break;
            default:
                throw new IllegalArgumentException("Unknown class: " + l.getClass().getSimpleName());
        }

        return ret;
    }

}

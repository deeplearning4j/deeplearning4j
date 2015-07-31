package org.deeplearning4j.nn.conf;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.override.ConfOverride;
import org.deeplearning4j.nn.layers.convolution.preprocessor.ConvolutionInputPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

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

                .layer(new RBM())
                .learningRate(0.1)

                .regularization(false)
                .l1(0.0)
                .l2(0.0)
                .dropOut(0.0)
                .momentum(0.0)
                .applySparsity(false)
                .sparsity(0.0)
                .seed(12345L)

                .list(2)
                .useDropConnect(false)
                .layer(1, new OutputLayer.Builder()
                        .nIn()
                        .nOut()
                        .build()
                )
                .override(hiddenLayerSizes.length, new ConfOverride() {
                    @Override
                    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                        builder.activationFunction("softmax");
                        builder;
                        builder.weightInit(WeightInit.DISTRIBUTION);
                        builder.dist(new NormalDistribution(0, 0.1));
                    }
                }).build();


        return c;
    }


}
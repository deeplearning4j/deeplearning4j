package org.deeplearning4j.integration.testcases;

import org.deeplearning4j.integration.TestCase;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping1D;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class CNN1DTestCases {


    /**
     * A simple synthetic CNN 1d test case using all CNN 1d layers:
     * Subsampling, Upsampling, Convolution, Cropping, Zero padding
     */
    public static TestCase getCnn1dTestCaseSynthetic(){
        return new Cnn1dSyntheticTestCase();
    }

    private static class Cnn1dSyntheticTestCase extends RNNTestCases.RnnCsvSequenceClassificationTestCase1 {

        private Cnn1dSyntheticTestCase(){
            testName = "Cnn1dTestCaseSynthetic";
            testType = TestType.RANDOM_INIT;
            testPredictions = true;
            testTrainingCurves = true;
            testGradients = true;
            testParamsPostTraining = true;
            testParallelInference = true;
            testEvaluation = true;
            testOverfitting = false;    //Not really possible with this config... not enough context by the time it reaches output layer

            labels2d = false;
        }

        @Override
        public Object getConfiguration(){
            int kernel = 4;
            int stride = 1;
            int padding = 0;
            int nIn = 1;

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(12345)
                    .updater(new AdaGrad(0.001))
                    .weightInit(WeightInit.XAVIER)
                    .convolutionMode(ConvolutionMode.Same)
                    .activation(Activation.SOFTSIGN)
                    .list()
                    .layer(new Convolution1DLayer.Builder().kernelSize(kernel)
                            .stride(stride).padding(padding).nOut(6)
                            .build())
                    .layer(new Cropping1D.Builder(1).build())
                    .layer(new Convolution1DLayer.Builder().kernelSize(kernel)
                            .stride(stride).padding(padding).nOut(32).build())
                    .layer(new ZeroPadding1DLayer.Builder(1).build())
                    .layer(new Subsampling1DLayer.Builder().kernelSize(2).stride(2).poolingType(PoolingType.AVG).build())
                    .layer(new Convolution1DLayer.Builder().kernelSize(kernel)
                            .stride(stride).padding(padding).nOut(64).build())
                    .layer(new Upsampling1D.Builder().size(2).build())
                    .layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                            .activation(Activation.SOFTMAX).nOut(6).build())
                    .setInputType(InputType.recurrent(nIn)).build();

            return conf;
        }
    }

}

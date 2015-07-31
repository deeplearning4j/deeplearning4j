package org.deeplearning4j.nn.layers.convolution;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.layers.convolution.preprocessor.ConvolutionInputPreProcessor;
import org.deeplearning4j.nn.layers.convolution.preprocessor.ConvolutionOutputPostProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by merlin on 7/31/15.
 */
public class CNNProcessorTest {

    public static MultiLayerConfiguration getCNNCong(String[] args) throws Exception {
        Nd4j.MAX_SLICES_TO_PRINT = -1;
        Nd4j.MAX_ELEMENTS_PER_SLICE = -1;

        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10;
        int numSamples = 100;
        int batchSize = 100;
        int iterations = 10;
        int splitTrainNum = (int) (batchSize * .8);
        int seed = 123;
        int listenerFreq = iterations / 5;
        DataSet mnist;
        SplitTestAndTrain trainTest;
        DataSet trainInput;
        List<INDArray> testInput = new ArrayList<>();
        List<INDArray> testLabels = new ArrayList<>();
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        DataSetIterator mnistIter = new MnistDataSetIterator(batchSize, numSamples, true);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .batchSize(batchSize)
                .iterations(iterations)
                .weightInit(WeightInit.XAVIER)
                .activationFunction("relu")
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .list(3)
                .layer(0, new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(new int[]{9, 9}, Convolution.Type.VALID)
                        .nIn(numRows * numColumns)
                        .nOut(8)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.poolingType.MAX, new int[]{2, 2})
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(8)
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .hiddenLayerSizes(50)
                .inputPreProcessor(0, new ConvolutionInputPreProcessor(numRows, numColumns))
                .preProcessor(1, new ConvolutionOutputPostProcessor())
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

    }
}

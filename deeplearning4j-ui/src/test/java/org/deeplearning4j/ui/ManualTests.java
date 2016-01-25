package org.deeplearning4j.ui;

import org.canova.image.loader.LFWLoader;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.activation.UpdateActivationIterationListener;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.deeplearning4j.ui.weights.WeightsVisualizationListener;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Test environment for building/debugging UI.
 *
 * @author raver119@gmail.com
 */
@Ignore
public class ManualTests {

    private static Logger log = LoggerFactory.getLogger(ManualTests.class);

    @Test
    public void testLaunch() throws Exception {

        UiServer server = UiServer.getInstance();

        System.out.println("http://localhost:" + server.getPort()+ "/");

        Thread.sleep(10000000000L);

        new ScoreIterationListener(100);
    }

    @Test
    public void testHistograms() throws Exception {
        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10;
        int numSamples = 60000;
        int batchSize = 100;
        int iterations = 10;
        int seed = 123;
        int listenerFreq = batchSize / 5;

        log.info("Load data....");
        DataSetIterator iter = new MnistDataSetIterator(batchSize,numSamples,true);

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .iterations(iterations)
                .momentum(0.5)
                .momentumAfter(Collections.singletonMap(3, 0.9))
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .list(4)
                .layer(0, new RBM.Builder().nIn(numRows*numColumns).nOut(500)
                        .weightInit(WeightInit.XAVIER).lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                        .visibleUnit(RBM.VisibleUnit.BINARY)
                        .hiddenUnit(RBM.HiddenUnit.BINARY)
                        .build())
                .layer(1, new RBM.Builder().nIn(500).nOut(250)
                        .weightInit(WeightInit.XAVIER).lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                        .visibleUnit(RBM.VisibleUnit.BINARY)
                        .hiddenUnit(RBM.HiddenUnit.BINARY)
                        .build())
                .layer(2, new RBM.Builder().nIn(250).nOut(200)
                        .weightInit(WeightInit.XAVIER).lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                        .visibleUnit(RBM.VisibleUnit.BINARY)
                        .hiddenUnit(RBM.HiddenUnit.BINARY)
                        .build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation("softmax")
                        .nIn(200).nOut(outputNum).build())
                .pretrain(true).backprop(false)
                .build();

        ;

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq),(IterationListener) new HistogramIterationListener(listenerFreq)));

        log.info("Train model....");
        model.fit(iter); // achieves end to end pre-training

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum);

        DataSetIterator testIter = new MnistDataSetIterator(100,10000);
        while(testIter.hasNext()) {
            DataSet testMnist = testIter.next();
            INDArray predict2 = model.output(testMnist.getFeatureMatrix());
            eval.eval(testMnist.getLabels(), predict2);
        }

        log.info(eval.stats());
        log.info("****************Example finished********************");
    }


    @Test
    public void testActivations() throws Exception {
        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10;
        int numSamples = 60000;
        int batchSize = 100;
        int iterations = 10;
        int seed = 123;
        int listenerFreq = batchSize / 5;

        log.info("Load data....");
        DataSetIterator iter = new MnistDataSetIterator(batchSize,numSamples,true);

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .iterations(iterations)
                .momentum(0.5)
                .momentumAfter(Collections.singletonMap(3, 0.9))
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .list(4)
                .layer(0, new RBM.Builder().nIn(numRows*numColumns).nOut(500)
                        .weightInit(WeightInit.XAVIER).lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                        .visibleUnit(RBM.VisibleUnit.BINARY)
                        .hiddenUnit(RBM.HiddenUnit.BINARY)
                        .build())
                .layer(1, new RBM.Builder().nIn(500).nOut(250)
                        .weightInit(WeightInit.XAVIER).lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                        .visibleUnit(RBM.VisibleUnit.BINARY)
                        .hiddenUnit(RBM.HiddenUnit.BINARY)
                        .build())
                .layer(2, new RBM.Builder().nIn(250).nOut(200)
                        .weightInit(WeightInit.XAVIER).lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                        .visibleUnit(RBM.VisibleUnit.BINARY)
                        .hiddenUnit(RBM.HiddenUnit.BINARY)
                        .build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation("softmax")
                        .nIn(200).nOut(outputNum).build())
                .pretrain(true).backprop(false)
                .build();

        ;

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq),(IterationListener) new UpdateActivationIterationListener(listenerFreq)));

        log.info("Train model....");
        model.fit(iter); // achieves end to end pre-training

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum);

        DataSetIterator testIter = new MnistDataSetIterator(100,10000);
        while(testIter.hasNext()) {
            DataSet testMnist = testIter.next();
            INDArray predict2 = model.output(testMnist.getFeatureMatrix());
            eval.eval(testMnist.getLabels(), predict2);
        }

        log.info(eval.stats());
        log.info("****************Example finished********************");
    }

    /**
     * This test is for manual execution only, since it's here just to get working CNN and visualize it's layers
     *
     * @throws Exception
     */
    @Test
    public void testCNNWeightsVisualization() throws Exception {
        final int numRows = 40;
        final int numColumns = 40;
        int nChannels = 3;
        int outputNum = LFWLoader.NUM_LABELS;
        int numSamples = 1000; // LFWLoader.NUM_IMAGES;
        boolean useSubset = false;
        int batchSize = 200;// numSamples/10;
        int iterations = 6;
        int splitTrainNum = (int) (batchSize*.8);
        int seed = 123;
        int listenerFreq = iterations/5;
        DataSet lfwNext;
        SplitTestAndTrain trainTest;
        DataSet trainInput;
        List<INDArray> testInput = new ArrayList<>();
        List<INDArray> testLabels = new ArrayList<>();

        log.info("Load data....");
        DataSetIterator lfw = new LFWDataSetIterator(batchSize, numSamples, new int[] {numRows, numColumns, nChannels}, outputNum, useSubset, new Random(seed));

        log.info("Build model....");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation("relu")
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.01)
                .momentum(0.9)
                .regularization(true)
                .updater(Updater.ADAGRAD)
                .useDropConnect(true)
                .list(6)
                .layer(0, new ConvolutionLayer.Builder(4, 4)
                        .name("cnn1")
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .name("pool1")
                        .build())
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .name("cnn2")
                        .stride(1,1)
                        .nOut(40)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .name("pool2")
                        .build())
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .name("cnn3")
                        .stride(1,1)
                        .nOut(60)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .name("pool3")
                        .build())
                .layer(2, new ConvolutionLayer.Builder(2, 2)
                        .name("cnn3")
                        .stride(1,1)
                        .nOut(80)
                        .build())
                .layer(4, new DenseLayer.Builder()
                        .name("ffn1")
                        .nOut(160)
                        .dropOut(0.5)
                        .build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);
        new ConvolutionLayerSetup(builder,numRows,numColumns,nChannels);

        MultiLayerNetwork model = new MultiLayerNetwork(builder.build());
        model.init();

        log.info("Train model....");
        model.setListeners(Arrays.asList(new ScoreIterationListener(listenerFreq), new WeightsVisualizationListener(listenerFreq), new HistogramIterationListener(listenerFreq)));

        while(lfw.hasNext()) {
            lfwNext = lfw.next();
            lfwNext.scale();
            trainTest = lfwNext.splitTestAndTrain(splitTrainNum, new Random(seed)); // train set that is the result
            trainInput = trainTest.getTrain(); // get feature matrix and labels for training
            testInput.add(trainTest.getTest().getFeatureMatrix());
            testLabels.add(trainTest.getTest().getLabels());
            model.fit(trainInput);
        }

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(lfw.getLabels());
        for(int i = 0; i < testInput.size(); i++) {
            INDArray output = model.output(testInput.get(i));
            eval.eval(testLabels.get(i), output);
        }
        INDArray output = model.output(testInput.get(0));
        eval.eval(testLabels.get(0), output);
        log.info(eval.stats());
        log.info("****************Example finished********************");

    }
}

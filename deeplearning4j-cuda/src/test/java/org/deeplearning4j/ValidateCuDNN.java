package org.deeplearning4j;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.layers.convolution.ConvolutionLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.CuDNNValidationUtil;
import org.junit.Test;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationELU;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.IsMax;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.impl.LossNegativeLogLikelihood;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

@Slf4j
public class ValidateCuDNN extends BaseDL4JTest {

    @Test
    public void validateConvLayers() {
        Nd4j.getRandom().setSeed(12345);

        int numClasses = 10;
        //imageHeight,imageWidth,channels
        int imageHeight = 240;
        int imageWidth = 240;
        int channels = 3;
        IActivation activation = new ActivationIdentity();
        MultiLayerConfiguration multiLayerConfiguration = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER).seed(42)
                .activation(new ActivationELU())
                .updater(Nesterovs.builder()
                        .momentum(0.9)
                        .learningRateSchedule(new StepSchedule(
                                ScheduleType.EPOCH,
                                1e-2,
                                0.1,
                                20)).build()).list(
                        new Convolution2D.Builder().nOut(96)
                                .kernelSize(11, 11).biasInit(0.0)
                                .stride(4, 4).build(),
                        new ActivationLayer.Builder().activation(activation).build(),
                        new BatchNormalization.Builder().build(),
                        new LocalResponseNormalization.Builder()
                                .alpha(1e-3).beta(0.75).k(2)
                                .n(5).build(),
                        new Pooling2D.Builder()
                                .poolingType(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(3, 3).stride(2, 2)
                                .build(),
                        new Convolution2D.Builder().nOut(256)
                                .kernelSize(5, 5).padding(2, 2)
                                .biasInit(0.0)
                                .stride(1, 1).build(),
                        new ActivationLayer.Builder().activation(activation).build(),
                        new BatchNormalization.Builder().build(),
                        new LocalResponseNormalization.Builder()
                                .alpha(1e-3).beta(0.75).k(2)
                                .n(5).build(),
                        new Pooling2D.Builder()
                                .poolingType(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(3, 3).stride(2, 2)
                                .build(),
                        new Convolution2D.Builder().nOut(384)
                                .kernelSize(3, 3).padding(1, 1)
                                .biasInit(0.0)
                                .stride(1, 1).build(),
                        new ActivationLayer.Builder().activation(activation).build(),
                        new Convolution2D.Builder().nOut(256)
                                .kernelSize(3, 3).padding(1, 1)
                                .stride(1, 1).build(),
                        new ActivationLayer.Builder().activation(activation).build(),
                        new BatchNormalization.Builder().build(),
                        new Pooling2D.Builder()
                                .poolingType(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(3, 3).stride(2, 2)
                                .build(),
                        new DenseLayer.Builder()
                                .nOut(4096)
                                .biasInit(0.0)
                                .build(),
                        new ActivationLayer.Builder().activation(activation).build(),
                        new OutputLayer.Builder().activation(new ActivationSoftmax())
                                .lossFunction(new LossNegativeLogLikelihood())
                                .nOut(numClasses)
                                .biasInit(0.0)
                                .build())
                .setInputType(InputType.convolutionalFlat(imageHeight, imageWidth, channels))
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(multiLayerConfiguration);
        net.init();

        int[] fShape = new int[]{32, channels, imageHeight, imageWidth};
        int[] lShape = new int[]{32, numClasses};

        List<Class<?>> classesToTest = new ArrayList<>();
        classesToTest.add(ConvolutionLayer.class);
        classesToTest.add(org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingLayer.class);
        classesToTest.add(org.deeplearning4j.nn.layers.normalization.LocalResponseNormalization.class);
        classesToTest.add(org.deeplearning4j.nn.layers.normalization.BatchNormalization.class);

        validateLayers(net, classesToTest, fShape, lShape);
    }


    @Test
    public void validateConvLayersSimple() {
        Nd4j.getRandom().setSeed(12345);

        int numClasses = 10;
        //imageHeight,imageWidth,channels
        int imageHeight = 240;
        int imageWidth = 240;
        int channels = 3;
        IActivation activation = new ActivationIdentity();
        MultiLayerConfiguration multiLayerConfiguration = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER).seed(42)
                .activation(new ActivationELU())
                .updater(new Nesterovs(1e-2, 0.9))
                .list(
                        new Convolution2D.Builder().nOut(96)
                                .kernelSize(11, 11).biasInit(0.0)
                                .stride(4, 4).build(),
                        new ActivationLayer.Builder().activation(activation).build(),
                        new Pooling2D.Builder()
                                .poolingType(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(3, 3).stride(2, 2)
                                .build(),
                        new Convolution2D.Builder().nOut(256)
                                .kernelSize(5, 5).padding(2, 2)
                                .biasInit(0.0)
                                .stride(1, 1).build(),
                        new ActivationLayer.Builder().activation(activation).build(),
                        new Pooling2D.Builder()
                                .poolingType(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(3, 3).stride(2, 2)
                                .build(),
                        new Convolution2D.Builder().nOut(384)
                                .kernelSize(3, 3).padding(1, 1)
                                .biasInit(0.0)
                                .stride(1, 1).build(),
                        new ActivationLayer.Builder().activation(activation).build(),
                        new Convolution2D.Builder().nOut(256)
                                .kernelSize(3, 3).padding(1, 1)
                                .stride(1, 1).build(),
                        new ActivationLayer.Builder().activation(activation).build(),
                        new Pooling2D.Builder()
                                .poolingType(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(3, 3).stride(2, 2)
                                .build(),
                        new DenseLayer.Builder()
                                .nOut(4096)
                                .biasInit(0.0)
                                .build(),
                        new ActivationLayer.Builder().activation(activation).build(),
                        new OutputLayer.Builder().activation(new ActivationSoftmax())
                                .lossFunction(new LossNegativeLogLikelihood())
                                .nOut(numClasses)
                                .biasInit(0.0)
                                .build())
                .setInputType(InputType.convolutionalFlat(imageHeight, imageWidth, channels))
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(multiLayerConfiguration);
        net.init();

        int[] fShape = new int[]{32, channels, imageHeight, imageWidth};
        int[] lShape = new int[]{32, numClasses};

        List<Class<?>> classesToTest = new ArrayList<>();
        classesToTest.add(ConvolutionLayer.class);
        classesToTest.add(org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingLayer.class);

        validateLayers(net, classesToTest, fShape, lShape);
    }

    @Test
    public void validateConvLayersSimpleBN() {
        Nd4j.getRandom().setSeed(12345);

        int numClasses = 10;
        //imageHeight,imageWidth,channels
        int imageHeight = 240;
        int imageWidth = 240;
        int channels = 3;
        IActivation activation = new ActivationIdentity();
        MultiLayerConfiguration multiLayerConfiguration = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER).seed(42)
                .activation(new ActivationELU())
                .updater(Nesterovs.builder()
                        .momentum(0.9)
                        .learningRateSchedule(new StepSchedule(
                                ScheduleType.EPOCH,
                                1e-2,
                                0.1,
                                20)).build()).list(
                        new Convolution2D.Builder().nOut(96)
                                .kernelSize(11, 11).biasInit(0.0)
                                .stride(4, 4).build(),
                        new ActivationLayer.Builder().activation(activation).build(),
                        new BatchNormalization.Builder().build(),
                        new Pooling2D.Builder()
                                .poolingType(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(3, 3).stride(2, 2)
                                .build(),
                        new DenseLayer.Builder()
                                .nOut(128)
                                .biasInit(0.0)
                                .build(),
                        new ActivationLayer.Builder().activation(activation).build(),
                        new OutputLayer.Builder().activation(new ActivationSoftmax())
                                .lossFunction(new LossNegativeLogLikelihood())
                                .nOut(numClasses)
                                .biasInit(0.0)
                                .build())
                .setInputType(InputType.convolutionalFlat(imageHeight, imageWidth, channels))
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(multiLayerConfiguration);
        net.init();

        int[] fShape = new int[]{32, channels, imageHeight, imageWidth};
        int[] lShape = new int[]{32, numClasses};

        List<Class<?>> classesToTest = new ArrayList<>();
        classesToTest.add(ConvolutionLayer.class);
        classesToTest.add(org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingLayer.class);
        classesToTest.add(org.deeplearning4j.nn.layers.normalization.BatchNormalization.class);

        validateLayers(net, classesToTest, fShape, lShape);
    }

    @Test
    public void validateConvLayersLRN() {
        Nd4j.getRandom().setSeed(12345);

        int numClasses = 10;
        //imageHeight,imageWidth,channels
        int imageHeight = 240;
        int imageWidth = 240;
        int channels = 3;
        IActivation activation = new ActivationIdentity();
        MultiLayerConfiguration multiLayerConfiguration = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER).seed(42)
                .activation(new ActivationELU())
                .updater(Nesterovs.builder()
                        .momentum(0.9)
                        .learningRateSchedule(new StepSchedule(
                                ScheduleType.EPOCH,
                                1e-2,
                                0.1,
                                20)).build()).list(
                        new Convolution2D.Builder().nOut(96)
                                .kernelSize(11, 11).biasInit(0.0)
                                .stride(4, 4).build(),
                        new ActivationLayer.Builder().activation(activation).build(),
                        new LocalResponseNormalization.Builder()
                                .alpha(1e-3).beta(0.75).k(2)
                                .n(5).build(),
                        new Pooling2D.Builder()
                                .poolingType(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(3, 3).stride(2, 2)
                                .build(),
                        new Convolution2D.Builder().nOut(256)
                                .kernelSize(5, 5).padding(2, 2)
                                .biasInit(0.0)
                                .stride(1, 1).build(),
                        new ActivationLayer.Builder().activation(activation).build(),
                        new OutputLayer.Builder().activation(new ActivationSoftmax())
                                .lossFunction(new LossNegativeLogLikelihood())
                                .nOut(numClasses)
                                .biasInit(0.0)
                                .build())
                .setInputType(InputType.convolutionalFlat(imageHeight, imageWidth, channels))
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(multiLayerConfiguration);
        net.init();

        int[] fShape = new int[]{32, channels, imageHeight, imageWidth};
        int[] lShape = new int[]{32, numClasses};

        List<Class<?>> classesToTest = new ArrayList<>();
        classesToTest.add(ConvolutionLayer.class);
        classesToTest.add(org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingLayer.class);
        classesToTest.add(org.deeplearning4j.nn.layers.normalization.LocalResponseNormalization.class);
        classesToTest.add(org.deeplearning4j.nn.layers.normalization.BatchNormalization.class);

        validateLayers(net, classesToTest, fShape, lShape);
    }


    public static void validateLayers(MultiLayerNetwork net, List<Class<?>> classesToTest, int[] fShape, int[] lShape) {

        for (WorkspaceMode wsm : new WorkspaceMode[]{WorkspaceMode.NONE, WorkspaceMode.ENABLED}) {

            net.getLayerWiseConfigurations().setTrainingWorkspaceMode(wsm);
            net.getLayerWiseConfigurations().setInferenceWorkspaceMode(wsm);

            Nd4j.getRandom().setSeed(12345);
            INDArray features = Nd4j.rand(fShape);
            INDArray labels = Nd4j.rand(lShape);
            Nd4j.getExecutioner().exec(new IsMax(labels, 1));

            List<CuDNNValidationUtil.TestCase> testCaseList = new ArrayList<>();

            for (Class<?> c : classesToTest) {
                String name = "WS=" + wsm + ", testCudnnFor=" + c.getSimpleName();
                testCaseList.add(CuDNNValidationUtil.TestCase.builder()
                        .testName(name)
                        .allowCudnnHelpersForClasses(Collections.<Class<?>>singletonList(c))
                        .testForward(true)
                        .testScore(true)
                        .testBackward(true)
                        .trainFirst(false)
                        .features(features)
                        .labels(labels)
                        .build());
            }
            testCaseList.add(CuDNNValidationUtil.TestCase.builder()
                    .testName("WS=" + wsm + ", ALL CLASSES")
                    .allowCudnnHelpersForClasses(classesToTest)
                    .testForward(true)
                    .testScore(true)
                    .testBackward(true)
                    .trainFirst(false)
                    .features(features)
                    .labels(labels)
                    .build());


            List<DataSet> dataSets = new ArrayList<>();
            for (int i = 0; i < 6; i++) {
                INDArray f = Nd4j.rand(fShape);
                INDArray l = Nd4j.rand(lShape);
                Nd4j.getExecutioner().exec(new IsMax(labels, 1));
                dataSets.add(new DataSet(f, l));
            }
            DataSetIterator iter = new ExistingDataSetIterator(dataSets);


            testCaseList.add(CuDNNValidationUtil.TestCase.builder()
                    .testName("WS=" + wsm + ", All cases, train first")
                    .allowCudnnHelpersForClasses(classesToTest)
                    .testForward(true)
                    .testScore(true)
                    .testBackward(true)
                    .trainFirst(true)
                    .features(features)
                    .labels(labels)
                    .data(iter)
                    .build());

            for (CuDNNValidationUtil.TestCase tc : testCaseList) {
                log.info("Running test: " + tc.getTestName());
                CuDNNValidationUtil.validateMLN(net, tc);
            }
        }
    }

}

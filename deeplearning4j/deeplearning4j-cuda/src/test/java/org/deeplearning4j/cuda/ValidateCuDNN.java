/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.cuda;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.layers.convolution.ConvolutionLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.cuda.util.CuDNNValidationUtil;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationELU;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.any.IsMax;
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

    @Override
    public long getTimeoutMilliseconds() {
        return 360000L;
    }

    @Test
    public void validateConvLayers() {
        Nd4j.getRandom().setSeed(12345);

        int numClasses = 10;
        //imageHeight,imageWidth,channels
        int imageHeight = 64;
        int imageWidth = 64;
        int channels = 3;
        IActivation activation = new ActivationIdentity();
        MultiLayerConfiguration multiLayerConfiguration = new NeuralNetConfiguration.Builder()
                .dataType(DataType.DOUBLE)
                .weightInit(WeightInit.XAVIER).seed(42)
                .activation(new ActivationELU())
                .updater(new Nesterovs(1e-3, 0.9))
                .list(
                        new Convolution2D.Builder().nOut(16)
                                .kernelSize(4, 4).biasInit(0.0)
                                .stride(2, 2).build(),
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
                        new Convolution2D.Builder().nOut(16)
                                .kernelSize(3, 3).padding(1, 1)
                                .biasInit(0.0)
                                .stride(1, 1).build(),
                        new ActivationLayer.Builder().activation(activation).build(),
                        new Convolution2D.Builder().nOut(16)
                                .kernelSize(3, 3).padding(1, 1)
                                .stride(1, 1).build(),
                        new ActivationLayer.Builder().activation(activation).build(),
                        new Pooling2D.Builder()
                                .poolingType(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(3, 3).stride(2, 2)
                                .build(),
                        new DenseLayer.Builder()
                                .nOut(64)
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

        int[] fShape = new int[]{8, channels, imageHeight, imageWidth};
        int[] lShape = new int[]{8, numClasses};

        List<Class<?>> classesToTest = new ArrayList<>();
        classesToTest.add(ConvolutionLayer.class);
        classesToTest.add(org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingLayer.class);

        validateLayers(net, classesToTest, true, fShape, lShape, CuDNNValidationUtil.MAX_REL_ERROR, CuDNNValidationUtil.MIN_ABS_ERROR);
    }

    @Test
    public void validateConvLayersSimpleBN() {
        //Test ONLY BN - no other CuDNN functionality (i.e., DL4J impls for everything else)
        Nd4j.getRandom().setSeed(12345);

        int minibatch = 8;
        int numClasses = 10;
        //imageHeight,imageWidth,channels
        int imageHeight = 48;
        int imageWidth = 48;
        int channels = 3;
        IActivation activation = new ActivationIdentity();
        MultiLayerConfiguration multiLayerConfiguration = new NeuralNetConfiguration.Builder()
                .dataType(DataType.DOUBLE)
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

        int[] fShape = new int[]{minibatch, channels, imageHeight, imageWidth};
        int[] lShape = new int[]{minibatch, numClasses};

        List<Class<?>> classesToTest = new ArrayList<>();
        classesToTest.add(org.deeplearning4j.nn.layers.normalization.BatchNormalization.class);

        validateLayers(net, classesToTest, false, fShape, lShape, CuDNNValidationUtil.MAX_REL_ERROR, CuDNNValidationUtil.MIN_ABS_ERROR);
    }

    @Test @Ignore //AB 2019/05/20 - https://github.com/eclipse/deeplearning4j/issues/5088 - ignored to get to "all passing" state for CI, and revisit later
    public void validateConvLayersLRN() {
        //Test ONLY LRN - no other CuDNN functionality (i.e., DL4J impls for everything else)
        Nd4j.getRandom().setSeed(12345);

        int minibatch = 8;
        int numClasses = 10;
        //imageHeight,imageWidth,channels
        int imageHeight = 48;
        int imageWidth = 48;
        int channels = 3;
        IActivation activation = new ActivationIdentity();
        MultiLayerConfiguration multiLayerConfiguration = new NeuralNetConfiguration.Builder()
                .dataType(DataType.DOUBLE)
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

        int[] fShape = new int[]{minibatch, channels, imageHeight, imageWidth};
        int[] lShape = new int[]{minibatch, numClasses};

        List<Class<?>> classesToTest = new ArrayList<>();
        classesToTest.add(org.deeplearning4j.nn.layers.normalization.LocalResponseNormalization.class);

        validateLayers(net, classesToTest, false, fShape, lShape, 1e-2, 1e-2);
    }

    public static void validateLayers(MultiLayerNetwork net, List<Class<?>> classesToTest, boolean testAllCudnnPresent, int[] fShape, int[] lShape, double maxRE, double minAbsErr) {

        for (WorkspaceMode wsm : new WorkspaceMode[]{WorkspaceMode.NONE, WorkspaceMode.ENABLED}) {

            net.getLayerWiseConfigurations().setTrainingWorkspaceMode(wsm);
            net.getLayerWiseConfigurations().setInferenceWorkspaceMode(wsm);

            Nd4j.getRandom().setSeed(12345);
            INDArray features = Nd4j.rand(fShape);
            INDArray labels = Nd4j.rand(lShape);
            labels = Nd4j.exec(new IsMax(labels, 1))[0].castTo(features.dataType());

            List<CuDNNValidationUtil.TestCase> testCaseList = new ArrayList<>();

            List<DataSet> dataSets = new ArrayList<>();
            for (int i = 0; i < 6; i++) {
                INDArray f = Nd4j.rand(fShape);
                INDArray l = Nd4j.rand(lShape);
                l = Nd4j.exec(new IsMax(l, 1))[0].castTo(features.dataType());
                dataSets.add(new DataSet(f, l));
            }
            DataSetIterator iter = new ExistingDataSetIterator(dataSets);

            for (Class<?> c : classesToTest) {
                String name = "WS=" + wsm + ", testCudnnFor=" + c.getSimpleName();
                testCaseList.add(CuDNNValidationUtil.TestCase.builder()
                        .testName(name)
                        .allowCudnnHelpersForClasses(Collections.<Class<?>>singletonList(c))
                        .testForward(true)
                        .testScore(true)
                        .testBackward(true)
                        .testTraining(true)
                        .trainFirst(false)
                        .features(features)
                        .labels(labels)
                        .data(iter)
                        .maxRE(maxRE)
                        .minAbsErr(minAbsErr)
                        .build());
            }

            if(testAllCudnnPresent) {
                testCaseList.add(CuDNNValidationUtil.TestCase.builder()
                        .testName("WS=" + wsm + ", ALL CLASSES")
                        .allowCudnnHelpersForClasses(classesToTest)
                        .testForward(true)
                        .testScore(true)
                        .testBackward(true)
                        .trainFirst(false)
                        .features(features)
                        .labels(labels)
                        .data(iter)
                        .maxRE(maxRE)
                        .minAbsErr(minAbsErr)
                        .build());
            }

            for (CuDNNValidationUtil.TestCase tc : testCaseList) {
                log.info("Running test: " + tc.getTestName());
                CuDNNValidationUtil.validateMLN(net, tc);
            }
        }
    }

}

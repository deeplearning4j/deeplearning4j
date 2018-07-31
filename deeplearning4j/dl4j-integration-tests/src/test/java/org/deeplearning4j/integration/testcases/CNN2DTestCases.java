/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.integration.testcases;

import org.deeplearning4j.integration.TestCase;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.iterator.EarlyTerminationDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MultiDataSetIteratorAdapter;
import org.deeplearning4j.datasets.iterator.impl.TinyImageNetDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.EvaluationCalibration;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.eval.ROCMultiClass;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.util.ComputationGraphUtil;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CNN2DTestCases {

    /**
     * Essentially: LeNet MNIST example
     */
    public static TestCase getLenetMnist() {
        return new TestCase() {
            {
                testName = "LenetMnist";
                testType = TestType.RANDOM_INIT;
                testPredictions = true;
                testTrainingCurves = true;
                testGradients = true;
                testParamsPostTraining = true;
                testEvaluation = true;
                testOverfitting = false;
            }

            public Object getConfiguration() throws Exception {
                int nChannels = 1; // Number of input channels
                int outputNum = 10; // The number of possible outcomes
                int seed = 123;

                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .seed(seed)
                        .l2(0.0005)
                        .weightInit(WeightInit.XAVIER)
                        .updater(new Nesterovs(0.01, 0.9))
                        .list()
                        .layer(0, new ConvolutionLayer.Builder(5, 5)
                                //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                                .nIn(nChannels)
                                .stride(1, 1)
                                .nOut(20)
                                .activation(Activation.IDENTITY)
                                .build())
                        .layer(1, new SubsamplingLayer.Builder(PoolingType.MAX)
                                .kernelSize(2, 2)
                                .stride(2, 2)
                                .build())
                        .layer(2, new ConvolutionLayer.Builder(5, 5)
                                //Note that nIn need not be specified in later layers
                                .stride(1, 1)
                                .nOut(50)
                                .activation(Activation.IDENTITY)
                                .build())
                        .layer(3, new SubsamplingLayer.Builder(PoolingType.MAX)
                                .kernelSize(2, 2)
                                .stride(2, 2)
                                .build())
                        .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                                .nOut(500).build())
                        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nOut(outputNum)
                                .activation(Activation.SOFTMAX)
                                .build())
                        .setInputType(InputType.convolutionalFlat(28, 28, 1)) //See note below
                        .backprop(true).pretrain(false).build();

                return conf;
            }

            @Override
            public MultiDataSet getGradientsTestData() throws Exception {
                DataSet ds = new MnistDataSetIterator(8, false, 12345).next();
                return new org.nd4j.linalg.dataset.MultiDataSet(ds.getFeatures(), ds.getLabels());
            }

            @Override
            public MultiDataSetIterator getTrainingData() throws Exception {
                DataSetIterator iter = new MnistDataSetIterator(16, true, 12345);

                iter = new EarlyTerminationDataSetIterator(iter, 60);
                return new MultiDataSetIteratorAdapter(iter);
            }

            @Override
            public MultiDataSetIterator getEvaluationTestData() throws Exception {
                return new MultiDataSetIteratorAdapter(new EarlyTerminationDataSetIterator(new MnistDataSetIterator(32, false, 12345), 10));
            }

            @Override
            public List<Pair<INDArray[],INDArray[]>> getPredictionsTestData() throws Exception {
                DataSetIterator iter = new MnistDataSetIterator(8, true, 12345);
                List<Pair<INDArray[], INDArray[]>> list = new ArrayList<>();

                DataSet ds = iter.next();
                ds = ds.asList().get(0);
                list.add(new Pair<>(new INDArray[]{ds.getFeatures()}, null));

                ds = iter.next();
                list.add(new Pair<>(new INDArray[]{ds.getFeatures()}, null));
                return list;
            }

            @Override
            public IEvaluation[] getNewEvaluations(){
                return new IEvaluation[]{
                        new Evaluation(),
                        new ROCMultiClass()};
            }

        };
    }


    /**
     * VGG16 + transfer learning + tiny imagenet
     */
    public static TestCase getVGG16TransferTinyImagenet() {
        return new TestCase() {

            {
                testName = "VGG16TransferTinyImagenet_224";
                testType = TestType.PRETRAINED;
                testPredictions = true;
                testTrainingCurves = true;
                testGradients = false;              //Skip - requires saving approx 1GB of data (gradients x2)
                testParamsPostTraining = false;     //Skip - requires saving all params (approx 500mb)
                testEvaluation = true;
                testOverfitting = false;
            }

            @Override
            public Model getPretrainedModel() throws Exception {
                VGG16 vgg16 = VGG16.builder()
                        .seed(12345)
                        .build();

                ComputationGraph pretrained = (ComputationGraph) vgg16.initPretrained(PretrainedType.IMAGENET);

                //Transfer learning
                ComputationGraph newGraph = new TransferLearning.GraphBuilder(pretrained)
                        .fineTuneConfiguration(new FineTuneConfiguration.Builder()
                                .updater(new Adam(1e-3))
                                .seed(12345)
                                .build())
                        .removeVertexKeepConnections("predictions")
                        .addLayer("predictions", new OutputLayer.Builder()
                                .nIn(4096)
                                .nOut(200)  //Tiny imagenet
                                .build(), "fc2")
                        .build();

                return newGraph;
            }

            @Override
            public List<Pair<INDArray[], INDArray[]>> getPredictionsTestData() throws Exception {
                List<Pair<INDArray[], INDArray[]>> out = new ArrayList<>();

                DataSetIterator iter = new TinyImageNetDataSetIterator(1, new int[]{224, 224}, DataSetType.TRAIN, null, 12345);
                iter.setPreProcessor(new VGG16ImagePreProcessor());
                DataSet ds = iter.next();
                out.add(new Pair<>(new INDArray[]{ds.getFeatures()}, null));

                iter = new TinyImageNetDataSetIterator(3, new int[]{224, 224}, DataSetType.TRAIN, null, 54321);
                iter.setPreProcessor(new VGG16ImagePreProcessor());
                ds = iter.next();
                out.add(new Pair<>(new INDArray[]{ds.getFeatures()}, null));

                return out;
            }

            @Override
            public MultiDataSet getGradientsTestData() throws Exception {
                DataSet ds = new TinyImageNetDataSetIterator(8, new int[]{224, 224}, DataSetType.TRAIN, null, 12345).next();
                return new org.nd4j.linalg.dataset.MultiDataSet(ds.getFeatures(), ds.getLabels());
            }

            @Override
            public MultiDataSetIterator getTrainingData() throws Exception {
                DataSetIterator iter = new TinyImageNetDataSetIterator(4, new int[]{224, 224}, DataSetType.TRAIN, null, 12345);
                iter.setPreProcessor(new VGG16ImagePreProcessor());

                iter = new EarlyTerminationDataSetIterator(iter, 6);
                return new MultiDataSetIteratorAdapter(iter);
            }

            @Override
            public IEvaluation[] getNewEvaluations() {
                return new IEvaluation[]{
                        new Evaluation(),
                        new ROCMultiClass(),
                        new EvaluationCalibration()
                };
            }

            @Override
            public MultiDataSetIterator getEvaluationTestData() throws Exception {
                DataSetIterator iter = new TinyImageNetDataSetIterator(4, new int[]{224, 224}, DataSetType.TEST, null, 12345);
                iter.setPreProcessor(new VGG16ImagePreProcessor());
                iter = new EarlyTerminationDataSetIterator(iter, 6);
                return new MultiDataSetIteratorAdapter(iter);
            }
        };
    }


    /**
     * Basically a cut-down version of the YOLO house numbers example
     */
    public static TestCase getYoloHouseNumbers() {

        throw new UnsupportedOperationException("Not yet implemented");
    }


    /**
     * A synthetic 2D CNN that uses all layers:
     * Convolution, Subsampling, Upsampling, Cropping, Depthwise conv, separable conv, deconv, space to batch,
     * space to depth, zero padding, batch norm, LRN
     */
    public static TestCase getCnn2DSynthetic() {

        throw new UnsupportedOperationException("Not yet implemented");
    }


    public static TestCase testLenetTransferDropoutRepeatability() {
        return new TestCase() {

            {
                testName = "LenetDropoutRepeatability";
                testType = TestType.PRETRAINED;
                testPredictions = true;
                testTrainingCurves = true;
                testGradients = true;
                testParamsPostTraining = true;
                testEvaluation = true;
                testOverfitting = true;
            }

            @Override
            public Model getPretrainedModel() throws Exception {

                Map<Integer, Double> lrSchedule = new HashMap<>();
                lrSchedule.put(0, 0.01);
                lrSchedule.put(1000, 0.005);
                lrSchedule.put(3000, 0.001);

                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .seed(12345)
                        .l2(0.0005)
                        .weightInit(WeightInit.XAVIER)
                        .updater(new Nesterovs(0.01, 0.9))
                        .list()
                        .layer(0, new ConvolutionLayer.Builder(5, 5)
                                //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                                .nIn(1)
                                .stride(1, 1)
                                .nOut(20)
                                .activation(Activation.IDENTITY)
                                .build())
                        .layer(1, new SubsamplingLayer.Builder(PoolingType.MAX)
                                .kernelSize(2, 2)
                                .stride(2, 2)
                                .build())
                        .layer(2, new ConvolutionLayer.Builder(5, 5)
                                //Note that nIn need not be specified in later layers
                                .stride(1, 1)
                                .nOut(50)
                                .activation(Activation.IDENTITY)
                                .dropOut(0.5)   //**** Dropout on conv layer
                                .build())
                        .layer(3, new SubsamplingLayer.Builder(PoolingType.MAX)
                                .kernelSize(2, 2)
                                .stride(2, 2)
                                .build())
                        .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                                .dropOut(0.5)   //**** Dropout on dense layer
                                .nOut(500).build())
                        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nOut(10)
                                .activation(Activation.SOFTMAX)
                                .build())
                        .setInputType(InputType.convolutionalFlat(28, 28, 1)) //See note below
                        .backprop(true).pretrain(false).build();


                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();

                DataSetIterator iter = new EarlyTerminationDataSetIterator(new MnistDataSetIterator(16, true, 12345), 10);
                net.fit(iter);

                MultiLayerNetwork pretrained = new TransferLearning.Builder(net)
                        .fineTuneConfiguration(
                                FineTuneConfiguration.builder()
                                        .updater(new Nesterovs(0.01, 0.9))
                                        .seed(98765)
                                        .build())
                        .nOutReplace(5, 10, WeightInit.XAVIER)
                        .build();

                return pretrained;
            }

            @Override
            public List<Pair<INDArray[], INDArray[]>> getPredictionsTestData() throws Exception {
                MnistDataSetIterator iter = new MnistDataSetIterator(1, true, 12345);
                List<Pair<INDArray[], INDArray[]>> out = new ArrayList<>();
                out.add(new Pair<>(new INDArray[]{iter.next().getFeatures()}, null));

                iter = new MnistDataSetIterator(10, true, 12345);
                out.add(new Pair<>(new INDArray[]{iter.next().getFeatures()}, null));
                return out;
            }

            @Override
            public MultiDataSet getGradientsTestData() throws Exception {
                DataSet ds = new MnistDataSetIterator(10, true, 12345).next();
                return new org.nd4j.linalg.dataset.MultiDataSet(ds.getFeatures(), ds.getLabels());
            }

            @Override
            public MultiDataSetIterator getTrainingData() throws Exception {
                DataSetIterator iter = new MnistDataSetIterator(16, true, 12345);
                iter = new EarlyTerminationDataSetIterator(iter, 32);
                return new MultiDataSetIteratorAdapter(iter);
            }

            @Override
            public IEvaluation[] getNewEvaluations() {
                return new IEvaluation[]{
                        new Evaluation(),
                        new ROCMultiClass(),
                        new EvaluationCalibration()
                };
            }

            @Override
            public MultiDataSetIterator getEvaluationTestData() throws Exception {
                DataSetIterator iter = new MnistDataSetIterator(16, true, 12345);
                iter = new EarlyTerminationDataSetIterator(iter, 10);
                return new MultiDataSetIteratorAdapter(iter);
            }

            @Override
            public MultiDataSet getOverfittingData() throws Exception {
                DataSet ds = new MnistDataSetIterator(1, true, 12345).next();
                return ComputationGraphUtil.toMultiDataSet(ds);
            }

            @Override
            public int getOverfitNumIterations() {
                return 200;
            }
        };
    }
}

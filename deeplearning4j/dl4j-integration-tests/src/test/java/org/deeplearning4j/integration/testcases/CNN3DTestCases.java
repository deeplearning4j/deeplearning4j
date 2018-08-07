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

import org.apache.commons.math3.stat.inference.TestUtils;
import org.deeplearning4j.datasets.iterator.EarlyTerminationDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MultiDataSetIteratorAdapter;
import org.deeplearning4j.datasets.iterator.impl.SingletonMultiDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.eval.ROCMultiClass;
import org.deeplearning4j.integration.TestCase;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class CNN3DTestCases {


    /**
     * A simple synthetic CNN 3d test case using all CNN 3d layers:
     * Subsampling, Upsampling, Convolution, Cropping, Zero padding
     */
    public static TestCase getCnn3dTestCaseSynthetic(){
        return new TestCase() {
            {
                testName = "Cnn3dSynthetic";
                testType = TestType.RANDOM_INIT;
                testPredictions = true;
                testTrainingCurves = true;
                testGradients = true;
                testParamsPostTraining = true;
                testEvaluation = true;
                testOverfitting = false;
            }

            public Object getConfiguration() throws Exception {
                int nChannels = 3; // Number of input channels
                int outputNum = 10; // The number of possible outcomes
                int seed = 123;

                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .seed(seed)
                        .l2(0.0005)
                        .weightInit(WeightInit.XAVIER)
                        .updater(new Nesterovs(0.01, 0.9))
                        .convolutionMode(ConvolutionMode.Same)
                        .list()
                        .layer(new Convolution3D.Builder(3,3,3)
                                .dataFormat(Convolution3D.DataFormat.NCDHW)
                                .nIn(nChannels)
                                .stride(2, 2, 2)
                                .nOut(8)
                                .activation(Activation.IDENTITY)
                                .build())
                        .layer(new Subsampling3DLayer.Builder(PoolingType.MAX)
                                .kernelSize(2, 2, 2)
                                .stride(2, 2, 2)
                                .build())
                        .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nOut(outputNum)
                                .activation(Activation.SOFTMAX)
                                .build())
                        .setInputType(InputType.convolutional3D(8,8,8,nChannels))
                        .backprop(true).pretrain(false).build();

                return conf;
            }

            @Override
            public MultiDataSet getGradientsTestData() throws Exception {
                Nd4j.getRandom().setSeed(12345);
                //NCDHW format
                INDArray arr = Nd4j.rand(new int[]{2, 3, 8, 8, 8});
                INDArray labels = org.deeplearning4j.integration.TestUtils.randomOneHot(2, 10);
                return new org.nd4j.linalg.dataset.MultiDataSet(arr, labels);
            }

            @Override
            public MultiDataSetIterator getTrainingData() throws Exception {
                return new SingletonMultiDataSetIterator(getGradientsTestData());
            }

            @Override
            public MultiDataSetIterator getEvaluationTestData() throws Exception {
                return getTrainingData();
            }

            @Override
            public List<Pair<INDArray[],INDArray[]>> getPredictionsTestData() throws Exception {
                MultiDataSet mds = getGradientsTestData();
                return Collections.singletonList(new Pair<>(mds.getFeatures(), null));
            }

            @Override
            public IEvaluation[] getNewEvaluations(){
                return new IEvaluation[]{new Evaluation()};
            }

        };
    };

}

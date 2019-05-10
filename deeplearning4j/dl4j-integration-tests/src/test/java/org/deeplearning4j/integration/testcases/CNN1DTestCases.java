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

import org.deeplearning4j.datasets.iterator.EarlyTerminationDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MultiDataSetIteratorAdapter;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.eval.ROCMultiClass;
import org.deeplearning4j.integration.TestCase;
import org.deeplearning4j.integration.testcases.misc.CharacterIterator;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping1D;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;

import java.util.Collections;
import java.util.List;

public class CNN1DTestCases {


    /**
     * A simple CNN 1d test case using most CNN 1d layers:
     * Subsampling, Upsampling, Convolution, Cropping, Zero padding
     */
    public static TestCase getCnn1dTestCaseCharRNN(){
        return new TestCase() {
            {
                testName = "CNN1dCharacterTestCase";
                testType = TestType.RANDOM_INIT;
                testPredictions = true;
                testTrainingCurves = true;
                testGradients = true;
                testParamsPostTraining = true;
                testEvaluation = true;
                testOverfitting = false;
            }

            int miniBatchSize = 16;
            int exampleLength = 128;

            @Override
            public Object getConfiguration() throws Exception {
                CharacterIterator iter = CharacterIterator.getShakespeareIterator(miniBatchSize,exampleLength);
                int nOut = iter.totalOutcomes();

                return new NeuralNetConfiguration.Builder()
                        .seed(12345)
                        .weightInit(WeightInit.XAVIER)
                        .updater(new Adam(0.01))
                        .convolutionMode(ConvolutionMode.Same)
                        .graphBuilder()
                        .addInputs("in")
                        .layer("0", new Convolution1DLayer.Builder().nOut(32).activation(Activation.TANH).kernelSize(3).stride(1).build(), "in")
                        .layer("1", new Subsampling1DLayer.Builder().kernelSize(2).stride(1).poolingType(SubsamplingLayer.PoolingType.MAX).build(), "0")
                        .layer("2", new Cropping1D(1), "1")
                        .layer("3", new ZeroPadding1DLayer(1), "2")
                        .layer("out", new RnnOutputLayer.Builder().activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).nOut(nOut).build(), "3")
                        .setInputTypes(InputType.recurrent(nOut))
                        .setOutputs("out")
                        .build();
            }

            @Override
            public List<Pair<INDArray[], INDArray[]>> getPredictionsTestData() throws Exception {
                MultiDataSet mds = getTrainingData().next();
                return Collections.singletonList(new Pair<>(mds.getFeatures(), mds.getFeaturesMaskArrays()));
            }

            @Override
            public MultiDataSet getGradientsTestData() throws Exception {
                return getTrainingData().next();
            }

            @Override
            public MultiDataSetIterator getTrainingData() throws Exception {
                DataSetIterator iter = CharacterIterator.getShakespeareIterator(miniBatchSize,exampleLength);
                iter = new EarlyTerminationDataSetIterator(iter, 2);    //3 minibatches, 1000/200 = 5 updates per minibatch
                return new MultiDataSetIteratorAdapter(iter);
            }

            @Override
            public IEvaluation[] getNewEvaluations(){
                return new IEvaluation[]{
                        new Evaluation(),
                        new ROCMultiClass()};
            }

            @Override
            public MultiDataSetIterator getEvaluationTestData() throws Exception {
                return getTrainingData();
            }
        };
    }

}

/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.integration.testcases.dl4j;


import org.deeplearning4j.datasets.iterator.EarlyTerminationDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.integration.ModelType;
import org.deeplearning4j.integration.TestCase;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.adapter.MultiDataSetIteratorAdapter;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.common.primitives.Pair;

import java.util.ArrayList;
import java.util.List;

public class UnsupervisedTestCases {

    /**
     * Basically: the MNIST VAE anomaly example
     */
    public static TestCase getVAEMnistAnomaly() {
        return new TestCase() {
            {
                testName = "VAEMnistAnomaly";
                testType = TestType.RANDOM_INIT;
                testPredictions = true;
                testUnsupervisedTraining = true;
                testTrainingCurves = false;
                testParamsPostTraining = false;
                testGradients = false;
                testEvaluation = false;
                testOverfitting = false;
                unsupervisedTrainLayersMLN = new int[]{0};
                maxRelativeErrorParamsPostTraining = 1e-4;
                minAbsErrorParamsPostTraining = 5e-4;
                maxRelativeErrorPretrainParams = 1e-4;
                minAbsErrorPretrainParams = 5e-4;
            }

            @Override
            public ModelType modelType() {
                return ModelType.MLN;
            }

            @Override
            public Object getConfiguration() {
                return new NeuralNetConfiguration.Builder()
                        .dataType(DataType.FLOAT)
                        .seed(12345)
                        .updater(new Adam(1e-3))
                        .weightInit(WeightInit.XAVIER)
                        .l2(1e-4)
                        .list()
                        .layer(0, new VariationalAutoencoder.Builder()
                                .activation(Activation.TANH)
                                .encoderLayerSizes(256, 256)                    //2 encoder layers, each of size 256
                                .decoderLayerSizes(256, 256)                    //2 decoder layers, each of size 256
                                .pzxActivationFunction(Activation.IDENTITY)     //p(z|data) activation function
                                //Bernoulli reconstruction distribution + sigmoid activation - for modelling binary data (or data in range 0 to 1)
                                .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID))
                                .nIn(28 * 28)                                   //Input size: 28x28
                                .nOut(32)                                       //Size of the latent variable space: p(z|x) - 32 values
                                .build())
                        .build();
            }

            @Override
            public List<Pair<INDArray[], INDArray[]>> getPredictionsTestData() throws Exception {
                MnistDataSetIterator iter = new MnistDataSetIterator(1, true, 12345);
                List<Pair<INDArray[],INDArray[]>> out = new ArrayList<>();
                out.add(new Pair<>(new INDArray[]{iter.next().getFeatures()}, null));

                iter = new MnistDataSetIterator(10, true, 12345);
                out.add(new Pair<>(new INDArray[]{iter.next().getFeatures()}, null));
                return out;
            }

            @Override
            public MultiDataSetIterator getUnsupervisedTrainData() throws Exception {
                DataSetIterator iter = new MnistDataSetIterator(16, true, 12345);
                iter = new EarlyTerminationDataSetIterator(iter, 32);
                return new MultiDataSetIteratorAdapter(iter);
            }
        };
    }

}

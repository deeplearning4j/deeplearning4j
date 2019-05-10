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

package org.deeplearning4j.parallelism.factory;

import lombok.val;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.deeplearning4j.parallelism.trainer.SymmetricTrainer;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.Assert.*;

public class DefaultTrainerContextTest {
    int nChannels = 1;
    int outputNum = 10;

    // for GPU you usually want to have higher batchSize
    int seed = 123;

    @Test
    public void testEqualUuid1() {
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().seed(seed)
                .l2(0.0005)
                //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.01, 0.9)).list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(nChannels).stride(1, 1).nOut(20).activation(Activation.IDENTITY).build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                        .stride(2, 2).build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        //Note that nIn needed be specified in later layers
                        .stride(1, 1).nOut(50).activation(Activation.IDENTITY).build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                        .stride(2, 2).build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU).nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum).activation(Activation.SOFTMAX).build())
                .setInputType(InputType.convolutionalFlat(28, 28, nChannels));

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        // ParallelWrapper will take care of load balancing between GPUs.
        ParallelWrapper wrapper = new ParallelWrapper.Builder(model)
                // DataSets prefetching options. Set this value with respect to number of actual devices
                .prefetchBuffer(24)

                // set number of workers equal or higher then number of available devices. x1-x2 are good values to start with
                .workers(2)

                // rare averaging improves performance, but might reduce model accuracy
                .averagingFrequency(3)

                // if set to TRUE, on every averaging model score will be reported
                .reportScoreAfterAveraging(true)

                // optinal parameter, set to false ONLY if your system has support P2P memory access across PCIe (hint: AWS do not support P2P)
                .build();

        val context = new DefaultTrainerContext();
        val trainer = context.create("alpha", 3, model, 0, true, wrapper, WorkspaceMode.NONE, 3);

        assertEquals("alpha_thread_3", trainer.getUuid());
    }

}
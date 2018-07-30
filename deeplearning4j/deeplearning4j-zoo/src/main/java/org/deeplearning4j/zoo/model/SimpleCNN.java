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

package org.deeplearning4j.zoo.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.ZooType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.IUpdater;

/**
 * A simple convolutional network for generic image classification.
 * Reference: https://github.com/oarriaga/face_classification/
 *
 * @author Justin Long (crockpotveggies)
 */
@AllArgsConstructor
@Builder
public class SimpleCNN extends ZooModel {

    @Builder.Default private long seed = 1234;
    @Builder.Default private int[] inputShape = new int[] {3, 48, 48};
    @Builder.Default private int numClasses = 0;
    @Builder.Default private IUpdater updater = new AdaDelta();
    @Builder.Default private CacheMode cacheMode = CacheMode.NONE;
    @Builder.Default private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
    @Builder.Default private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

    private SimpleCNN() {}

    @Override
    public String pretrainedUrl(PretrainedType pretrainedType) {
        return null;
    }

    @Override
    public long pretrainedChecksum(PretrainedType pretrainedType) {
        return 0L;
    }

    @Override
    public Class<? extends Model> modelType() {
        return MultiLayerNetwork.class;
    }

    public MultiLayerConfiguration conf() {
        MultiLayerConfiguration conf =
                        new NeuralNetConfiguration.Builder().seed(seed)
                                        .activation(Activation.IDENTITY)
                                        .weightInit(WeightInit.RELU)
                                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                        .updater(updater)
                                        .cacheMode(cacheMode)
                                        .trainingWorkspaceMode(workspaceMode)
                                        .inferenceWorkspaceMode(workspaceMode)
                                        .convolutionMode(ConvolutionMode.Same)
                                        .list()
                                        // block 1
                                        .layer(0, new ConvolutionLayer.Builder(new int[] {7, 7}).name("image_array")
                                                        .nIn(inputShape[0]).nOut(16).build())
                                        .layer(1, new BatchNormalization.Builder().build())
                                        .layer(2, new ConvolutionLayer.Builder(new int[] {7, 7}).nIn(16).nOut(16)
                                                        .build())
                                        .layer(3, new BatchNormalization.Builder().build())
                                        .layer(4, new ActivationLayer.Builder().activation(Activation.RELU).build())
                                        .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG,
                                                        new int[] {2, 2}).build())
                                        .layer(6, new DropoutLayer.Builder(0.5).build())

                                        // block 2
                                        .layer(7, new ConvolutionLayer.Builder(new int[] {5, 5}).nOut(32).build())
                                        .layer(8, new BatchNormalization.Builder().build())
                                        .layer(9, new ConvolutionLayer.Builder(new int[] {5, 5}).nOut(32).build())
                                        .layer(10, new BatchNormalization.Builder().build())
                                        .layer(11, new ActivationLayer.Builder().activation(Activation.RELU).build())
                                        .layer(12, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG,
                                                        new int[] {2, 2}).build())
                                        .layer(13, new DropoutLayer.Builder(0.5).build())

                                        // block 3
                                        .layer(14, new ConvolutionLayer.Builder(new int[] {3, 3}).nOut(64).build())
                                        .layer(15, new BatchNormalization.Builder().build())
                                        .layer(16, new ConvolutionLayer.Builder(new int[] {3, 3}).nOut(64).build())
                                        .layer(17, new BatchNormalization.Builder().build())
                                        .layer(18, new ActivationLayer.Builder().activation(Activation.RELU).build())
                                        .layer(19, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG,
                                                        new int[] {2, 2}).build())
                                        .layer(20, new DropoutLayer.Builder(0.5).build())

                                        // block 4
                                        .layer(21, new ConvolutionLayer.Builder(new int[] {3, 3}).nOut(128).build())
                                        .layer(22, new BatchNormalization.Builder().build())
                                        .layer(23, new ConvolutionLayer.Builder(new int[] {3, 3}).nOut(128).build())
                                        .layer(24, new BatchNormalization.Builder().build())
                                        .layer(25, new ActivationLayer.Builder().activation(Activation.RELU).build())
                                        .layer(26, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG,
                                                        new int[] {2, 2}).build())
                                        .layer(27, new DropoutLayer.Builder(0.5).build())


                                        // block 5
                                        .layer(28, new ConvolutionLayer.Builder(new int[] {3, 3}).nOut(256).build())
                                        .layer(29, new BatchNormalization.Builder().build())
                                        .layer(30, new ConvolutionLayer.Builder(new int[] {3, 3}).nOut(numClasses)
                                                        .build())
                                        .layer(31, new GlobalPoolingLayer.Builder(PoolingType.AVG).build())
                                        .layer(32, new ActivationLayer.Builder().activation(Activation.SOFTMAX).build())

                                        .setInputType(InputType.convolutional(inputShape[2], inputShape[1],
                                                        inputShape[0]))
                                        .backprop(true).pretrain(false).build();

        return conf;
    }

    @Override
    public Model init() {
        MultiLayerNetwork network = new MultiLayerNetwork(conf());
        network.init();
        return network;
    }

    @Override
    public ModelMetaData metaData() {
        return new ModelMetaData(new int[][] {inputShape}, 1, ZooType.CNN);
    }

    @Override
    public void setInputShape(int[][] inputShape) {
        this.inputShape = inputShape[0];
    }
}

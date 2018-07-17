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
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.ZooType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * LSTM designed for text generation. Can be trained on a corpus of text. For this model, numClasses is
 * used to input {@code totalUniqueCharacters} for the LSTM input layer.
 *
 * Architecture follows this implementation: https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
 *
 * <p>Walt Whitman weights are available for generating text from his works, adapted from https://github.com/craigomac/InfiniteMonkeys.</p>
 *
 * @author Justin Long (crockpotveggies)
 */
@AllArgsConstructor
@Builder
public class TextGenerationLSTM extends ZooModel {

    @Builder.Default private long seed = 1234;
    @Builder.Default private int maxLength = 40;
    @Builder.Default private int totalUniqueCharacters = 47;
    private int[] inputShape = new int[] {maxLength, totalUniqueCharacters};
    @Builder.Default private IUpdater updater = new RmsProp(0.01);
    @Builder.Default private CacheMode cacheMode = CacheMode.NONE;
    @Builder.Default private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
    @Builder.Default private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

    private TextGenerationLSTM() {}

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
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .l2(0.001)
                        .weightInit(WeightInit.XAVIER)
                        .updater(updater)
                        .cacheMode(cacheMode)
                        .trainingWorkspaceMode(workspaceMode)
                        .inferenceWorkspaceMode(workspaceMode)
                        .cudnnAlgoMode(cudnnAlgoMode)
                        .list()
                        .layer(0, new GravesLSTM.Builder().nIn(inputShape[1]).nOut(256).activation(Activation.TANH)
                                        .build())
                        .layer(1, new GravesLSTM.Builder().nOut(256).activation(Activation.TANH).build())
                        .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                        .activation(Activation.SOFTMAX) //MCXENT + softmax for classification
                                        .nOut(totalUniqueCharacters).build())
                        .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(50).tBPTTBackwardLength(50)
                        .pretrain(false).backprop(true).build();

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
        return new ModelMetaData(new int[][] {inputShape}, 1, ZooType.RNN);
    }

    @Override
    public void setInputShape(int[][] inputShape) {
        this.inputShape = inputShape[0];
    }
}

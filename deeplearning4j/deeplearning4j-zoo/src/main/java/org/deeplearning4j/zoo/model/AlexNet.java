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
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.ZooType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;


/**
 * AlexNet
 *
 * Dl4j's AlexNet model interpretation based on the original paper ImageNet Classification with Deep Convolutional Neural Networks
 * and the imagenetExample code referenced.
 *<br>
 * References:<br>
 * <a href="http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf">http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf</a>
 * <a href="https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/train_val.prototxt">https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/train_val.prototxt</a>
 *
 * Model is built in dl4j based on available functionality and notes indicate where there are gaps waiting for enhancements.
 *
 * Bias initialization in the paper is 1 in certain layers but 0.1 in the imagenetExample code
 * Weight distribution uses 0.1 std for all layers in the paper but 0.005 in the dense layers in the imagenetExample code
 *
 */
@AllArgsConstructor
@Builder
public class AlexNet extends ZooModel {

    @Builder.Default private long seed = 1234;
    @Builder.Default private int[] inputShape = new int[] {3, 224, 224};
    @Builder.Default private int numClasses = 0;
    @Builder.Default private IUpdater updater = new Nesterovs(1e-2, 0.9);
    @Builder.Default private CacheMode cacheMode = CacheMode.NONE;
    @Builder.Default private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
    @Builder.Default private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

    private AlexNet() {}

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
        double nonZeroBias = 1;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed)
                        .weightInit(new NormalDistribution(0.0, 0.01))
                        .activation(Activation.RELU)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(updater)
                        .biasUpdater(new Nesterovs(2e-2, 0.9))
                        .convolutionMode(ConvolutionMode.Same)
                        .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                        .trainingWorkspaceMode(workspaceMode)
                        .inferenceWorkspaceMode(workspaceMode)
                        .cacheMode(cacheMode)
                        .l2(5 * 1e-4)
                        .miniBatch(false)
                        .list()
                .layer(0, new ConvolutionLayer.Builder(new int[]{11,11}, new int[]{4, 4})
                        .name("cnn1")
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(ConvolutionMode.Truncate)
                        .nIn(inputShape[0])
                        .nOut(96)
                        .build())
                .layer(1, new LocalResponseNormalization.Builder().build())
                .layer(2, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3,3)
                        .stride(2,2)
                        .padding(1,1)
                        .name("maxpool1")
                        .build())
                .layer(3, new ConvolutionLayer.Builder(new int[]{5,5}, new int[]{1,1}, new int[]{2,2})
                        .name("cnn2")
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(ConvolutionMode.Truncate)
                        .nOut(256)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(4, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3}, new int[]{2, 2})
                        .convolutionMode(ConvolutionMode.Truncate)
                        .name("maxpool2")
                        .build())
                .layer(5, new LocalResponseNormalization.Builder().build())
                .layer(6, new ConvolutionLayer.Builder()
                        .kernelSize(3,3)
                        .stride(1,1)
                        .convolutionMode(ConvolutionMode.Same)
                        .name("cnn3")
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .nOut(384)
                        .build())
                .layer(7, new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{1,1})
                        .name("cnn4")
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .nOut(384)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(8, new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{1,1})
                        .name("cnn5")
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .nOut(256)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(9, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3,3}, new int[]{2,2})
                        .name("maxpool3")
                        .convolutionMode(ConvolutionMode.Truncate)
                        .build())
                .layer(10, new DenseLayer.Builder()
                        .name("ffn1")
                        .nIn(256*6*6)
                        .nOut(4096)
                        .dist(new GaussianDistribution(0, 0.005))
                        .biasInit(nonZeroBias)
                        .build())
                .layer(11, new DenseLayer.Builder()
                        .name("ffn2")
                        .nOut(4096)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new GaussianDistribution(0, 0.005))
                        .biasInit(nonZeroBias)
                        .dropOut(0.5)
                        .build())
                .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new GaussianDistribution(0, 0.005))
                        .biasInit(0.1)
                        .build())


                .setInputType(InputType.convolutional(inputShape[2], inputShape[1], inputShape[0]))
                .build();

        return conf;
    }

    @Override
    public MultiLayerNetwork init() {
        MultiLayerConfiguration conf = conf();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
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

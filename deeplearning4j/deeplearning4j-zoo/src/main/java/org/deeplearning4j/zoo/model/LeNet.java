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
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.ZooType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * LeNet was an early promising achiever on the ImageNet dataset.
 * References:<br>
 * - <a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf">http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf</a><br>
 * - <a href="https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt">https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt</a><br>
 *
 * <p>MNIST weights for this model are available and have been converted from <a href="https://github.com/f00-/mnist-lenet-keras">https://github.com/f00-/mnist-lenet-keras</a>.</p>
 *
 * @author kepricon
 * @author Justin Long (crockpotveggies)
 */
@AllArgsConstructor
@Builder
public class LeNet extends ZooModel {

    @Builder.Default private long seed = 1234;
    @Builder.Default private int[] inputShape = new int[] {3, 224, 224};
    @Builder.Default private int numClasses = 0;
    @Builder.Default private IUpdater updater = new AdaDelta();
    @Builder.Default private CacheMode cacheMode = CacheMode.NONE;
    @Builder.Default private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
    @Builder.Default private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

    private LeNet() {}

    @Override
    public String pretrainedUrl(PretrainedType pretrainedType) {
        if (pretrainedType == PretrainedType.MNIST)
            return DL4JResources.getURLString("models/lenet_dl4j_mnist_inference.zip");
        else
            return null;
    }

    @Override
    public long pretrainedChecksum(PretrainedType pretrainedType) {
        if (pretrainedType == PretrainedType.MNIST)
            return 1906861161L;
        else
            return 0L;
    }

    @Override
    public Class<? extends Model> modelType() {
        return MultiLayerNetwork.class;
    }

    public MultiLayerConfiguration conf() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed)
                        .activation(Activation.IDENTITY)
                        .weightInit(WeightInit.XAVIER)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(updater)
                        .cacheMode(cacheMode)
                        .trainingWorkspaceMode(workspaceMode)
                        .inferenceWorkspaceMode(workspaceMode)
                        .cudnnAlgoMode(cudnnAlgoMode)
                        .convolutionMode(ConvolutionMode.Same)
                        .list()
                        // block 1
                        .layer(0, new ConvolutionLayer.Builder(new int[] {5, 5}, new int[] {1, 1}).name("cnn1")
                                        .nIn(inputShape[0]).nOut(20).activation(Activation.RELU).build())
                        .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {2, 2},
                                        new int[] {2, 2}).name("maxpool1").build())
                        // block 2
                        .layer(2, new ConvolutionLayer.Builder(new int[] {5, 5}, new int[] {1, 1}).name("cnn2").nOut(50)
                                        .activation(Activation.RELU).build())
                        .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {2, 2},
                                        new int[] {2, 2}).name("maxpool2").build())
                        // fully connected
                        .layer(4, new DenseLayer.Builder().name("ffn1").activation(Activation.RELU).nOut(500).build())
                        // output
                        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).name("output")
                                        .nOut(numClasses).activation(Activation.SOFTMAX) // radial basis function required
                                        .build())
                        .setInputType(InputType.convolutionalFlat(inputShape[2], inputShape[1], inputShape[0]))
                        .build();

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

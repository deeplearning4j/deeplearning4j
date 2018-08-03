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
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.ZooType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * U-Net
 *
 * An implementation of Xception in Deeplearning4j. A novel deep convolutional neural network architecture inspired by Inception, where Inception modules have been replaced with depthwise separable convolutions.
 *
 * <p>Paper: https://arxiv.org/abs/1610.02357</p>
 * <p>ImageNet weights for this model are available and have been converted from https://keras.io/applications/.</p>
 *
 * @author Justin Long (crockpotveggies)
 *
 */
@AllArgsConstructor
@Builder
public class Xception extends ZooModel {

    @Builder.Default private long seed = 1234;
    @Builder.Default private int[] inputShape = new int[] {3, 299, 299};
    @Builder.Default private int numClasses = 0;
    @Builder.Default private WeightInit weightInit = WeightInit.RELU;
    @Builder.Default private IUpdater updater = new AdaDelta();
    @Builder.Default private CacheMode cacheMode = CacheMode.NONE;
    @Builder.Default private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
    @Builder.Default private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

    private Xception() {}

    @Override
    public String pretrainedUrl(PretrainedType pretrainedType) {
        if (pretrainedType == PretrainedType.IMAGENET)
            return DL4JResources.getURLString("models/xception_dl4j_inference.v2.zip");
        else
            return null;
    }

    @Override
    public long pretrainedChecksum(PretrainedType pretrainedType) {
        if (pretrainedType == PretrainedType.IMAGENET)
            return 3277876097L;
        else
            return 0L;
    }

    @Override
    public Class<? extends Model> modelType() {
        return ComputationGraph.class;
    }

    @Override
    public ComputationGraph init() {
        ComputationGraphConfiguration.GraphBuilder graph = graphBuilder();

        graph.addInputs("input").setInputTypes(InputType.convolutional(inputShape[2], inputShape[1], inputShape[0]));

        ComputationGraphConfiguration conf = graph.build();
        ComputationGraph model = new ComputationGraph(conf);
        model.init();

        return model;
    }

    public ComputationGraphConfiguration.GraphBuilder graphBuilder() {

        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(updater)
                .weightInit(weightInit)
                .l2(4e-5)
                .miniBatch(true)
                .cacheMode(cacheMode)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .convolutionMode(ConvolutionMode.Truncate)
                .graphBuilder();


        graph
                // block1
                .addLayer("block1_conv1", new ConvolutionLayer.Builder(3,3).stride(2,2).nOut(32).hasBias(false)
                        .cudnnAlgoMode(cudnnAlgoMode).build(), "input")
                .addLayer("block1_conv1_bn", new BatchNormalization(), "block1_conv1")
                .addLayer("block1_conv1_act", new ActivationLayer(Activation.RELU), "block1_conv1_bn")
                .addLayer("block1_conv2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(64).hasBias(false)
                        .cudnnAlgoMode(cudnnAlgoMode).build(), "block1_conv1_act")
                .addLayer("block1_conv2_bn", new BatchNormalization(), "block1_conv2")
                .addLayer("block1_conv2_act", new ActivationLayer(Activation.RELU), "block1_conv1_bn")

                // residual1
                .addLayer("residual1_conv", new ConvolutionLayer.Builder(1,1).stride(2,2).nOut(128).hasBias(false)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), "block1_conv2_act")
                .addLayer("residual1", new BatchNormalization(), "residual1_conv")

                // block2
                .addLayer("block2_sepconv1", new SeparableConvolution2D.Builder(3,3).nOut(128).hasBias(false)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), "block1_conv1_bn")
                .addLayer("block2_sepconv1_bn", new BatchNormalization(), "block2_sepconv1")
                .addLayer("block2_sepconv1_act",new ActivationLayer(Activation.RELU), "block2_sepconv1_bn")
                .addLayer("block2_sepconv2", new SeparableConvolution2D.Builder(3,3).nOut(128).hasBias(false)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), "block2_sepconv1_act")
                .addLayer("block2_sepconv2_bn", new BatchNormalization(), "block2_sepconv2")
                .addLayer("block2_pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(3,3).stride(2,2)
                        .convolutionMode(ConvolutionMode.Same).build(), "block2_sepconv2_bn")
                .addVertex("add1", new ElementWiseVertex(ElementWiseVertex.Op.Add), "block2_pool", "residual1")

                // residual2
                .addLayer("residual2_conv", new ConvolutionLayer.Builder(1,1).stride(2,2).nOut(256).hasBias(false)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), "add1")
                .addLayer("residual2", new BatchNormalization(), "residual2_conv")

                // block3
                .addLayer("block3_sepconv1_act", new ActivationLayer(Activation.RELU), "add1")
                .addLayer("block3_sepconv1", new SeparableConvolution2D.Builder(3,3).nOut(256).hasBias(false)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), "block3_sepconv1_act")
                .addLayer("block3_sepconv1_bn", new BatchNormalization(), "block3_sepconv1")
                .addLayer("block3_sepconv2_act", new ActivationLayer(Activation.RELU), "block3_sepconv1_bn")
                .addLayer("block3_sepconv2", new SeparableConvolution2D.Builder(3,3).nOut(256).hasBias(false)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), "block3_sepconv2_act")
                .addLayer("block3_sepconv2_bn", new BatchNormalization(), "block3_sepconv2")
                .addLayer("block3_pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(3,3).stride(2,2)
                        .convolutionMode(ConvolutionMode.Same).build(), "block3_sepconv2_bn")
                .addVertex("add2", new ElementWiseVertex(ElementWiseVertex.Op.Add), "block3_pool", "residual2")

                // residual3
                .addLayer("residual3_conv", new ConvolutionLayer.Builder(1,1).stride(2,2).nOut(728).hasBias(false)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), "add1")
                .addLayer("residual3", new BatchNormalization(), "residual3_conv")

                // block4
                .addLayer("block4_sepconv1_act", new ActivationLayer(Activation.RELU), "add2")
                .addLayer("block4_sepconv1", new SeparableConvolution2D.Builder(3,3).nOut(728).hasBias(false)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), "block4_sepconv1_act")
                .addLayer("block4_sepconv1_bn", new BatchNormalization(), "block4_sepconv1")
                .addLayer("block4_sepconv2_act", new ActivationLayer(Activation.RELU), "block4_sepconv1_bn")
                .addLayer("block4_sepconv2", new SeparableConvolution2D.Builder(3,3).nOut(728).hasBias(false)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), "block4_sepconv2_act")
                .addLayer("block4_sepconv2_bn", new BatchNormalization(), "block4_sepconv2")
                .addLayer("block4_pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(3,3).stride(2,2)
                        .convolutionMode(ConvolutionMode.Same).build(), "block4_sepconv2_bn")
                .addVertex("add3", new ElementWiseVertex(ElementWiseVertex.Op.Add), "block4_pool", "residual3");

        // towers
        int residual = 3;
        int block = 5;
        for(int i = 0; i < 8; i++) {
            String previousInput = "add"+residual;
            String blockName = "block"+block;

            graph
                    .addLayer(blockName+"_sepconv1_act", new ActivationLayer(Activation.RELU), previousInput)
                    .addLayer(blockName+"_sepconv1", new SeparableConvolution2D.Builder(3,3).nOut(728).hasBias(false)
                            .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), blockName+"_sepconv1_act")
                    .addLayer(blockName+"_sepconv1_bn", new BatchNormalization(), blockName+"_sepconv1")
                    .addLayer(blockName+"_sepconv2_act", new ActivationLayer(Activation.RELU), blockName+"_sepconv1_bn")
                    .addLayer(blockName+"_sepconv2", new SeparableConvolution2D.Builder(3,3).nOut(728).hasBias(false)
                            .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), blockName+"_sepconv2_act")
                    .addLayer(blockName+"_sepconv2_bn", new BatchNormalization(), blockName+"_sepconv2")
                    .addLayer(blockName+"_sepconv3_act", new ActivationLayer(Activation.RELU), blockName+"_sepconv2_bn")
                    .addLayer(blockName+"_sepconv3", new SeparableConvolution2D.Builder(3,3).nOut(728).hasBias(false)
                            .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), blockName+"_sepconv3_act")
                    .addLayer(blockName+"_sepconv3_bn", new BatchNormalization(), blockName+"_sepconv3")
                    .addVertex("add"+(residual+1), new ElementWiseVertex(ElementWiseVertex.Op.Add), blockName+"_sepconv3_bn", previousInput);

            residual += 1;
            block += 1;
        }

        graph
                // residual11
                .addLayer("residual11_conv", new ConvolutionLayer.Builder(1,1).stride(2,2).nOut(1024).hasBias(false)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), "add10")
                .addLayer("residual11", new BatchNormalization(), "residual11_conv")

                // block13
                .addLayer("block13_sepconv1_act", new ActivationLayer(Activation.RELU), "add10")
                .addLayer("block13_sepconv1", new SeparableConvolution2D.Builder(3,3).nOut(728).hasBias(false)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), "block13_sepconv1_act")
                .addLayer("block13_sepconv1_bn", new BatchNormalization(), "block13_sepconv1")
                .addLayer("block13_sepconv2_act", new ActivationLayer(Activation.RELU), "block13_sepconv1_bn")
                .addLayer("block13_sepconv2", new SeparableConvolution2D.Builder(3,3).nOut(1024).hasBias(false)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), "block13_sepconv2_act")
                .addLayer("block13_sepconv2_bn", new BatchNormalization(), "block13_sepconv2")
                .addLayer("block13_pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(3,3).stride(2,2)
                        .convolutionMode(ConvolutionMode.Same).build(), "block13_sepconv2_bn")
                .addVertex("add11", new ElementWiseVertex(ElementWiseVertex.Op.Add), "block13_pool", "residual11")

                // block14
                .addLayer("block14_sepconv1", new SeparableConvolution2D.Builder(3,3).nOut(1536).hasBias(false)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), "add11")
                .addLayer("block14_sepconv1_bn", new BatchNormalization(), "block14_sepconv1")
                .addLayer("block14_sepconv1_act", new ActivationLayer(Activation.RELU), "block14_sepconv1_bn")
                .addLayer("block14_sepconv2", new SeparableConvolution2D.Builder(3,3).nOut(2048).hasBias(false)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode).build(), "add11")
                .addLayer("block14_sepconv2_bn", new BatchNormalization(), "block14_sepconv2")
                .addLayer("block14_sepconv2_act", new ActivationLayer(Activation.RELU), "block14_sepconv2_bn")

                .addLayer("avg_pool", new GlobalPoolingLayer.Builder(PoolingType.AVG).build(), "block14_sepconv2_act")
                .addLayer("predictions", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).build(), "avg_pool")

                .setOutputs("predictions")
                .backprop(true)
                .pretrain(false);

        return graph;
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

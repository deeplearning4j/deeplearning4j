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
import org.deeplearning4j.nn.conf.graph.MergeVertex;
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
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * U-Net
 *
 * An implementation of SqueezeNet. Touts similar accuracy to AlexNet with a fraction of the parameters.
 *
 * <p>Paper: https://arxiv.org/abs/1602.07360</p>
 * <p>ImageNet weights for this model are available and have been converted from https://github.com/rcmalli/keras-squeezenet/.</p>
 *
 * @note Pretrained ImageNet weights are "special". Output shape is (1,1000,1,1).
 * @author Justin Long (crockpotveggies)
 *
 */
@AllArgsConstructor
@Builder
public class SqueezeNet extends ZooModel {

    @Builder.Default private long seed = 1234;
    @Builder.Default private int[] inputShape = new int[] {3, 227, 227};
    @Builder.Default private int numClasses = 0;
    @Builder.Default private WeightInit weightInit = WeightInit.RELU;
    @Builder.Default private IUpdater updater = new AdaDelta();
    @Builder.Default private CacheMode cacheMode = CacheMode.NONE;
    @Builder.Default private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
    @Builder.Default private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

    private SqueezeNet() {}

    @Override
    public String pretrainedUrl(PretrainedType pretrainedType) {
        if (pretrainedType == PretrainedType.IMAGENET)
            return DL4JResources.getURLString("models/squeezenet_dl4j_inference.v2.zip");
        else
            return null;
    }

    @Override
    public long pretrainedChecksum(PretrainedType pretrainedType) {
        if (pretrainedType == PretrainedType.IMAGENET)
            return 3711411239L;
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
                .l2(5e-5)
                .miniBatch(true)
                .cacheMode(cacheMode)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .convolutionMode(ConvolutionMode.Truncate)
                .graphBuilder();


        graph
                // stem
                .addLayer("conv1", new ConvolutionLayer.Builder(3,3).stride(2,2).nOut(64)
                        .cudnnAlgoMode(cudnnAlgoMode).build(), "input")
                .addLayer("conv1_act", new ActivationLayer(Activation.RELU), "conv1")
                .addLayer("pool1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(3,3).stride(2,2).build(), "conv1_act");

        // fire modules
        fireModule(graph, 2, 16, 64, "pool1");
        fireModule(graph, 3, 16, 64, "fire2");
        graph.addLayer("pool3", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(3,3).stride(2,2).build(), "fire3");

        fireModule(graph, 4, 32, 128, "pool3");
        fireModule(graph, 5, 32, 128, "fire4");
        graph.addLayer("pool5", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(3,3).stride(2,2).build(), "fire5");

        fireModule(graph, 6, 48, 192, "pool5");
        fireModule(graph, 7, 48, 192, "fire6");
        fireModule(graph, 8, 64, 256, "fire7");
        fireModule(graph, 9, 64, 256, "fire8");

        graph
                // output
                .addLayer("drop9", new DropoutLayer.Builder(0.5).build(), "fire9")
                .addLayer("conv10", new ConvolutionLayer.Builder(1,1).nOut(numClasses)
                        .cudnnAlgoMode(cudnnAlgoMode).build(), "input")
                .addLayer("conv10_act", new ActivationLayer(Activation.RELU), "conv10")
                .addLayer("avg_pool", new GlobalPoolingLayer(PoolingType.AVG), "conv10_act")

                .addLayer("softmax", new ActivationLayer(Activation.SOFTMAX), "avg_pool")
                .addLayer("loss", new LossLayer.Builder(LossFunctions.LossFunction.MCXENT).build(), "softmax")

                .setOutputs("loss")

                ;

        return graph;
    }

    private String fireModule(ComputationGraphConfiguration.GraphBuilder graphBuilder, int fireId, int squeeze, int expand, String input) {
        String prefix = "fire"+fireId;

        graphBuilder
                .addLayer(prefix+"_sq1x1", new ConvolutionLayer.Builder(1, 1).nOut(squeeze)
                        .cudnnAlgoMode(cudnnAlgoMode).build(), input)
                .addLayer(prefix+"_relu_sq1x1", new ActivationLayer(Activation.RELU), prefix+"_sq1x1")

                .addLayer(prefix+"exp1x1", new ConvolutionLayer.Builder(1, 1).nOut(expand)
                        .cudnnAlgoMode(cudnnAlgoMode).build(), prefix+"_relu_sq1x1")
                .addLayer(prefix+"_relu_exp1x1", new ActivationLayer(Activation.RELU), prefix+"_exp1x1")

                .addLayer(prefix+"_exp3x3", new ConvolutionLayer.Builder(3,3).nOut(expand)
                        .cudnnAlgoMode(cudnnAlgoMode).build(), prefix+"_relu_sq1x1")
                .addLayer(prefix+"_relu_exp3x3", new ActivationLayer(Activation.RELU), prefix+"_exp3x3")

                .addVertex(prefix, new MergeVertex(), prefix+"_relu_exp1x1", prefix+"_relu_exp3x3");

        return prefix;
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

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
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.TruncatedNormalDistribution;
import org.deeplearning4j.nn.conf.graph.L2NormalizeVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.ZooType;
import org.deeplearning4j.zoo.model.helper.InceptionResNetHelper;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * A variant of the original FaceNet model that relies on embeddings and triplet loss.<br>
 * Reference: <a href="https://arxiv.org/abs/1503.03832">https://arxiv.org/abs/1503.03832</a><br>
 * Also based on the OpenFace implementation: <a href="http://reports-archive.adm.cs.cmu.edu/anon/2016/CMU-CS-16-118.pdf">
 * http://reports-archive.adm.cs.cmu.edu/anon/2016/CMU-CS-16-118.pdf</a>
 *
 * Revised and consolidated version by @crockpotveggies
 */
@AllArgsConstructor
@Builder
public class InceptionResNetV1 extends ZooModel {

    @Builder.Default private long seed = 1234;
    @Builder.Default private int[] inputShape = new int[] {3, 160, 160};
    @Builder.Default private int numClasses = 0;
    @Builder.Default private IUpdater updater = new RmsProp(0.1, 0.96, 0.001);
    @Builder.Default private CacheMode cacheMode = CacheMode.NONE;
    @Builder.Default private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
    @Builder.Default private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

    private InceptionResNetV1() {}

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
        return ComputationGraph.class;
    }

    @Override
    public ComputationGraph init() {
        int embeddingSize = 128;
        ComputationGraphConfiguration.GraphBuilder graph = graphBuilder("input1");

        graph.addInputs("input1").setInputTypes(InputType.convolutional(inputShape[2], inputShape[1], inputShape[0]))
                        // Logits
                        .addLayer("bottleneck", new DenseLayer.Builder().nIn(5376).nOut(embeddingSize).build(),
                                        "avgpool")
                        // Embeddings
                        .addVertex("embeddings", new L2NormalizeVertex(new int[] {1}, 1e-10), "bottleneck")
                        // Output
                        .addLayer("outputLayer",
                                        new CenterLossOutputLayer.Builder()
                                                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                                        .activation(Activation.SOFTMAX).alpha(0.9).lambda(1e-4)
                                                        .nIn(embeddingSize).nOut(numClasses).build(),
                                        "embeddings")
                        .setOutputs("outputLayer");

        ComputationGraphConfiguration conf = graph.build();
        ComputationGraph model = new ComputationGraph(conf);
        model.init();

        return model;
    }

    public ComputationGraphConfiguration.GraphBuilder graphBuilder(String input) {

        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)
                        .activation(Activation.RELU)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(updater)
                        .weightInit(new TruncatedNormalDistribution(0.0, 0.5))
                        .l2(5e-5)
                        .miniBatch(true)
                        .cacheMode(cacheMode)
                        .trainingWorkspaceMode(workspaceMode)
                        .inferenceWorkspaceMode(workspaceMode)
                        .convolutionMode(ConvolutionMode.Truncate).graphBuilder();


        graph
                        // stem
                        .addLayer("stem-cnn1",
                                        new ConvolutionLayer.Builder(new int[] {3, 3}, new int[] {2, 2})
                                                        .nIn(inputShape[0]).nOut(32)
                                                        .cudnnAlgoMode(cudnnAlgoMode).build(),
                                        input)
                        .addLayer("stem-batch1",
                                        new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(32).nOut(32)
                                                        .build(),
                                        "stem-cnn1")
                        .addLayer("stem-cnn2",
                                        new ConvolutionLayer.Builder(new int[] {3, 3}).nIn(32).nOut(32)
                                                        .cudnnAlgoMode(cudnnAlgoMode).build(),
                                        "stem-batch1")
                        .addLayer("stem-batch2",
                                        new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(32).nOut(32)
                                                        .build(),
                                        "stem-cnn2")
                        .addLayer("stem-cnn3",
                                        new ConvolutionLayer.Builder(new int[] {3, 3})
                                                        .convolutionMode(ConvolutionMode.Same).nIn(32).nOut(64)
                                                        .cudnnAlgoMode(cudnnAlgoMode).build(),
                                        "stem-batch2")
                        .addLayer("stem-batch3", new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(64)
                                        .nOut(64).build(), "stem-cnn3")

                        .addLayer("stem-pool4", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX,
                                        new int[] {3, 3}, new int[] {2, 2}).build(), "stem-batch3")

                        .addLayer("stem-cnn5",
                                        new ConvolutionLayer.Builder(new int[] {1, 1}).nIn(64).nOut(80)
                                                        .cudnnAlgoMode(cudnnAlgoMode).build(),
                                        "stem-pool4")
                        .addLayer("stem-batch5",
                                        new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(80).nOut(80)
                                                        .build(),
                                        "stem-cnn5")
                        .addLayer("stem-cnn6",
                                        new ConvolutionLayer.Builder(new int[] {3, 3}).nIn(80).nOut(128)
                                                        .cudnnAlgoMode(cudnnAlgoMode).build(),
                                        "stem-batch5")
                        .addLayer("stem-batch6",
                                        new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(128).nOut(128)
                                                        .build(),
                                        "stem-cnn6")
                        .addLayer("stem-cnn7",
                                        new ConvolutionLayer.Builder(new int[] {3, 3}, new int[] {2, 2}).nIn(128)
                                                        .nOut(192).cudnnAlgoMode(cudnnAlgoMode)
                                                        .build(),
                                        "stem-batch6")
                        .addLayer("stem-batch7", new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(192)
                                        .nOut(192).build(), "stem-cnn7");


        // 5xInception-resnet-A
        InceptionResNetHelper.inceptionV1ResA(graph, "resnetA", 5, 0.17, "stem-batch7");


        // Reduction-A
        graph
                        // 3x3
                        .addLayer("reduceA-cnn1",
                                        new ConvolutionLayer.Builder(new int[] {3, 3}, new int[] {2, 2}).nIn(192)
                                                        .nOut(192).cudnnAlgoMode(cudnnAlgoMode)
                                                        .build(),
                                        "resnetA")
                        .addLayer("reduceA-batch1",
                                        new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(192).nOut(192)
                                                        .build(),
                                        "reduceA-cnn1")
                        // 1x1 -> 3x3 -> 3x3
                        .addLayer("reduceA-cnn2",
                                        new ConvolutionLayer.Builder(new int[] {1, 1})
                                                        .convolutionMode(ConvolutionMode.Same).nIn(192).nOut(128)
                                                        .cudnnAlgoMode(cudnnAlgoMode).build(),
                                        "resnetA")
                        .addLayer("reduceA-batch2",
                                        new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(128).nOut(128)
                                                        .build(),
                                        "reduceA-cnn2")
                        .addLayer("reduceA-cnn3",
                                        new ConvolutionLayer.Builder(new int[] {3, 3})
                                                        .convolutionMode(ConvolutionMode.Same).nIn(128).nOut(128)
                                                        .cudnnAlgoMode(cudnnAlgoMode).build(),
                                        "reduceA-batch2")
                        .addLayer("reduceA-batch3",
                                        new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(128).nOut(128)
                                                        .build(),
                                        "reduceA-cnn3")
                        .addLayer("reduceA-cnn4",
                                        new ConvolutionLayer.Builder(new int[] {3, 3}, new int[] {2, 2}).nIn(128)
                                                        .nOut(192).cudnnAlgoMode(cudnnAlgoMode)
                                                        .build(),
                                        "reduceA-batch3")
                        .addLayer("reduceA-batch4",
                                        new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(192).nOut(192)
                                                        .build(),
                                        "reduceA-cnn4")
                        // maxpool
                        .addLayer("reduceA-pool5",
                                        new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {3, 3},
                                                        new int[] {2, 2}).build(),
                                        "resnetA")
                        // -->
                        .addVertex("reduceA", new MergeVertex(), "reduceA-batch1", "reduceA-batch4", "reduceA-pool5");


        // 10xInception-resnet-B
        InceptionResNetHelper.inceptionV1ResB(graph, "resnetB", 10, 0.10, "reduceA");


        // Reduction-B
        graph
                        // 3x3 pool
                        .addLayer("reduceB-pool1",
                                        new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {3, 3},
                                                        new int[] {2, 2}).build(),
                                        "resnetB")
                        // 1x1 -> 3x3
                        .addLayer("reduceB-cnn2",
                                        new ConvolutionLayer.Builder(new int[] {1, 1})
                                                        .convolutionMode(ConvolutionMode.Same).nIn(576).nOut(256)
                                                        .cudnnAlgoMode(cudnnAlgoMode).build(),
                                        "resnetB")
                        .addLayer("reduceB-batch1",
                                        new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(256).nOut(256)
                                                        .build(),
                                        "reduceB-cnn2")
                        .addLayer("reduceB-cnn3",
                                        new ConvolutionLayer.Builder(new int[] {3, 3}, new int[] {2, 2}).nIn(256)
                                                        .nOut(256).cudnnAlgoMode(cudnnAlgoMode)
                                                        .build(),
                                        "reduceB-batch1")
                        .addLayer("reduceB-batch2",
                                        new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(256).nOut(256)
                                                        .build(),
                                        "reduceB-cnn3")
                        // 1x1 -> 3x3
                        .addLayer("reduceB-cnn4",
                                        new ConvolutionLayer.Builder(new int[] {1, 1})
                                                        .convolutionMode(ConvolutionMode.Same).nIn(576).nOut(256)
                                                        .cudnnAlgoMode(cudnnAlgoMode).build(),
                                        "resnetB")
                        .addLayer("reduceB-batch3",
                                        new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(256).nOut(256)
                                                        .build(),
                                        "reduceB-cnn4")
                        .addLayer("reduceB-cnn5",
                                        new ConvolutionLayer.Builder(new int[] {3, 3}, new int[] {2, 2}).nIn(256)
                                                        .nOut(256).cudnnAlgoMode(cudnnAlgoMode)
                                                        .build(),
                                        "reduceB-batch3")
                        .addLayer("reduceB-batch4",
                                        new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(256).nOut(256)
                                                        .build(),
                                        "reduceB-cnn5")
                        // 1x1 -> 3x3 -> 3x3
                        .addLayer("reduceB-cnn6",
                                        new ConvolutionLayer.Builder(new int[] {1, 1})
                                                        .convolutionMode(ConvolutionMode.Same).nIn(576).nOut(256)
                                                        .cudnnAlgoMode(cudnnAlgoMode).build(),
                                        "resnetB")
                        .addLayer("reduceB-batch5",
                                        new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(256).nOut(256)
                                                        .build(),
                                        "reduceB-cnn6")
                        .addLayer("reduceB-cnn7",
                                        new ConvolutionLayer.Builder(new int[] {3, 3})
                                                        .convolutionMode(ConvolutionMode.Same).nIn(256).nOut(256)
                                                        .cudnnAlgoMode(cudnnAlgoMode).build(),
                                        "reduceB-batch5")
                        .addLayer("reduceB-batch6",
                                        new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(256).nOut(256)
                                                        .build(),
                                        "reduceB-cnn7")
                        .addLayer("reduceB-cnn8",
                                        new ConvolutionLayer.Builder(new int[] {3, 3}, new int[] {2, 2}).nIn(256)
                                                        .nOut(256).cudnnAlgoMode(cudnnAlgoMode)
                                                        .build(),
                                        "reduceB-batch6")
                        .addLayer("reduceB-batch7",
                                        new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(256).nOut(256)
                                                        .build(),
                                        "reduceB-cnn8")
                        // -->
                        .addVertex("reduceB", new MergeVertex(), "reduceB-pool1", "reduceB-batch2", "reduceB-batch4",
                                        "reduceB-batch7");


        // 10xInception-resnet-C
        InceptionResNetHelper.inceptionV1ResC(graph, "resnetC", 5, 0.20, "reduceB");

        // Average pooling
        graph.addLayer("avgpool",
                        new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[] {1, 1}).build(),
                        "resnetC");

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

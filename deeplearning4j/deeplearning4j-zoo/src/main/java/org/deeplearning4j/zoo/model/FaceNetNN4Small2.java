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
import org.deeplearning4j.zoo.model.helper.FaceNetHelper;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * A variant of the original FaceNet model that relies on embeddings and triplet loss.
 * Reference: <a href="https://arxiv.org/abs/1503.03832">https://arxiv.org/abs/1503.03832</a><br>
 * Also based on the OpenFace implementation: <a href="http://reports-archive.adm.cs.cmu.edu/anon/2016/CMU-CS-16-118.pdf">
 *     http://reports-archive.adm.cs.cmu.edu/anon/2016/CMU-CS-16-118.pdf</a>
 *
 * Revised and consolidated version by @crockpotveggies
 */
@AllArgsConstructor
@Builder
public class FaceNetNN4Small2 extends ZooModel {

    @Builder.Default private long seed = 1234;
    @Builder.Default private int[] inputShape = new int[] {3, 96, 96};
    @Builder.Default private int numClasses = 0;
    @Builder.Default private IUpdater updater = new Adam(0.1, 0.9, 0.999, 0.01);
    @Builder.Default private Activation transferFunction = Activation.RELU;
    @Builder.Default CacheMode cacheMode = CacheMode.NONE;
    @Builder.Default private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
    @Builder.Default private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;
    @Builder.Default private int embeddingSize = 128;

    private FaceNetNN4Small2() {}

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

    public ComputationGraphConfiguration conf() {

        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)
                        .activation(Activation.IDENTITY)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(updater)
                        .weightInit(WeightInit.RELU)
                        .l2(5e-5)
                        .miniBatch(true)
                        .cacheMode(cacheMode)
                        .trainingWorkspaceMode(workspaceMode)
                        .inferenceWorkspaceMode(workspaceMode)
                        .cudnnAlgoMode(cudnnAlgoMode)
                        .convolutionMode(ConvolutionMode.Same)
                        .graphBuilder();


        graph.addInputs("input1")
                        .addLayer("stem-cnn1",
                                        new ConvolutionLayer.Builder(new int[] {7, 7}, new int[] {2, 2},
                                                        new int[] {3, 3}).nIn(inputShape[0]).nOut(64)
                                                                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                                                                        .build(),
                                        "input1")
                        .addLayer("stem-batch1", new BatchNormalization.Builder(false).nIn(64).nOut(64).build(),
                                        "stem-cnn1")
                        .addLayer("stem-activation1", new ActivationLayer.Builder().activation(Activation.RELU).build(),
                                        "stem-batch1")

                        // pool -> norm
                        .addLayer("stem-pool1",
                                        new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {3, 3},
                                                        new int[] {2, 2}, new int[] {1, 1}).build(),
                                        "stem-activation1")
                        .addLayer("stem-lrn1", new LocalResponseNormalization.Builder(1, 5, 1e-4, 0.75).build(),
                                        "stem-pool1")

                        // Inception 2
                        .addLayer("inception-2-cnn1",
                                        new ConvolutionLayer.Builder(new int[] {1, 1}).nIn(64).nOut(64)
                                                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build(),
                                        "stem-lrn1")
                        .addLayer("inception-2-batch1", new BatchNormalization.Builder(false).nIn(64).nOut(64).build(),
                                        "inception-2-cnn1")
                        .addLayer("inception-2-activation1",
                                        new ActivationLayer.Builder().activation(Activation.RELU).build(),
                                        "inception-2-batch1")
                        .addLayer("inception-2-cnn2",
                                        new ConvolutionLayer.Builder(new int[] {3, 3}, new int[] {1, 1},
                                                        new int[] {1, 1}).nIn(64).nOut(192)
                                                                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                                                                        .build(),
                                        "inception-2-activation1")
                        .addLayer("inception-2-batch2",
                                        new BatchNormalization.Builder(false).nIn(192).nOut(192).build(),
                                        "inception-2-cnn2")
                        .addLayer("inception-2-activation2",
                                        new ActivationLayer.Builder().activation(Activation.RELU).build(),
                                        "inception-2-batch2")

                        // norm -> pool
                        .addLayer("inception-2-lrn1", new LocalResponseNormalization.Builder(1, 5, 1e-4, 0.75).build(),
                                        "inception-2-activation2")
                        .addLayer("inception-2-pool1",
                                        new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {3, 3},
                                                        new int[] {2, 2}, new int[] {1, 1}).build(),
                                        "inception-2-lrn1");

        // Inception 3a
        FaceNetHelper.appendGraph(graph, "3a", 192, new int[] {3, 5}, new int[] {1, 1}, new int[] {128, 32},
                        new int[] {96, 16, 32, 64}, SubsamplingLayer.PoolingType.MAX, transferFunction,
                        "inception-2-pool1");
        // Inception 3b
        FaceNetHelper.appendGraph(graph, "3b", 256, new int[] {3, 5}, new int[] {1, 1}, new int[] {128, 64},
                        new int[] {96, 32, 64, 64}, SubsamplingLayer.PoolingType.PNORM, 2, transferFunction,
                        "inception-3a");
        // Inception 3c
        //    Inception.appendGraph(graph, "3c", 320,
        //        new int[]{3,5}, new int[]{1,1}, new int[]{256,64}, new int[]{128,64},
        //        SubsamplingLayer.PoolingType.PNORM, 2, true, "inception-3b");

        graph.addLayer("3c-1x1",
                        new ConvolutionLayer.Builder(new int[] {1, 1}, new int[] {1, 1}).nIn(320).nOut(128)
                                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build(),
                        "inception-3b")
                        .addLayer("3c-1x1-norm", FaceNetHelper.batchNorm(128, 128), "3c-1x1")
                        .addLayer("3c-transfer1", new ActivationLayer.Builder().activation(transferFunction).build(),
                                        "3c-1x1-norm")
                        .addLayer("3c-3x3",
                                        new ConvolutionLayer.Builder(new int[] {3, 3}, new int[] {2, 2}).nIn(128)
                                                        .nOut(256).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                                                        .build(),
                                        "3c-transfer1")
                        .addLayer("3c-3x3-norm", FaceNetHelper.batchNorm(256, 256), "3c-3x3")
                        .addLayer("3c-transfer2", new ActivationLayer.Builder().activation(transferFunction).build(),
                                        "3c-3x3-norm")

                        .addLayer("3c-2-1x1",
                                        new ConvolutionLayer.Builder(new int[] {1, 1}, new int[] {1, 1}).nIn(320)
                                                        .nOut(32).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                                                        .build(),
                                        "inception-3b")
                        .addLayer("3c-2-1x1-norm", FaceNetHelper.batchNorm(32, 32), "3c-2-1x1")
                        .addLayer("3c-2-transfer3", new ActivationLayer.Builder().activation(transferFunction).build(),
                                        "3c-2-1x1-norm")
                        .addLayer("3c-2-5x5",
                                        new ConvolutionLayer.Builder(new int[] {3, 3}, new int[] {2, 2}).nIn(32)
                                                        .nOut(64).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                                                        .build(),
                                        "3c-2-transfer3")
                        .addLayer("3c-2-5x5-norm", FaceNetHelper.batchNorm(64, 64), "3c-2-5x5")
                        .addLayer("3c-2-transfer4", new ActivationLayer.Builder().activation(transferFunction).build(),
                                        "3c-2-5x5-norm")

                        .addLayer("3c-pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX,
                                        new int[] {3, 3}, new int[] {2, 2}, new int[] {1, 1}).build(), "inception-3b")

                        .addVertex("inception-3c", new MergeVertex(), "3c-transfer2", "3c-2-transfer4", "3c-pool");

        // Inception 4a
        FaceNetHelper.appendGraph(graph, "4a", 640, new int[] {3, 5}, new int[] {1, 1}, new int[] {192, 64},
                        new int[] {96, 32, 128, 256}, SubsamplingLayer.PoolingType.PNORM, 2, transferFunction,
                        "inception-3c");

        //    // Inception 4e
        //    Inception.appendGraph(graph, "4e", 640,
        //        new int[]{3,5}, new int[]{2,2}, new int[]{256,128}, new int[]{160,64},
        //        SubsamplingLayer.PoolingType.MAX, 2, 1, true, "inception-4a");

        graph.addLayer("4e-1x1",
                        new ConvolutionLayer.Builder(new int[] {1, 1}, new int[] {1, 1}).nIn(640).nOut(160)
                                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build(),
                        "inception-4a")
                        .addLayer("4e-1x1-norm", FaceNetHelper.batchNorm(160, 160), "4e-1x1")
                        .addLayer("4e-transfer1", new ActivationLayer.Builder().activation(transferFunction).build(),
                                        "4e-1x1-norm")
                        .addLayer("4e-3x3",
                                        new ConvolutionLayer.Builder(new int[] {3, 3}, new int[] {2, 2}).nIn(160)
                                                        .nOut(256).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                                                        .build(),
                                        "4e-transfer1")
                        .addLayer("4e-3x3-norm", FaceNetHelper.batchNorm(256, 256), "4e-3x3")
                        .addLayer("4e-transfer2", new ActivationLayer.Builder().activation(transferFunction).build(),
                                        "4e-3x3-norm")

                        .addLayer("4e-2-1x1",
                                        new ConvolutionLayer.Builder(new int[] {1, 1}, new int[] {1, 1}).nIn(640)
                                                        .nOut(64).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                                                        .build(),
                                        "inception-4a")
                        .addLayer("4e-2-1x1-norm", FaceNetHelper.batchNorm(64, 64), "4e-2-1x1")
                        .addLayer("4e-2-transfer3", new ActivationLayer.Builder().activation(transferFunction).build(),
                                        "4e-2-1x1-norm")
                        .addLayer("4e-2-5x5",
                                        new ConvolutionLayer.Builder(new int[] {3, 3}, new int[] {2, 2}).nIn(64)
                                                        .nOut(128).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                                                        .build(),
                                        "4e-2-transfer3")
                        .addLayer("4e-2-5x5-norm", FaceNetHelper.batchNorm(128, 128), "4e-2-5x5")
                        .addLayer("4e-2-transfer4", new ActivationLayer.Builder().activation(transferFunction).build(),
                                        "4e-2-5x5-norm")

                        .addLayer("4e-pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX,
                                        new int[] {3, 3}, new int[] {2, 2}, new int[] {1, 1}).build(), "inception-4a")

                        .addVertex("inception-4e", new MergeVertex(), "4e-transfer2", "4e-2-transfer4", "4e-pool");

        // Inception 5a
        //    Inception.appendGraph(graph, "5a", 1024,
        //        new int[]{3}, new int[]{1}, new int[]{384}, new int[]{96,96,256},
        //        SubsamplingLayer.PoolingType.PNORM, 2, true, "inception-4e");

        graph.addLayer("5a-1x1",
                        new ConvolutionLayer.Builder(new int[] {1, 1}, new int[] {1, 1}).nIn(1024).nOut(256)
                                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build(),
                        "inception-4e").addLayer("5a-1x1-norm", FaceNetHelper.batchNorm(256, 256), "5a-1x1")
                        .addLayer("5a-transfer1", new ActivationLayer.Builder().activation(transferFunction).build(),
                                        "5a-1x1-norm")

                        .addLayer("5a-2-1x1",
                                        new ConvolutionLayer.Builder(new int[] {1, 1}, new int[] {1, 1}).nIn(1024)
                                                        .nOut(96).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                                                        .build(),
                                        "inception-4e")
                        .addLayer("5a-2-1x1-norm", FaceNetHelper.batchNorm(96, 96), "5a-2-1x1")
                        .addLayer("5a-2-transfer2", new ActivationLayer.Builder().activation(transferFunction).build(),
                                        "5a-2-1x1-norm")
                        .addLayer("5a-2-3x3",
                                        new ConvolutionLayer.Builder(new int[] {3, 3}, new int[] {1, 1}).nIn(96)
                                                        .nOut(384).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                                                        .build(),
                                        "5a-2-transfer2")
                        .addLayer("5a-2-3x3-norm", FaceNetHelper.batchNorm(384, 384), "5a-2-3x3")
                        .addLayer("5a-transfer3", new ActivationLayer.Builder().activation(transferFunction).build(),
                                        "5a-2-3x3-norm")

                        .addLayer("5a-3-pool",
                                        new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.PNORM,
                                                        new int[] {3, 3}, new int[] {1, 1}).pnorm(2).build(),
                                        "inception-4e")
                        .addLayer("5a-3-1x1reduce",
                                        new ConvolutionLayer.Builder(new int[] {1, 1}, new int[] {1, 1}).nIn(1024)
                                                        .nOut(96).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                                                        .build(),
                                        "5a-3-pool")
                        .addLayer("5a-3-1x1reduce-norm", FaceNetHelper.batchNorm(96, 96), "5a-3-1x1reduce")
                        .addLayer("5a-3-transfer4", new ActivationLayer.Builder().activation(Activation.RELU).build(),
                                        "5a-3-1x1reduce-norm")

                        .addVertex("inception-5a", new MergeVertex(), "5a-transfer1", "5a-transfer3", "5a-3-transfer4");


        // Inception 5b
        //    Inception.appendGraph(graph, "5b", 736,
        //        new int[]{3}, new int[]{1}, new int[]{384}, new int[]{96,96,256},
        //        SubsamplingLayer.PoolingType.MAX, 1, 1, true, "inception-5a");

        graph.addLayer("5b-1x1",
                        new ConvolutionLayer.Builder(new int[] {1, 1}, new int[] {1, 1}).nIn(736).nOut(256)
                                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build(),
                        "inception-5a").addLayer("5b-1x1-norm", FaceNetHelper.batchNorm(256, 256), "5b-1x1")
                        .addLayer("5b-transfer1", new ActivationLayer.Builder().activation(transferFunction).build(),
                                        "5b-1x1-norm")

                        .addLayer("5b-2-1x1",
                                        new ConvolutionLayer.Builder(new int[] {1, 1}, new int[] {1, 1}).nIn(736)
                                                        .nOut(96).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                                                        .build(),
                                        "inception-5a")
                        .addLayer("5b-2-1x1-norm", FaceNetHelper.batchNorm(96, 96), "5b-2-1x1")
                        .addLayer("5b-2-transfer2", new ActivationLayer.Builder().activation(transferFunction).build(),
                                        "5b-2-1x1-norm")
                        .addLayer("5b-2-3x3",
                                        new ConvolutionLayer.Builder(new int[] {3, 3}, new int[] {1, 1}).nIn(96)
                                                        .nOut(384).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                                                        .build(),
                                        "5b-2-transfer2")
                        .addLayer("5b-2-3x3-norm", FaceNetHelper.batchNorm(384, 384), "5b-2-3x3")
                        .addLayer("5b-2-transfer3", new ActivationLayer.Builder().activation(transferFunction).build(),
                                        "5b-2-3x3-norm")

                        .addLayer("5b-3-pool",
                                        new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {3, 3},
                                                        new int[] {1, 1}, new int[] {1, 1}).build(),
                                        "inception-5a")
                        .addLayer("5b-3-1x1reduce",
                                        new ConvolutionLayer.Builder(new int[] {1, 1}, new int[] {1, 1}).nIn(736)
                                                        .nOut(96).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                                                        .build(),
                                        "5b-3-pool")
                        .addLayer("5b-3-1x1reduce-norm", FaceNetHelper.batchNorm(96, 96), "5b-3-1x1reduce")
                        .addLayer("5b-3-transfer4", new ActivationLayer.Builder().activation(transferFunction).build(),
                                        "5b-3-1x1reduce-norm")

                        .addVertex("inception-5b", new MergeVertex(), "5b-transfer1", "5b-2-transfer3",
                                        "5b-3-transfer4");

        graph.addLayer("avgpool",
                        new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[] {3, 3},
                                        new int[] {3, 3}).build(),
                        "inception-5b")
                        .addLayer("bottleneck",new DenseLayer.Builder().nOut(embeddingSize)
                                        .activation(Activation.IDENTITY).build(),"avgpool")
                        .addVertex("embeddings", new L2NormalizeVertex(new int[] {}, 1e-6), "bottleneck")
                        .addLayer("lossLayer", new CenterLossOutputLayer.Builder()
                                        .lossFunction(LossFunctions.LossFunction.SQUARED_LOSS)
                                        .activation(Activation.SOFTMAX).nOut(numClasses).lambda(1e-4).alpha(0.9)
                                        .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer).build(),
                                        "embeddings")
                        .setOutputs("lossLayer").backprop(true).pretrain(false)
                        .setInputTypes(InputType.convolutional(inputShape[2], inputShape[1], inputShape[0]));

        return graph.build();
    }

    @Override
    public ComputationGraph init() {
        ComputationGraph model = new ComputationGraph(conf());
        model.init();

        return model;
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

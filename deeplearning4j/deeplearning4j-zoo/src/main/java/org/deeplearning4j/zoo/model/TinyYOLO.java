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
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.ZooType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;

import static org.deeplearning4j.zoo.model.helper.DarknetHelper.addLayers;

/**
 * Tiny YOLO
 *  Reference: https://arxiv.org/pdf/1612.08242.pdf
 *
 * <p>ImageNet+VOC weights for this model are available and have been converted from https://pjreddie.com/darknet/yolo/
 * using https://github.com/allanzelener/YAD2K and the following code.</p>
 *
 * <pre>{@code
 *     String filename = "tiny-yolo-voc.h5";
 *     ComputationGraph graph = KerasModelImport.importKerasModelAndWeights(filename, false);
 *     INDArray priors = Nd4j.create(priorBoxes);
 *
 *     FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
 *             .seed(seed)
 *             .iterations(iterations)
 *             .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
 *             .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
 *             .gradientNormalizationThreshold(1.0)
 *             .updater(new Adam.Builder().learningRate(1e-3).build())
 *             .l2(0.00001)
 *             .activation(Activation.IDENTITY)
 *             .trainingWorkspaceMode(workspaceMode)
 *             .inferenceWorkspaceMode(workspaceMode)
 *             .build();
 *
 *     ComputationGraph model = new TransferLearning.GraphBuilder(graph)
 *             .fineTuneConfiguration(fineTuneConf)
 *             .addLayer("outputs",
 *                     new Yolo2OutputLayer.Builder()
 *                             .boundingBoxPriors(priors)
 *                             .build(),
 *                     "conv2d_9")
 *             .setOutputs("outputs")
 *             .build();
 *
 *     System.out.println(model.summary(InputType.convolutional(416, 416, 3)));
 *
 *     ModelSerializer.writeModel(model, "tiny-yolo-voc_dl4j_inference.v1.zip", false);
 *}</pre>
 *
 * The channels of the 416x416 input images need to be in RGB order (not BGR), with values normalized within [0, 1].
 *
 * @author saudet
 */
@AllArgsConstructor
@Builder
public class TinyYOLO extends ZooModel {

    @Builder.Default @Getter private int nBoxes = 5;
    @Builder.Default @Getter private double[][] priorBoxes = {{1.08, 1.19}, {3.42, 4.41}, {6.63, 11.38}, {9.42, 5.11}, {16.62, 10.52}};

    @Builder.Default private long seed = 1234;
    @Builder.Default private int[] inputShape = {3, 416, 416};
    @Builder.Default private int numClasses = 0;
    @Builder.Default private IUpdater updater = new Adam(1e-3);
    @Builder.Default private CacheMode cacheMode = CacheMode.NONE;
    @Builder.Default private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
    @Builder.Default private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

    private TinyYOLO() {}

    @Override
    public String pretrainedUrl(PretrainedType pretrainedType) {
        if (pretrainedType == PretrainedType.IMAGENET)
            return DL4JResources.getURLString("models/tiny-yolo-voc_dl4j_inference.v2.zip");
        else
            return null;
    }

    @Override
    public long pretrainedChecksum(PretrainedType pretrainedType) {
        if (pretrainedType == PretrainedType.IMAGENET)
            return 1256226465L;
        else
            return 0L;
    }

    @Override
    public Class<? extends Model> modelType() {
        return ComputationGraph.class;
    }

    public ComputationGraphConfiguration conf() {
        INDArray priors = Nd4j.create(priorBoxes);

        GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(1.0)
                .updater(updater)
                .l2(0.00001)
                .activation(Activation.IDENTITY)
                .cacheMode(cacheMode)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .cudnnAlgoMode(cudnnAlgoMode)
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.convolutional(inputShape[2], inputShape[1], inputShape[0]));

        addLayers(graphBuilder, 1, 3, inputShape[0], 16, 2, 2);

        addLayers(graphBuilder, 2, 3, 16, 32, 2, 2);

        addLayers(graphBuilder, 3, 3, 32, 64, 2, 2);

        addLayers(graphBuilder, 4, 3, 64, 128, 2, 2);

        addLayers(graphBuilder, 5, 3, 128, 256, 2, 2);

        addLayers(graphBuilder, 6, 3, 256, 512, 2, 1);

        addLayers(graphBuilder, 7, 3, 512, 1024, 0, 0);
        addLayers(graphBuilder, 8, 3, 1024, 1024, 0, 0);

        int layerNumber = 9;
        graphBuilder
                .addLayer("convolution2d_" + layerNumber,
                        new ConvolutionLayer.Builder(1,1)
                                .nIn(1024)
                                .nOut(nBoxes * (5 + numClasses))
                                .weightInit(WeightInit.XAVIER)
                                .stride(1,1)
                                .convolutionMode(ConvolutionMode.Same)
                                .weightInit(WeightInit.RELU)
                                .activation(Activation.IDENTITY)
                                .build(),
                        "activation_" + (layerNumber - 1))
                .addLayer("outputs",
                        new Yolo2OutputLayer.Builder()
                                .boundingBoxPriors(priors)
                                .build(),
                        "convolution2d_" + layerNumber)
                .setOutputs("outputs");

        return graphBuilder.build();
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

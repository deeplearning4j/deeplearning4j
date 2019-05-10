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

package org.deeplearning4j.zoo.model.helper;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.ScaleVertex;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.nd4j.linalg.activations.Activation;

/**
 * Inception is based on GoogleLeNet configuration of convolutional layers for optimization of
 * resources and learning. You can use this module to add Inception to your own custom models.
 * <br>
 * The GoogleLeNet paper: <a href="https://arxiv.org/abs/1409.4842">https://arxiv.org/abs/1409.4842</a>
 * <br>
 * This module is based on the Inception-ResNet paper that combined residual shortcuts with
 * Inception-style networks: <a href="https://arxiv.org/abs/1602.07261">https://arxiv.org/abs/1602.07261</a>
 *
 * Revised and consolidated. Likely needs further tuning for specific applications.
 *
 * @author Justin Long (crockpotveggies)
 */
public class InceptionResNetHelper {

    public static String nameLayer(String blockName, String layerName, int i) {
        return blockName + "-" + layerName + "-" + i;
    }

    /**
     * Append Inception-ResNet A to a computation graph.
     * @param graph
     * @param blockName
     * @param scale
     * @param activationScale
     * @param input
     * @return
     */
    public static ComputationGraphConfiguration.GraphBuilder inceptionV1ResA(
                    ComputationGraphConfiguration.GraphBuilder graph, String blockName, int scale,
                    double activationScale, String input) {
        //        // first add the RELU activation layer
        //        graph.addLayer(nameLayer(blockName,"activation1",0), new ActivationLayer.Builder().activation(Activation.TANH).build(), input);

        // loop and add each subsequent resnet blocks
        String previousBlock = input;
        for (int i = 1; i <= scale; i++) {
            graph
                            // 1x1
                            .addLayer(nameLayer(blockName, "cnn1", i),
                                            new ConvolutionLayer.Builder(new int[] {1, 1})
                                                            .convolutionMode(ConvolutionMode.Same).nIn(192).nOut(32)
                                                            .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                                                            .build(),
                                            previousBlock)
                            .addLayer(nameLayer(blockName, "batch1", i),
                                            new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(32)
                                                            .nOut(32).build(),
                                            nameLayer(blockName, "cnn1", i))
                            // 1x1 -> 3x3
                            .addLayer(nameLayer(blockName, "cnn2", i),
                                            new ConvolutionLayer.Builder(new int[] {1, 1})
                                                            .convolutionMode(ConvolutionMode.Same).nIn(192).nOut(32)
                                                            .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                                                            .build(),
                                            previousBlock)
                            .addLayer(nameLayer(blockName, "batch2", i),
                                            new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(32)
                                                            .nOut(32).build(),
                                            nameLayer(blockName, "cnn2", i))
                            .addLayer(nameLayer(blockName, "cnn3", i),
                                            new ConvolutionLayer.Builder(new int[] {3, 3})
                                                            .convolutionMode(ConvolutionMode.Same).nIn(32).nOut(32)
                                                            .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                                                            .build(),
                                            nameLayer(blockName, "batch2", i))
                            .addLayer(nameLayer(blockName, "batch3", i),
                                            new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(32)
                                                            .nOut(32).build(),
                                            nameLayer(blockName, "cnn3", i))
                            // 1x1 -> 3x3 -> 3x3
                            .addLayer(nameLayer(blockName, "cnn4", i),
                                            new ConvolutionLayer.Builder(new int[] {1, 1})
                                                            .convolutionMode(ConvolutionMode.Same).nIn(192).nOut(32)
                                                            .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                                                            .build(),
                                            previousBlock)
                            .addLayer(nameLayer(blockName, "batch4", i),
                                            new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(32)
                                                            .nOut(32).build(),
                                            nameLayer(blockName, "cnn4", i))
                            .addLayer(nameLayer(blockName, "cnn5", i),
                                            new ConvolutionLayer.Builder(new int[] {3, 3})
                                                            .convolutionMode(ConvolutionMode.Same).nIn(32).nOut(32)
                                                            .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                                                            .build(),
                                            nameLayer(blockName, "batch4", i))
                            .addLayer(nameLayer(blockName, "batch5", i),
                                            new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(32)
                                                            .nOut(32).build(),
                                            nameLayer(blockName, "cnn5", i))
                            .addLayer(nameLayer(blockName, "cnn6", i),
                                            new ConvolutionLayer.Builder(new int[] {3, 3})
                                                            .convolutionMode(ConvolutionMode.Same).nIn(32).nOut(32)
                                                            .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                                                            .build(),
                                            nameLayer(blockName, "batch5", i))
                            .addLayer(nameLayer(blockName, "batch6", i),
                                            new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(32)
                                                            .nOut(32).build(),
                                            nameLayer(blockName, "cnn6", i))
                            // --> 1x1 --> scaling -->
                            .addVertex(nameLayer(blockName, "merge1", i), new MergeVertex(),
                                            nameLayer(blockName, "batch1", i), nameLayer(blockName, "batch3", i),
                                            nameLayer(blockName, "batch6", i))
                            .addLayer(nameLayer(blockName, "cnn7", i),
                                            new ConvolutionLayer.Builder(new int[] {3, 3})
                                                            .convolutionMode(ConvolutionMode.Same).nIn(96).nOut(192)
                                                            .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                                                            .build(),
                                            nameLayer(blockName, "merge1", i))
                            .addLayer(nameLayer(blockName, "batch7", i),
                                            new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(192)
                                                            .nOut(192).build(),
                                            nameLayer(blockName, "cnn7", i))
                            .addVertex(nameLayer(blockName, "scaling", i), new ScaleVertex(activationScale),
                                            nameLayer(blockName, "batch7", i))
                            // -->
                            .addLayer(nameLayer(blockName, "shortcut-identity", i),
                                            new ActivationLayer.Builder().activation(Activation.IDENTITY).build(),
                                            previousBlock)
                            .addVertex(nameLayer(blockName, "shortcut", i),
                                            new ElementWiseVertex(ElementWiseVertex.Op.Add),
                                            nameLayer(blockName, "scaling", i),
                                            nameLayer(blockName, "shortcut-identity", i));

            // leave the last vertex as the block name for convenience
            if (i == scale)
                graph.addLayer(blockName, new ActivationLayer.Builder().activation(Activation.TANH).build(),
                                nameLayer(blockName, "shortcut", i));
            else
                graph.addLayer(nameLayer(blockName, "activation", i),
                                new ActivationLayer.Builder().activation(Activation.TANH).build(),
                                nameLayer(blockName, "shortcut", i));

            previousBlock = nameLayer(blockName, "activation", i);
        }
        return graph;
    }

    /**
     * Append Inception-ResNet B to a computation graph.
     * @param graph
     * @param blockName
     * @param scale
     * @param activationScale
     * @param input
     * @return
     */
    public static ComputationGraphConfiguration.GraphBuilder inceptionV1ResB(
                    ComputationGraphConfiguration.GraphBuilder graph, String blockName, int scale,
                    double activationScale, String input) {
        // first add the RELU activation layer
        graph.addLayer(nameLayer(blockName, "activation1", 0),
                        new ActivationLayer.Builder().activation(Activation.TANH).build(), input);

        // loop and add each subsequent resnet blocks
        String previousBlock = nameLayer(blockName, "activation1", 0);
        for (int i = 1; i <= scale; i++) {
            graph
                            // 1x1
                            .addLayer(nameLayer(blockName, "cnn1", i),
                                            new ConvolutionLayer.Builder(new int[] {1, 1})
                                                            .convolutionMode(ConvolutionMode.Same).nIn(576).nOut(128)
                                                            .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                                                            .build(),
                                            previousBlock)
                            .addLayer(nameLayer(blockName, "batch1", i),
                                            new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(128)
                                                            .nOut(128).build(),
                                            nameLayer(blockName, "cnn1", i))
                            // 1x1 -> 3x3 -> 3x3
                            .addLayer(nameLayer(blockName, "cnn2", i),
                                            new ConvolutionLayer.Builder(new int[] {1, 1})
                                                            .convolutionMode(ConvolutionMode.Same).nIn(576).nOut(128)
                                                            .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                                                            .build(),
                                            previousBlock)
                            .addLayer(nameLayer(blockName, "batch2", i),
                                            new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(128)
                                                            .nOut(128).build(),
                                            nameLayer(blockName, "cnn2", i))
                            .addLayer(nameLayer(blockName, "cnn3", i),
                                            new ConvolutionLayer.Builder(new int[] {1, 3})
                                                            .convolutionMode(ConvolutionMode.Same).nIn(128).nOut(128)
                                                            .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                                                            .build(),
                                            nameLayer(blockName, "batch2", i))
                            .addLayer(nameLayer(blockName, "batch3", i),
                                            new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(128)
                                                            .nOut(128).build(),
                                            nameLayer(blockName, "cnn3", i))
                            .addLayer(nameLayer(blockName, "cnn4", i),
                                            new ConvolutionLayer.Builder(new int[] {3, 1})
                                                            .convolutionMode(ConvolutionMode.Same).nIn(128).nOut(128)
                                                            .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                                                            .build(),
                                            nameLayer(blockName, "batch3", i))
                            .addLayer(nameLayer(blockName, "batch4", i),
                                            new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(128)
                                                            .nOut(128).build(),
                                            nameLayer(blockName, "cnn4", i))
                            // --> 1x1 --> scaling -->
                            .addVertex(nameLayer(blockName, "merge1", i), new MergeVertex(),
                                            nameLayer(blockName, "batch1", i), nameLayer(blockName, "batch4", i))
                            .addLayer(nameLayer(blockName, "cnn5", i),
                                            new ConvolutionLayer.Builder(new int[] {1, 1})
                                                            .convolutionMode(ConvolutionMode.Same).nIn(256).nOut(576)
                                                            .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                                                            .build(),
                                            nameLayer(blockName, "merge1", i))
                            .addLayer(nameLayer(blockName, "batch5", i),
                                            new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(576)
                                                            .nOut(576).build(),
                                            nameLayer(blockName, "cnn5", i))
                            .addVertex(nameLayer(blockName, "scaling", i), new ScaleVertex(activationScale),
                                            nameLayer(blockName, "batch5", i))
                            // -->
                            .addLayer(nameLayer(blockName, "shortcut-identity", i),
                                            new ActivationLayer.Builder().activation(Activation.IDENTITY).build(),
                                            previousBlock)
                            .addVertex(nameLayer(blockName, "shortcut", i),
                                            new ElementWiseVertex(ElementWiseVertex.Op.Add),
                                            nameLayer(blockName, "scaling", i),
                                            nameLayer(blockName, "shortcut-identity", i));

            // leave the last vertex as the block name for convenience
            if (i == scale)
                graph.addLayer(blockName, new ActivationLayer.Builder().activation(Activation.TANH).build(),
                                nameLayer(blockName, "shortcut", i));
            else
                graph.addLayer(nameLayer(blockName, "activation", i),
                                new ActivationLayer.Builder().activation(Activation.TANH).build(),
                                nameLayer(blockName, "shortcut", i));

            previousBlock = nameLayer(blockName, "activation", i);
        }
        return graph;
    }

    /**
     * Append Inception-ResNet C to a computation graph.
     * @param graph
     * @param blockName
     * @param scale
     * @param activationScale
     * @param input
     * @return
     */
    public static ComputationGraphConfiguration.GraphBuilder inceptionV1ResC(
                    ComputationGraphConfiguration.GraphBuilder graph, String blockName, int scale,
                    double activationScale, String input) {
        // loop and add each subsequent resnet blocks
        String previousBlock = input;
        for (int i = 1; i <= scale; i++) {
            graph
                            // 1x1
                            .addLayer(nameLayer(blockName, "cnn1", i),
                                            new ConvolutionLayer.Builder(new int[] {1, 1})
                                                            .convolutionMode(ConvolutionMode.Same).nIn(1344).nOut(192)
                                                            .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                                                            .build(),
                                            previousBlock)
                            .addLayer(nameLayer(blockName, "batch1", i),
                                            new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(192)
                                                            .nOut(192).build(),
                                            nameLayer(blockName, "cnn1", i))
                            // 1x1 -> 1x3 -> 3x1
                            .addLayer(nameLayer(blockName, "cnn2", i),
                                            new ConvolutionLayer.Builder(new int[] {1, 1})
                                                            .convolutionMode(ConvolutionMode.Same).nIn(1344).nOut(192)
                                                            .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                                                            .build(),
                                            previousBlock)
                            .addLayer(nameLayer(blockName, "batch2", i),
                                            new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(192)
                                                            .nOut(192).build(),
                                            nameLayer(blockName, "cnn2", i))
                            .addLayer(nameLayer(blockName, "cnn3", i),
                                            new ConvolutionLayer.Builder(new int[] {1, 3})
                                                            .convolutionMode(ConvolutionMode.Same).nIn(192).nOut(192)
                                                            .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                                                            .build(),
                                            nameLayer(blockName, "batch2", i))
                            .addLayer(nameLayer(blockName, "batch3", i),
                                            new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(192)
                                                            .nOut(192).build(),
                                            nameLayer(blockName, "cnn3", i))
                            .addLayer(nameLayer(blockName, "cnn4", i),
                                            new ConvolutionLayer.Builder(new int[] {3, 1})
                                                            .convolutionMode(ConvolutionMode.Same).nIn(192).nOut(192)
                                                            .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                                                            .build(),
                                            nameLayer(blockName, "batch3", i))
                            .addLayer(nameLayer(blockName, "batch4", i),
                                            new BatchNormalization.Builder(false).decay(0.995).eps(0.001)
                                                            .activation(Activation.TANH).nIn(192).nOut(192).build(),
                                            nameLayer(blockName, "cnn4", i))
                            // --> 1x1 --> scale -->
                            .addVertex(nameLayer(blockName, "merge1", i), new MergeVertex(),
                                            nameLayer(blockName, "batch1", i), nameLayer(blockName, "batch4", i))
                            .addLayer(nameLayer(blockName, "cnn5", i),
                                            new ConvolutionLayer.Builder(new int[] {1, 1})
                                                            .convolutionMode(ConvolutionMode.Same).nIn(384).nOut(1344)
                                                            .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                                                            .build(),
                                            nameLayer(blockName, "merge1", i))
                            .addLayer(nameLayer(blockName, "batch5", i),
                                            new BatchNormalization.Builder(false).decay(0.995).eps(0.001)
                                                            .activation(Activation.TANH).nIn(1344).nOut(1344).build(),
                                            nameLayer(blockName, "cnn5", i))
                            .addVertex(nameLayer(blockName, "scaling", i), new ScaleVertex(activationScale),
                                            nameLayer(blockName, "batch5", i))
                            // -->
                            .addLayer(nameLayer(blockName, "shortcut-identity", i),
                                            new ActivationLayer.Builder().activation(Activation.IDENTITY).build(),
                                            previousBlock)
                            .addVertex(nameLayer(blockName, "shortcut", i),
                                            new ElementWiseVertex(ElementWiseVertex.Op.Add),
                                            nameLayer(blockName, "scaling", i),
                                            nameLayer(blockName, "shortcut-identity", i));

            // leave the last vertex as the block name for convenience
            if (i == scale)
                graph.addLayer(blockName, new ActivationLayer.Builder().activation(Activation.TANH).build(),
                                nameLayer(blockName, "shortcut", i));
            else
                graph.addLayer(nameLayer(blockName, "activation", i),
                                new ActivationLayer.Builder().activation(Activation.TANH).build(),
                                nameLayer(blockName, "shortcut", i));

            previousBlock = nameLayer(blockName, "activation", i);
        }
        return graph;
    }

}

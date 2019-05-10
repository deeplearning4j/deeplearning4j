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
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.model.Darknet19;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.deeplearning4j.zoo.model.YOLO2;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationLReLU;

/**
 * Contains functionality shared by {@link Darknet19}, {@link TinyYOLO}, and {@link YOLO2}.
 *
 * @author saudet
 */
public class DarknetHelper {

    /** Returns {@code inputShape[1] / 32}, where {@code inputShape[1]} should be a multiple of 32. */
    public static int getGridWidth(int[] inputShape) {
        return inputShape[1] / 32;
    }

    /** Returns {@code inputShape[2] / 32}, where {@code inputShape[2]} should be a multiple of 32. */
    public static int getGridHeight(int[] inputShape) {
        return inputShape[2] / 32;
    }

    public static ComputationGraphConfiguration.GraphBuilder addLayers(ComputationGraphConfiguration.GraphBuilder graphBuilder, int layerNumber, int filterSize, int nIn, int nOut, int poolSize) {
        return addLayers(graphBuilder, layerNumber, filterSize, nIn, nOut, poolSize, poolSize);
    }

    public static ComputationGraphConfiguration.GraphBuilder addLayers(ComputationGraphConfiguration.GraphBuilder graphBuilder, int layerNumber, int filterSize, int nIn, int nOut, int poolSize, int poolStride) {
        String input = "maxpooling2d_" + (layerNumber - 1);
        if (!graphBuilder.getVertices().containsKey(input)) {
            input = "activation_" + (layerNumber - 1);
        }
        if (!graphBuilder.getVertices().containsKey(input)) {
            input = "concatenate_" + (layerNumber - 1);
        }
        if (!graphBuilder.getVertices().containsKey(input)) {
            input = "input";
        }

        return addLayers(graphBuilder, layerNumber, input, filterSize, nIn, nOut, poolSize, poolStride);
    }

    public static ComputationGraphConfiguration.GraphBuilder addLayers(ComputationGraphConfiguration.GraphBuilder graphBuilder, int layerNumber, String input, int filterSize, int nIn, int nOut, int poolSize, int poolStride) {
        graphBuilder
                .addLayer("convolution2d_" + layerNumber,
                        new ConvolutionLayer.Builder(filterSize,filterSize)
                                .nIn(nIn)
                                .nOut(nOut)
                                .weightInit(WeightInit.XAVIER)
                                .convolutionMode(ConvolutionMode.Same)
                                .hasBias(false)
                                .stride(1,1)
                                .activation(Activation.IDENTITY)
                                .build(),
                        input)
                .addLayer("batchnormalization_" + layerNumber,
                        new BatchNormalization.Builder()
                                .nIn(nOut).nOut(nOut)
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.IDENTITY)
                                .build(),
                        "convolution2d_" + layerNumber)
                .addLayer("activation_" + layerNumber,
                        new ActivationLayer.Builder()
                                .activation(new ActivationLReLU(0.1))
                                .build(),
                        "batchnormalization_" + layerNumber);
        if (poolSize > 0) {
            graphBuilder
                    .addLayer("maxpooling2d_" + layerNumber,
                            new SubsamplingLayer.Builder()
                                    .kernelSize(poolSize, poolSize)
                                    .stride(poolStride, poolStride)
                                    .convolutionMode(ConvolutionMode.Same)
                                    .build(),
                            "activation_" + layerNumber);
        }

        return graphBuilder;
    }

}

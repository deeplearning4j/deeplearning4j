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

package org.deeplearning4j.nn.conf.layers;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.params.EmptyParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * Global pooling layer - used to do pooling over time for RNNs, and 2d pooling for CNNs.<br>
 * Supports the following {@link PoolingType}s: SUM, AVG, MAX, PNORM<br>
 *
 * Global pooling layer can also handle mask arrays when dealing with variable length inputs. Mask arrays are assumed
 * to be 2d, and are fed forward through the network during training or post-training forward pass:<br>
 * - Time series: mask arrays are shape [miniBatchSize, maxTimeSeriesLength] and contain values 0 or 1 only<br>
 * - CNNs: mask have shape [miniBatchSize, height] or [miniBatchSize, width]. Important: the current implementation assumes
 *   that for CNNs + variable length (masking), the input shape is [miniBatchSize, channels, height, 1] or
 *   [miniBatchSize, channels, 1, width] respectively. This is the case with global pooling in architectures like CNN for
 *   sentence classification.<br>
 * <p>
 *
 * Behaviour with default settings:<br>
 * - 3d (time series) input with shape [miniBatchSize, vectorSize, timeSeriesLength] -> 2d output [miniBatchSize, vectorSize]<br>
 * - 4d (CNN) input with shape [miniBatchSize, channels, height, width] -> 2d output [miniBatchSize, channels]<br>
 * - 5d (CNN3D) input with shape [miniBatchSize, channels, depth, height, width] -> 2d output [miniBatchSize, channels]<br>
 *
 * <p>
 * Alternatively, by setting collapseDimensions = false in the configuration, it is possible to retain the reduced dimensions
 * as 1s: this gives
 * - [miniBatchSize, vectorSize, 1] for RNN output,
 * - [miniBatchSize, channels, 1, 1] for CNN output, and
 * - [miniBatchSize, channels, 1, 1, 1] for CNN3D output.
 * <br>
 *
 * @author Alex Black
 */
@Data
@NoArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class GlobalPoolingLayer extends NoParamLayer {

    private PoolingType poolingType;
    private int[] poolingDimensions;
    private int pnorm;
    private boolean collapseDimensions = true;

    private GlobalPoolingLayer(Builder builder) {
        super(builder);
        this.poolingType = builder.poolingType;
        this.poolingDimensions = builder.poolingDimensions;
        this.collapseDimensions = builder.collapseDimensions;
        this.pnorm = builder.pnorm;
        this.layerName = builder.layerName;
    }

    public GlobalPoolingLayer(PoolingType poolingType){
        this(new GlobalPoolingLayer.Builder().poolingType(poolingType));
    }


    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf,
                    Collection<TrainingListener> trainingListeners, int layerIndex, INDArray layerParamsView,
                    boolean initializeParams) {
        org.deeplearning4j.nn.layers.pooling.GlobalPoolingLayer ret =
                        new org.deeplearning4j.nn.layers.pooling.GlobalPoolingLayer(conf);
        ret.setListeners(trainingListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return EmptyParamInitializer.getInstance();
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {

        switch (inputType.getType()) {
            case FF:
                throw new UnsupportedOperationException(
                                "Global max pooling cannot be applied to feed-forward input type. Got input type = "
                                                + inputType);
            case RNN:
                InputType.InputTypeRecurrent recurrent = (InputType.InputTypeRecurrent) inputType;
                if (collapseDimensions) {
                    //Return 2d (feed-forward) activations
                    return InputType.feedForward(recurrent.getSize());
                } else {
                    //Return 3d activations, with shape [minibatch, timeStepSize, 1]
                    return recurrent;
                }
            case CNN:
                InputType.InputTypeConvolutional conv = (InputType.InputTypeConvolutional) inputType;
                if (collapseDimensions) {
                    return InputType.feedForward(conv.getChannels());
                } else {
                    return InputType.convolutional(1, 1, conv.getChannels());
                }
            case CNN3D:
                InputType.InputTypeConvolutional3D conv3d = (InputType.InputTypeConvolutional3D) inputType;
                if (collapseDimensions) {
                    return InputType.feedForward(conv3d.getChannels());
                } else {
                    return InputType.convolutional3D(1,1,1, conv3d.getChannels());
                }
            case CNNFlat:
                InputType.InputTypeConvolutionalFlat convFlat = (InputType.InputTypeConvolutionalFlat) inputType;
                if (collapseDimensions) {
                    return InputType.feedForward(convFlat.getDepth());
                } else {
                    return InputType.convolutional(1, 1, convFlat.getDepth());
                }
            default:
                throw new UnsupportedOperationException("Unknown or not supported input type: " + inputType);
        }
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        //Not applicable
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {

        switch (inputType.getType()) {
            case FF:
                throw new UnsupportedOperationException(
                                "Global max pooling cannot be applied to feed-forward input type. Got input type = "
                                                + inputType);
            case RNN:
            case CNN:
            case CNN3D:
                //No preprocessor required
                return null;
            case CNNFlat:
                InputType.InputTypeConvolutionalFlat cFlat = (InputType.InputTypeConvolutionalFlat) inputType;
                return new FeedForwardToCnnPreProcessor(cFlat.getHeight(), cFlat.getWidth(), cFlat.getDepth());
        }

        return null;
    }

    @Override
    public double getL1ByParam(String paramName) {
        //Not applicable
        return 0;
    }

    @Override
    public double getL2ByParam(String paramName) {
        //Not applicable
        return 0;
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        throw new UnsupportedOperationException("Global pooling layer does not contain parameters");
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        InputType outputType = getOutputType(-1, inputType);

        long fwdTrainInferenceWorkingPerEx = 0;
        //Here: we'll assume we are doing 'full array' global pooling.
        //For max/avg/sum pooling, no working memory (GlobalPoolingLayer.activateHelperFullArray
        //But for pnorm, we have working memory
        if (poolingType == PoolingType.PNORM) {
            //Dup the input array once before
            fwdTrainInferenceWorkingPerEx = inputType.arrayElementsPerExample();
        }

        return new LayerMemoryReport.Builder(layerName, GlobalPoolingLayer.class, inputType, outputType)
                        .standardMemory(0, 0) //No params
                        //Train + Inference: no additional working memory (except pnorm) - the reduction is the output activations
                        .workingMemory(0, fwdTrainInferenceWorkingPerEx, 0, fwdTrainInferenceWorkingPerEx)
                        .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching
                        .build();
    }

    public static class Builder extends Layer.Builder<Builder> {

        private PoolingType poolingType = PoolingType.MAX;
        private int[] poolingDimensions;
        private int pnorm = 2;
        private boolean collapseDimensions = true;

        public Builder() {

        }

        public Builder(PoolingType poolingType) {
            this.poolingType = poolingType;
        }

        /**
         * Pooling dimensions. Note: most of the time, this doesn't need to be set, and the defaults can be used.
         * Default for RNN data: pooling dimension 2 (time).
         * Default for CNN data: pooling dimensions 2,3 (height and width)
         * Default for CNN3D data: pooling dimensions 2,3,4 (depth, height and width)

         * @param poolingDimensions Pooling dimensions to use
         */
        public Builder poolingDimensions(int... poolingDimensions) {
            this.poolingDimensions = poolingDimensions;
            return this;
        }

        /**
         * @param poolingType Pooling type for global pooling
         */
        public Builder poolingType(PoolingType poolingType) {
            this.poolingType = poolingType;
            return this;
        }

        /**
         * Whether to collapse dimensions when pooling or not. Usually you *do* want to do this. Default: true.
         * If true:<br>
         * - 3d (time series) input with shape [miniBatchSize, vectorSize, timeSeriesLength] -> 2d output [miniBatchSize, vectorSize]<br>
         * - 4d (CNN) input with shape [miniBatchSize, channels, height, width] -> 2d output [miniBatchSize, channels]<br>
         * - 5d (CNN3D) input with shape [miniBatchSize, channels, depth, height, width] -> 2d output [miniBatchSize, channels]<br>

         *
         * If false:<br>
         * - 3d (time series) input with shape [miniBatchSize, vectorSize, timeSeriesLength] -> 3d output [miniBatchSize, vectorSize, 1]<br>
         * - 4d (CNN) input with shape [miniBatchSize, channels, height, width] -> 2d output [miniBatchSize, channels, 1, 1]<br>
         * - 5d (CNN3D) input with shape [miniBatchSize, channels, depth, height, width] -> 2d output [miniBatchSize, channels, 1, 1, 1]<br>
         *
         * @param collapseDimensions Whether to collapse the dimensions or not
         */
        public Builder collapseDimensions(boolean collapseDimensions) {
            this.collapseDimensions = collapseDimensions;
            return this;
        }

        /**
         * P-norm constant. Only used if using {@link PoolingType#PNORM} for the pooling type
         *
         * @param pnorm P-norm constant
         */
        public Builder pnorm(int pnorm) {
            if (pnorm <= 0)
                throw new IllegalArgumentException("Invalid input: p-norm value must be greater than 0. Got: " + pnorm);
            this.pnorm = pnorm;
            return this;
        }

        @SuppressWarnings("unchecked")
        public GlobalPoolingLayer build() {
            return new GlobalPoolingLayer(this);
        }
    }
}

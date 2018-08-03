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

package org.deeplearning4j.nn.conf.inputs;

import lombok.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonIgnore;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonSubTypes;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;
import java.util.Arrays;

/**
 * The InputType class is used to track and define the types of activations etc used in a ComputationGraph.
 * This is most useful for automatically adding preprocessors between layers, and automatically setting nIn values.
 * See: {@link org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder#setInputTypes(InputType...)} and
 * {@link org.deeplearning4j.nn.conf.ComputationGraphConfiguration#addPreProcessors(InputType...)}
 *
 * @author Alex Black
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
@JsonSubTypes(value = {@JsonSubTypes.Type(value = InputType.InputTypeFeedForward.class, name = "FeedForward"),
        @JsonSubTypes.Type(value = InputType.InputTypeRecurrent.class, name = "Recurrent"),
        @JsonSubTypes.Type(value = InputType.InputTypeConvolutional.class, name = "Convolutional"),
        @JsonSubTypes.Type(value = InputType.InputTypeConvolutionalFlat.class, name = "ConvolutionalFlat"),
        @JsonSubTypes.Type(value = InputType.InputTypeConvolutional3D.class, name = "Convolutional3D")})
public abstract class InputType implements Serializable {

    /**
     * The type of activations in/out of a given GraphVertex<br>
     * FF: Standard feed-foward (2d minibatch, 1d per example) data<br>
     * RNN: Recurrent neural network (3d minibatch) time series data<br>
     * CNN: 2D Convolutional neural network (4d minibatch, [miniBatchSize, channels, height, width])
     * CNNFlat: Flattened 2D conv net data (2d minibatch, [miniBatchSize, height * width * channels])
     * CNN3D: 3D convolutional neural network (5d minibatch, [miniBatchSize, channels, height, width, channels])
     */
    public enum Type {
        FF, RNN, CNN, CNNFlat, CNN3D
    }


    @JsonIgnore
    public abstract Type getType();

    @Override
    public abstract String toString();

    @JsonIgnore
    public abstract long arrayElementsPerExample();

    /**
     * Returns the shape of this InputType
     *
     * @param includeBatchDim Whether to include minibatch in the return shape array
     * @return int[]
     */
    @JsonIgnore
    public abstract long[] getShape(boolean includeBatchDim);

    /**
     * Returns the shape of this InputType without minibatch dimension in the returned array
     *
     * @return int[]
     */
    public long[] getShape() {
        return getShape(false);
    }

    /**
     * InputType for feed forward network data
     *
     * @param size The size of the activations
     * @return InputTypeFeedForward
     */
    public static InputType feedForward(long size) {
        return new InputTypeFeedForward(size);
    }

    /**
     * InputType for recurrent neural network (time series) data
     *
     * @param size The size of the activations
     * @return InputTypeRecurrent
     */
    public static InputType recurrent(long size) {
        return new InputTypeRecurrent(size);
    }

    /**
     * InputType for recurrent neural network (time series) data
     *
     * @param size             The size of the activations
     * @param timeSeriesLength Length of the input time series
     * @return InputTypeRecurrent
     */
    public static InputType recurrent(long size, long timeSeriesLength) {
        return new InputTypeRecurrent(size, timeSeriesLength);
    }

    /**
     * Input type for convolutional (CNN) data, that is 4d with shape [miniBatchSize, channels, height, width].
     * For CNN data that has been flattened, use {@link #convolutionalFlat(long, long, long)}
     *
     * @param height height of the input
     * @param width  Width of the input
     * @param depth  Depth, or number of channels
     * @return InputTypeConvolutional
     */
    public static InputType convolutional(long height, long width, long depth) {
        return new InputTypeConvolutional(height, width, depth);
    }

    /**
     * Input type for 3D convolutional (CNN3D) data, that is 5d with shape
     * [miniBatchSize, channels, height, width, channels].
     *
     * @param height   height of the input
     * @param width    Width of the input
     * @param depth    Depth of the input
     * @param channels Number of channels of the input
     * @return InputTypeConvolutional3D
     */
    public static InputType convolutional3D(long depth, long height, long width,  long channels) {
        return new InputTypeConvolutional3D(depth, height, width, channels);
    }

    /**
     * Input type for convolutional (CNN) data, where the data is in flattened (row vector) format.
     * Expect data with shape [miniBatchSize, height * width * channels]. For CNN data in 4d format,
     * use {@link #convolutional(long, long, long)}
     *
     * @param height Height of the (unflattened) data represented by this input type
     * @param width  Width of the (unflattened) data represented by this input type
     * @param depth  Depth of the (unflattened) data represented by this input type
     * @return InputTypeConvolutionalFlat
     */
    public static InputType convolutionalFlat(long height, long width, long depth) {
        return new InputTypeConvolutionalFlat(height, width, depth);
    }


    @AllArgsConstructor
    @Getter
    @NoArgsConstructor
    @EqualsAndHashCode(callSuper = false)
    public static class InputTypeFeedForward extends InputType {
        private long size;

        @Override
        public Type getType() {
            return Type.FF;
        }

        @Override
        public String toString() {
            return "InputTypeFeedForward(" + size + ")";
        }

        @Override
        public long arrayElementsPerExample() {
            return size;
        }

        @Override
        public long[] getShape(boolean includeBatchDim) {
            if(includeBatchDim) return new long[]{-1, size};
            else return new long[]{size};
        }
    }

    @Getter
    @NoArgsConstructor
    @AllArgsConstructor
    @EqualsAndHashCode(callSuper = false)
    public static class InputTypeRecurrent extends InputType {
        private long size;
        private long timeSeriesLength;

        public InputTypeRecurrent(long size) {
            this(size, -1);
        }

        @Override
        public Type getType() {
            return Type.RNN;
        }

        @Override
        public String toString() {
            if (timeSeriesLength > 0) {
                return "InputTypeRecurrent(" + size + ",timeSeriesLength=" + timeSeriesLength + ")";
            } else {
                return "InputTypeRecurrent(" + size + ")";
            }
        }

        @Override
        public long arrayElementsPerExample() {
            if (timeSeriesLength <= 0) {
                throw new IllegalStateException("Cannot calculate number of array elements per example: "
                        + "time series length is not set. Use InputType.recurrent(int size, int timeSeriesLength) instead?");
            }
            return timeSeriesLength * size;
        }

        @Override
        public long[] getShape(boolean includeBatchDim) {
            if(includeBatchDim) return new long[]{-1, size, timeSeriesLength};
            else return new long[]{size, timeSeriesLength};
        }
    }

    @AllArgsConstructor
    @Data
    @EqualsAndHashCode(callSuper = false)
    @NoArgsConstructor
    public static class InputTypeConvolutional extends InputType {
        private long height;
        private long width;
        private long channels;


        /**
         * Return the number of channels / depth for this 2D convolution. This method has been deprecated,
         * for consistency purposes, use getChannels() instead.
         *
         * @return number of channels, i.e. depth for 2D convolutions
         */
        @Deprecated
        public long getDepth() {
            return channels;
        }

        /**
         * Set the number of channels / depth for this 2D convolution. This method has been deprecated,
         * for consistency purposes, use setChannels(channels) instead.
         *
         **/
        @Deprecated
        public void setDepth(long depth) {
            this.channels = depth;
        }

        @Override
        public Type getType() {
            return Type.CNN;
        }

        @Override
        public String toString() {
            return "InputTypeConvolutional(h=" + height + ",w=" + width + ",c=" + channels + ")";
        }

        @Override
        public long arrayElementsPerExample() {
            return height * width * channels;
        }

        @Override
        public long[] getShape(boolean includeBatchDim) {
            if(includeBatchDim) return new long[]{-1, channels, height, width};
            else return new long[]{channels, height, width};
        }
    }

    @AllArgsConstructor
    @Data
    @EqualsAndHashCode(callSuper = false)
    @NoArgsConstructor
    public static class InputTypeConvolutional3D extends InputType {
        private long depth;
        private long height;
        private long width;
        private long channels;

        @Override
        public Type getType() {
            return Type.CNN3D;
        }

        @Override
        public String toString() {
            return "InputTypeConvolutional3D(d=" + depth + ",h=" + height + ",w=" + width + ",c=" + channels + ")";
        }

        @Override
        public long arrayElementsPerExample() {
            return height * width * depth * channels;
        }

        @Override
        public long[] getShape(boolean includeBatchDim) {
            if(includeBatchDim) return new long[]{-1, channels, depth, height, width};
            else return new long[]{channels, depth, height, width};
        }
    }

    @AllArgsConstructor
    @Data
    @EqualsAndHashCode(callSuper = false)
    @NoArgsConstructor
    public static class InputTypeConvolutionalFlat extends InputType {
        private long height;
        private long width;
        private long depth;

        @Override
        public Type getType() {
            return Type.CNNFlat;
        }

        public long getFlattenedSize() {
            return height * width * depth;
        }

        public InputType getUnflattenedType() {
            return InputType.convolutional(height, width, depth);
        }

        @Override
        public String toString() {
            return "InputTypeConvolutionalFlat(h=" + height + ",w=" + width + ",d=" + depth + ")";
        }

        @Override
        public long arrayElementsPerExample() {
            return height * width * depth;
        }

        @Override
        public long[] getShape(boolean includeBatchDim) {
            if(includeBatchDim) return new long[]{-1, depth, height, width};
            else return new long[]{depth, height, width};
        }
    }


    public static InputType inferInputType(INDArray inputArray) {
        //Note: ConvolutionalFlat and FeedForward look identical... but either should work OK if using something
        // like FeedForwardToCnnPreProcessor

        // FIXME: int cast
        switch (inputArray.rank()) {
            case 2:
                return InputType.feedForward((int) inputArray.size(1));
            case 3:
                return InputType.recurrent((int) inputArray.size(1), (int) inputArray.size(2));
            case 4:
                //Order: [minibatch, channels, height, width] -> [h, w, c]
                return InputType.convolutional((int) inputArray.size(2), (int) inputArray.size(3), (int) inputArray.size(1));
            case 5:
                //Order: [minibatch, channels, depth, height, width] -> [d, h, w, c]
                return InputType.convolutional3D((int) inputArray.size(2), (int) inputArray.size(3),
                        (int) inputArray.size(4), (int) inputArray.size(1));
            default:
                throw new IllegalArgumentException(
                        "Cannot infer input type for array with shape: " + Arrays.toString(inputArray.shape()));
        }
    }

    public static InputType[] inferInputTypes(INDArray... inputArrays) {
        InputType[] out = new InputType[inputArrays.length];
        for (int i = 0; i < inputArrays.length; i++) {
            out[i] = inferInputType(inputArrays[i]);
        }

        return out;
    }

}

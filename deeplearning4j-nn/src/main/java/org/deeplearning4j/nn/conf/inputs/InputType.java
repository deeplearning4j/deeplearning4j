/*-
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.conf.inputs;

import lombok.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonIgnore;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonSubTypes;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;
import java.util.Arrays;

/** The InputType class is used to track and define the types of activations etc used in a ComputationGraph.
 * This is most useful for automatically adding preprocessors between layers, and automatically setting nIn values.
 * See: {@link org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder#setInputTypes(InputType...)} and
 * {@link org.deeplearning4j.nn.conf.ComputationGraphConfiguration#addPreProcessors(InputType...)}
 * @author Alex Black
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
@JsonSubTypes(value = {@JsonSubTypes.Type(value = InputType.InputTypeFeedForward.class, name = "FeedForward"),
                @JsonSubTypes.Type(value = InputType.InputTypeRecurrent.class, name = "Recurrent"),
                @JsonSubTypes.Type(value = InputType.InputTypeConvolutional.class, name = "Convolutional"),
                @JsonSubTypes.Type(value = InputType.InputTypeConvolutionalFlat.class, name = "ConvolutionalFlat")})
public abstract class InputType implements Serializable {

    /** The type of activations in/out of a given GraphVertex<br>
     * FF: Standard feed-foward (2d minibatch, 1d per example) data<br>
     * RNN: Recurrent neural network (3d minibatch) time series data<br>
     * CNN: Convolutional neural n
     */
    public enum Type {
        FF, RNN, CNN, CNNFlat
    }


    @JsonIgnore
    public abstract Type getType();

    @Override
    public abstract String toString();

    @JsonIgnore
    public abstract int arrayElementsPerExample();

    /** InputType for feed forward network data
     * @param size The size of the activations
     */
    public static InputType feedForward(int size) {
        return new InputTypeFeedForward(size);
    }

    /** InputType for recurrent neural network (time series) data
     * @param size The size of the activations
     * @return
     */
    public static InputType recurrent(int size) {
        return new InputTypeRecurrent(size);
    }

    /** InputType for recurrent neural network (time series) data
     * @param size The size of the activations
     * @param timeSeriesLength Length of the input time series
     * @return
     */
    public static InputType recurrent(int size, int timeSeriesLength) {
        return new InputTypeRecurrent(size, timeSeriesLength);
    }

    /**Input type for convolutional (CNN) data, that is 4d with shape [miniBatchSize, depth, height, width].
     * For CNN data that has been flattened, use {@link #convolutionalFlat(int, int, int)}
     * @param height height of the input
     * @param width Width of the input
     * @param depth Depth, or number of channels
     * @return
     */
    public static InputType convolutional(int height, int width, int depth) {
        return new InputTypeConvolutional(height, width, depth);
    }

    /**
     * Input type for convolutional (CNN) data, where the data is in flattened (row vector) format.
     * Expect data with shape [miniBatchSize, height * width * depth]. For CNN data in 4d format, use {@link #convolutional(int, int, int)}
     *
     * @param height    Height of the (unflattened) data represented by this input type
     * @param width     Width of the (unflattened) data represented by this input type
     * @param depth     Depth of the (unflattened) data represented by this input type
     * @return
     */
    public static InputType convolutionalFlat(int height, int width, int depth) {
        return new InputTypeConvolutionalFlat(height, width, depth);
    }


    @AllArgsConstructor
    @Getter
    @NoArgsConstructor
    @EqualsAndHashCode(callSuper = false)
    public static class InputTypeFeedForward extends InputType {
        private int size;

        @Override
        public Type getType() {
            return Type.FF;
        }

        @Override
        public String toString() {
            return "InputTypeFeedForward(" + size + ")";
        }

        @Override
        public int arrayElementsPerExample() {
            return size;
        }
    }

    @Getter
    @NoArgsConstructor
    @AllArgsConstructor
    @EqualsAndHashCode(callSuper = false)
    public static class InputTypeRecurrent extends InputType {
        private int size;
        private int timeSeriesLength;

        public InputTypeRecurrent(int size) {
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
        public int arrayElementsPerExample() {
            if (timeSeriesLength <= 0) {
                throw new IllegalStateException("Cannot calculate number of array elements per example: "
                                + "time series length is not set. Use InputType.recurrent(int size, int timeSeriesLength) instead?");
            }
            return timeSeriesLength * size;
        }
    }

    @AllArgsConstructor
    @Data
    @EqualsAndHashCode(callSuper = false)
    @NoArgsConstructor
    public static class InputTypeConvolutional extends InputType {
        private int height;
        private int width;
        private int depth;

        @Override
        public Type getType() {
            return Type.CNN;
        }

        @Override
        public String toString() {
            return "InputTypeConvolutional(h=" + height + ",w=" + width + ",d=" + depth + ")";
        }

        @Override
        public int arrayElementsPerExample() {
            return height * width * depth;
        }
    }

    @AllArgsConstructor
    @Data
    @EqualsAndHashCode(callSuper = false)
    @NoArgsConstructor
    public static class InputTypeConvolutionalFlat extends InputType {
        private int height;
        private int width;
        private int depth;

        @Override
        public Type getType() {
            return Type.CNNFlat;
        }

        public int getFlattenedSize() {
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
        public int arrayElementsPerExample() {
            return height * width * depth;
        }
    }



    public static InputType inferInputType(INDArray inputArray) {
        //Note: ConvolutionalFlat and FeedForward look identical... but either should work OK if using something
        // like FeedForwardToCnnPreProcessor

        switch (inputArray.rank()) {
            case 2:
                return InputType.feedForward(inputArray.size(1));
            case 3:
                return InputType.recurrent(inputArray.size(1), inputArray.size(2));
            case 4:
                //Order: [minibatch, depth, height, width] -> [h, w, d]
                return InputType.convolutional(inputArray.size(2), inputArray.size(3), inputArray.size(1));
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

/*
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

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.Getter;

import java.io.Serializable;

/** The InputType class is used to track and define the types of activations etc used in a ComputationGraph.
 * This is most useful for automatically adding preprocessors between layers, and automatically setting nIn values.
 * See: {@link org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder#setInputTypes(InputType...)} and
 * {@link org.deeplearning4j.nn.conf.ComputationGraphConfiguration#addPreProcessors(InputType...)}
 * @author Alex Black
 */
public abstract class InputType implements Serializable {

    /** The type of activations in/out of a given GraphVertex<br>
     * FF: Standard feed-foward (2d minibatch, 1d per example) data<br>
     * RNN: Recurrent neural network (3d minibatch) time series data<br>
     * CNN: Convolutional neural n
     */
    public enum Type {FF, RNN, CNN, CNNFlat}


    public abstract Type getType();

    @Override
    public abstract String toString();

    /** InputType for feed forward network data
     * @param size The size of the activations
     */
    public static InputType feedForward(int size){
        return new InputTypeFeedForward(size);
    }

    /** InputType for recurrent neural network (time series) data
     * @param size The size of the activations
     * @return
     */
    public static InputType recurrent(int size){
        return new InputTypeRecurrent(size);
    }

    /**Input type for convolutional (CNN) data, that is 4d with shape [miniBatchSize, depth, height, width].
     * For CNN data that has been flattened, use {@link #convolutionalFlat(int, int, int)}
     * @param height height of the input
     * @param width Width of the input
     * @param depth Depth, or number of channels
     * @return
     */
    public static InputType convolutional(int height, int width, int depth){
        return new InputTypeConvolutional(height,width,depth);
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
    public static InputType convolutionalFlat(int height, int width, int depth){
        return new InputTypeConvolutionalFlat(height, width, depth);
    }


    @AllArgsConstructor @Getter
    public static class InputTypeFeedForward extends InputType{
        private int size;

        @Override
        public Type getType() {
            return Type.FF;
        }

        @Override
        public String toString(){
            return "InputTypeFeedForward(" + size + ")";
        }
    }

    @AllArgsConstructor @Getter
    public static class InputTypeRecurrent extends InputType{
        private int size;

        @Override
        public Type getType() {
            return Type.RNN;
        }

        @Override
        public String toString(){
            return "InputTypeRecurrent(" + size + ")";
        }
    }

    @AllArgsConstructor @Data  @EqualsAndHashCode(callSuper=false)
    public static class InputTypeConvolutional extends InputType {
        private int height;
        private int width;
        private int depth;

        @Override
        public Type getType() {
            return Type.CNN;
        }

        @Override
        public String toString(){
            return "InputTypeConvolutional(h=" + height + ",w=" + width + ",d=" + depth + ")";
        }
    }

    @AllArgsConstructor @Data  @EqualsAndHashCode(callSuper=false)
    public static class InputTypeConvolutionalFlat extends InputType {
        private int height;
        private int width;
        private int depth;

        @Override
        public Type getType() {
            return Type.CNNFlat;
        }

        public int getFlattenedSize(){
            return height * width * depth;
        }

        public InputType getUnflattenedType(){
            return InputType.convolutional(height, width, depth);
        }

        @Override
        public String toString(){
            return "InputTypeConvolutionalFlat(h=" + height + ",w=" + width + ",d=" + depth + ")";
        }
    }


}

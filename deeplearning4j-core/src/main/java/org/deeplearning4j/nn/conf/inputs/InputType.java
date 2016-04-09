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
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.io.Serializable;

/** The InputType class is used to track and define the types of activations etc used in a ComputationGraph.
 * This is most useful for automatically adding preprocessors between layers.
 * See: {@link org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder#setInputTypes(InputType...)} and
 * {@link org.deeplearning4j.nn.conf.ComputationGraphConfiguration#addPreProcessors(InputType...)}
 * @author Alex Black
 */
public abstract class InputType implements Serializable {

    /** The type of activations in/out of a given GraphVertex. */
    public enum Type {FF, RNN, CNN};


    public abstract Type getType();

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

    /**Input type for convolutional (CNN) data
     * @param height height of the input
     * @param width Width of the input
     * @param depth Depth, or number of channels
     * @return
     */
    public static InputType convolutional(int height, int width, int depth){
        return new InputTypeConvolutional(height,width,depth);
    }


    @AllArgsConstructor @Getter
    public static class InputTypeFeedForward extends InputType{
        private int size;

        @Override
        public Type getType() {
            return Type.FF;
        }
    }

    @AllArgsConstructor @Getter
    public static class InputTypeRecurrent extends InputType{
        private int size;

        @Override
        public Type getType() {
            return Type.RNN;
        }
    }

    @AllArgsConstructor @Data
    public static class InputTypeConvolutional extends InputType {
        private int height;
        private int width;
        private int depth;

        @Override
        public Type getType() {
            return Type.CNN;
        }
    }


}

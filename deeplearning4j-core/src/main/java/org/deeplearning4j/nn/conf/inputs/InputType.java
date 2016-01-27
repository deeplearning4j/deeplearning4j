package org.deeplearning4j.nn.conf.inputs;

import lombok.AllArgsConstructor;
import lombok.Data;

/** The InputType class is used to track and define the types of activations etc used in a ComputationGraph.
 * This is most useful for automatically adding preprocessors between layers.
 * @author Alex Black
 */
public abstract class InputType {

    public enum Type {FF, RNN, CNN};

    private static InputType FFInstance = new InputTypeFeedForward();
    private static InputType RNNInstance = new InputTypeRecurrent();


    public abstract Type getType();

    /** InputType for feed forward network inputs */
    public static InputType feedForward(){
        return FFInstance;
    }

    /** InputType for recurrent neural network (time series) data */
    public static InputType recurrent(){
        return RNNInstance;
    }

    /**Input type for convolutional (CNN) data
     * @param depth Depth, or number of channels
     * @param width Width of the input
     * @param height height of the input
     * @return
     */
    public static InputType convolutional(int depth, int width, int height){
        return new InputTypeConvolutional(depth,width,height);
    }


    public static class InputTypeFeedForward extends InputType{
        @Override
        public Type getType() {
            return Type.FF;
        }
    }

    public static class InputTypeRecurrent extends InputType{
        @Override
        public Type getType() {
            return Type.RNN;
        }
    }

    @AllArgsConstructor @Data
    public static class InputTypeConvolutional extends InputType {
        private int depth;
        private int width;
        private int height;

        @Override
        public Type getType() {
            return Type.CNN;
        }
    }


}

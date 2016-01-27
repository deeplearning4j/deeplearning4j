package org.deeplearning4j.nn.conf.inputs;

import lombok.AllArgsConstructor;
import lombok.Data;

public abstract class InputType {

    public enum Type {FF, RNN, CNN};

    private static InputType FFInstance = new InputTypeFeedForward();
    private static InputType RNNInstance = new InputTypeRecurrent();


    public abstract Type getType();

    public static InputType feedForward(){
        return FFInstance;
    }

    public static InputType recurrent(){
        return RNNInstance;
    }

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

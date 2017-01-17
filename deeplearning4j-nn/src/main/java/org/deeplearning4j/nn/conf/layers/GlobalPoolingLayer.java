package org.deeplearning4j.nn.conf.layers;

import lombok.Getter;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.params.EmptyParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;

/**
 * Global pooling layer
 *
 * @author Alex Black
 */
@Getter
public class GlobalPoolingLayer extends Layer {

    private PoolingType poolingType;
    private int[] poolingDimensions;
    private int pnorm;
    private boolean collapseDimensions;

    private GlobalPoolingLayer(Builder builder){
        this.poolingType = builder.poolingType;
        this.poolingDimensions = builder.poolingDimensions;
        this.collapseDimensions = builder.collapseDimensions;
        this.pnorm = builder.pnorm;
    }


    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners, int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        return null;
    }

    @Override
    public ParamInitializer initializer() {
        return EmptyParamInitializer.getInstance();
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {

        //TODO check pooling dimensions wrt. input type

        switch (inputType.getType()){
            case FF:
                throw new UnsupportedOperationException("Global max pooling cannot be applied to feed-forward input type. Got input type = " + inputType);
            case RNN:
                InputType.InputTypeRecurrent recurrent = (InputType.InputTypeRecurrent)inputType;
                if(collapseDimensions){
                    //Return 2d (feed-forward) activations
                    return InputType.feedForward(recurrent.getSize());
                } else {
                    //Return 3d activations, with shape [minibatch, timeStepSize, 1]
                    return recurrent;
                }
            case CNN:
                InputType.InputTypeConvolutional conv = (InputType.InputTypeConvolutional)inputType;
                if(collapseDimensions){
                    return InputType.feedForward(conv.getDepth());
                } else {
                    return InputType.convolutional(1, 1, conv.getDepth());
                }
            case CNNFlat:
                InputType.InputTypeConvolutionalFlat convFlat = (InputType.InputTypeConvolutionalFlat)inputType;
                if(collapseDimensions){
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

        switch(inputType.getType()){
            case FF:
                throw new UnsupportedOperationException("Global max pooling cannot be applied to feed-forward input type. Got input type = " + inputType);
            case RNN:
            case CNN:
                //No preprocessor required
                return null;
            case CNNFlat:
                InputType.InputTypeConvolutionalFlat cFlat = (InputType.InputTypeConvolutionalFlat)inputType;
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
    public double getLearningRateByParam(String paramName) {
        //Not applicable
        return 0;
    }

    public static class Builder extends Layer.Builder<Builder> {

        private PoolingType poolingType = PoolingType.MAX;
        private int[] poolingDimensions;
        private int pnorm;
        private boolean collapseDimensions = true;

        public Builder(PoolingType poolingType){
            this.poolingType = poolingType;
        }

        public Builder poolingDimensions(int... poolingDimensions){
            this.poolingDimensions = poolingDimensions;
            return this;
        }

        public Builder poolingType(PoolingType poolingType){
            this.poolingType = poolingType;
            return this;
        }

        public Builder collapseDimensions(boolean collapseDimensions){
            this.collapseDimensions = collapseDimensions;
            return this;
        }

        public Builder pnorm(int pnorm){
            if(pnorm <= 0) throw new IllegalArgumentException("Invalid input: p-norm value must be greater than 0");
            this.pnorm = pnorm;
            return this;
        }

        @SuppressWarnings("unchecked")
        public GlobalPoolingLayer build(){
            return new GlobalPoolingLayer(this);
        }
    }
}

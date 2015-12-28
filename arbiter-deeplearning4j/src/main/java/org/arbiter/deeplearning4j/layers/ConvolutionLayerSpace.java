package org.arbiter.deeplearning4j.layers;

import org.arbiter.optimize.parameter.FixedValue;
import org.arbiter.optimize.parameter.ParameterSpace;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.nd4j.linalg.convolution.Convolution;

public class ConvolutionLayerSpace extends FeedForwardLayerSpace<ConvolutionLayer> {

    protected ParameterSpace<Convolution.Type> convolutionType;
    protected ParameterSpace<int[]> kernelSize;
    protected ParameterSpace<int[]> stride;
    protected ParameterSpace<int[]> padding;

    private ConvolutionLayerSpace(Builder builder){
        super(builder);
        this.convolutionType = builder.convolutionType;
        this.kernelSize = builder.kernelSize;
        this.stride = builder.stride;
        this.padding = builder.padding;
    }

    @Override
    public ConvolutionLayer randomLayer() {
        ConvolutionLayer.Builder b = new ConvolutionLayer.Builder();
        setLayerOptionsBuilder(b);
        return b.build();
    }

    protected void setLayerOptionsBuilder(ConvolutionLayer.Builder builder){
        super.setLayerOptionsBuilder(builder);
        if(convolutionType != null) builder.convolutionType(convolutionType.randomValue());
        if(kernelSize != null) builder.kernelSize(kernelSize.randomValue());
        if(stride != null) builder.stride(stride.randomValue());
        if(padding != null) builder.padding(padding.randomValue());
    }

    @Override
    public String toString(){
        return toString(", ");
    }

    @Override
    public String toString(String delim){
        StringBuilder sb = new StringBuilder("ConvolutionLayerSpace(");
        if(convolutionType != null) sb.append("poolingType: ").append(convolutionType).append(delim);
        if(kernelSize != null) sb.append("kernelSize: ").append(kernelSize).append(delim);
        if(stride != null) sb.append("stride: ").append(stride).append(delim);
        if(padding != null) sb.append("padding: ").append(padding).append(delim);
        sb.append(super.toString(delim)).append(")");
        return sb.toString();
    }


    public static class Builder extends FeedForwardLayerSpace.Builder<Builder>{

        protected ParameterSpace<Convolution.Type> convolutionType;
        protected ParameterSpace<int[]> kernelSize;
        protected ParameterSpace<int[]> stride;
        protected ParameterSpace<int[]> padding;

        public Builder convolutionType(Convolution.Type convolutionType){
            return convolutionType(new FixedValue<Convolution.Type>(convolutionType));
        }

        public Builder convolutionType(ParameterSpace<Convolution.Type> convolutionType ){
            this.convolutionType = convolutionType;
            return this;
        }

        public Builder kernelSize(int... kernelSize){
            return kernelSize(new FixedValue<int[]>(kernelSize));
        }

        public Builder kernelSize(ParameterSpace<int[]> kernelSize){
            this.kernelSize = kernelSize;
            return this;
        }

        public Builder stride(int... stride){
            return stride(new FixedValue<int[]>(stride));
        }

        public Builder stride(ParameterSpace<int[]> stride){
            this.stride = stride;
            return this;
        }

        public Builder padding(int... padding){
            return padding(new FixedValue<int[]>(padding));
        }

        public Builder padding(ParameterSpace<int[]> padding){
            this.padding = padding;
            return this;
        }

        public ConvolutionLayerSpace build(){
            return new ConvolutionLayerSpace(this);
        }

    }

}

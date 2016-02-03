package org.deeplearning4j.nn.conf.layers;

import lombok.*;

import org.nd4j.linalg.convolution.Convolution;

/**
 * @author Adam Gibson
 */
@Data @NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class ConvolutionLayer extends FeedForwardLayer {
    protected Convolution.Type convolutionType;
    protected int[] kernelSize; // Square filter
    protected int[] stride; // Default is 2. Down-sample by a factor of 2
    protected int[] padding;

    private ConvolutionLayer(Builder builder) {
    	super(builder);
        this.convolutionType = builder.convolutionType;
        if(builder.kernelSize.length != 2)
            throw new IllegalArgumentException("Kernel size of should be rows x columns (a 2d array)");
        this.kernelSize = builder.kernelSize;
        if(builder.stride.length != 2)
            throw new IllegalArgumentException("Invalid stride, must be length 2");
        this.stride = builder.stride;
        this.padding = builder.padding;
    }

    @Override
    public ConvolutionLayer clone() {
        ConvolutionLayer clone = (ConvolutionLayer) super.clone();
        if(clone.kernelSize != null) clone.kernelSize = clone.kernelSize.clone();
        if(clone.stride != null) clone.stride = clone.stride.clone();
        if(clone.padding != null) clone.padding = clone.padding.clone();
        return clone;
    }

    @AllArgsConstructor
    public static class Builder extends FeedForwardLayer.Builder<Builder> {
        private Convolution.Type convolutionType = Convolution.Type.VALID;
        private int[] kernelSize = new int[] {5,5};
        private int[] stride = new int[] {1,1};
        private int[] padding = new int[] {0, 0};


        public Builder(int[] kernelSize, int[] stride, int[] padding) {
            this.kernelSize = kernelSize;
            this.stride = stride;
            this.padding = padding;
        }

        public Builder(int[] kernelSize, int[] stride) {
            this.kernelSize = kernelSize;
            this.stride = stride;
        }

        public Builder(int... kernelSize) {
            this.kernelSize = kernelSize;
        }

        public Builder() {}

        public Builder convolutionType(Convolution.Type convolutionType) {
            this.convolutionType = convolutionType;
            return this;
        }

        /**
         * Size of the convolution
         * rows/columns
         * @param kernelSize the height and width of the
         *                   kernel
         * @return
         */
        public Builder kernelSize(int... kernelSize){
            this.kernelSize = kernelSize;
            return this;
        }

        public Builder stride(int... stride){
            this.stride = stride;
            return this;
        }

        public Builder padding(int... padding){
            this.padding = padding;
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")
        public ConvolutionLayer build() {
            return new ConvolutionLayer(this);
        }
    }
}

package org.deeplearning4j.nn.conf.layers;

import lombok.*;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Batch normalization configuration
 *
 * @author Adam Gibson
 */
@Data @NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
@Builder
public class BatchNormalization extends FeedForwardLayer {
    protected int[] shape; // shape of input
    protected double decay;
    protected double eps = Nd4j.EPS_THRESHOLD;
    protected boolean useBatchMean;
    private boolean finetune;
    protected int N;

    private BatchNormalization(Builder builder){
        super(builder);
        if(builder.shape.length != 2)
            throw new IllegalArgumentException("Kernel size of should be rows x columns (a 2d array)");
        this.decay = builder.decay;
        this.finetune = builder.finetune;
        this.useBatchMean = builder.useBatchMean;
        this.N = builder.N;
    }

    @Override
    public BatchNormalization clone() {
        BatchNormalization clone = (BatchNormalization) super.clone();
        if(clone.shape != null) clone.shape = clone.shape.clone();
        return clone;
    }

    @AllArgsConstructor
    public static class Builder extends FeedForwardLayer.Builder<Builder> {
        protected int[] shape = new int[] {0,0};
        protected double decay = 0.9;
        protected boolean useBatchMean = true; // TODO set this if layer conf is batch
        protected boolean finetune;
        protected int N;

        public Builder(int[] shape, double decay, boolean useBatchMean) {
            this.shape = shape;
            this.decay = decay;
            this.useBatchMean = useBatchMean;
        }

        public Builder(){}
        @Override
        public BatchNormalization build() {
            return new BatchNormalization(this);
        }
    }

}

package org.deeplearning4j.nn.conf.layers;

import lombok.*;

/**Embedding layer: feed-forward layer that expects single integers per example as input (class numbers, in range 0 to numClass-1)
 * as input. This input has shape [numExamples,1] instead of [numExamples,numClasses] for the equivalent one-hot representation.
 * Mathematically, EmbeddingLayer is equivalent to using a DenseLayer with a one-hot representation for the input; however,
 * it can be much more efficient with a large number of classes (as a dense layer + one-hot input does a matrix multiply
 * with all but one value being zero).<br>
 * <b>Note</b>: can only be used as the first layer for a network<br>
 * <b>Note 2</b>: For a given example index i, the output is activationFunction(weights.getRow(i) + bias), hence the
 * weight rows can be considered a vector/embedding for each example.
 * @author Alex Black
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class EmbeddingLayer extends FeedForwardLayer {

    private EmbeddingLayer(Builder builder){
        super(builder);
    }

    @AllArgsConstructor
    public static class Builder extends FeedForwardLayer.Builder<Builder> {

        @Override
        @SuppressWarnings("unchecked")
        public EmbeddingLayer build(){
            return new EmbeddingLayer(this);
        }
    }
}

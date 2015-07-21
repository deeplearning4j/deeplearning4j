package org.deeplearning4j.nn.conf.layers;

import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * @author Adam Gibson
 */
@Data
@NoArgsConstructor
public class SubsamplingLayer extends Layer {
    
    private static final long serialVersionUID = -7095644470333017030L;
    private poolingType poolingType;
    private int[] filterSize; // Get from convolutional layer
    private int[] stride; // Default is 2. Down-sample by a factor of 2

    public enum poolingType {
        MAX, AVG, SUM, NONE
    }


    public SubsamplingLayer(int[] stride) {
        this.stride = stride;
    }

    public SubsamplingLayer(poolingType poolingType) {
        this.poolingType = poolingType;
    }

    public SubsamplingLayer(int[] stride, poolingType poolingType) {
        this.stride = stride;
        this.poolingType = poolingType;
    }

}

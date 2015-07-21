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
    private subsamplingType poolingType;
    private int[] filterSize; // Get from convolutional layer
    private int[] stride; // Default is 2. Down-sample by a factor of 2

    public enum subsamplingType {
        MAX, AVG, SUM, NONE
    }


    public SubsamplingLayer(int[] stride) {
        this.stride = stride;
    }

    public SubsamplingLayer(subsamplingType poolingType) {
        this.poolingType = poolingType;
    }

    public SubsamplingLayer(int[] stride, subsamplingType poolingType) {
        this.stride = stride;
        this.poolingType = poolingType;
    }

}

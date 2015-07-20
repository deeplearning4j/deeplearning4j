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
    //convolutional nets: this is the feature map shape
    private int[] filterSize;
    //aka pool size for subsampling
    private int[] stride;

    public SubsamplingLayer(int nIn, int nOut) {
        this.nIn = nIn;
        this.nOut = nOut;
    }

    public SubsamplingLayer(int nIn, int nOut, int[] filterSize) {
        this.nIn = nIn;
        this.nOut = nOut;
        this.filterSize = filterSize;
    }


    public SubsamplingLayer(int nIn, int nOut, int[] filterSize, int[] stride) {
        this.nIn = nIn;
        this.nOut = nOut;
        this.filterSize = filterSize;
        this.stride = stride;
    }



}

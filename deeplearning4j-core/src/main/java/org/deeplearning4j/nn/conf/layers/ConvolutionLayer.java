package org.deeplearning4j.nn.conf.layers;

import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * @author Adam Gibson
 */
@Data
@NoArgsConstructor
public class ConvolutionLayer extends Layer {
    private static final long serialVersionUID = 3073633667258683720L;


    /**
     * Convolution type: max avg or sum
     */
    public enum ConvolutionType {
        MAX,AVG,SUM,NONE
    }

    private ConvolutionType convolutionType;
    //batch size: primarily used for conv nets. Will be reinforced if set.
    protected int batchSize;
    //feature map
    protected int[] featureMapSize;
    //number of channels for a conv net
    protected int channels = 1;
    //CONVOLUTION_WEIGHTS ??
    //CONVOLUTION_BIAS ??

    public ConvolutionLayer(int nIn, int nOut, int[] featureMapSize) {
        this.nIn = nIn;
        this.nOut = nOut;
        this.featureMapSize = featureMapSize;
    }

    public ConvolutionLayer(int nIn, int nOut, int[] featureMapSize, int channels) {
        this.nIn = nIn;
        this.nOut = nOut;
        this.featureMapSize = featureMapSize;
        this.channels = channels;
    }

    public ConvolutionLayer(int nIn, int nOut, int[] featureMapSize, int channels, int batchSize) {
        this.nIn = nIn;
        this.nOut = nOut;
        this.featureMapSize = featureMapSize;
        this.channels = channels;
        this.batchSize = batchSize;
    }

}

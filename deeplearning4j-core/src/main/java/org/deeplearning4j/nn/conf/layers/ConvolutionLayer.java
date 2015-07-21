package org.deeplearning4j.nn.conf.layers;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.convolution.Convolution;

/**
 * @author Adam Gibson
 */
@Data
@NoArgsConstructor
public class ConvolutionLayer extends Layer {
    private static final long serialVersionUID = 3073633667258683720L;

    private Convolution.Type convolutionType; // FULL / VALID / SAME
    private int[] filterSize; // Square filter
    private WeightInit weightInit; // For initializing weights in kernel

//    A stride of greater than (1, 1) will be implemented in the future.
//    Stride decides the number of learnable filters
//    private int numFilter;
//    private int[] stride;

    public ConvolutionLayer(int[] filterSize) {
        this.filterSize = filterSize;
    }

    public ConvolutionLayer(int[] filterSize, Convolution.Type convolutionType) {
        this.filterSize = filterSize;
        this.convolutionType = convolutionType;
    }

    public ConvolutionLayer(int[] filterSize, Convolution.Type convolutionType, WeightInit weightInit) {
        this.filterSize = filterSize;
        this.convolutionType = convolutionType;
        this.weightInit = weightInit;
    }
}

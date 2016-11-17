package org.deeplearning4j.nn.conf;

/**
 * ConvolutionMode defines how convolution operations should be executed for Convolutional and Subsampling layers,
 * for a given input size and network configuration (specifically stride/padding/kernel sizes).<br>
 * Currently, 2 modes are provided (future releases of DL4J will include others - for example, 'Same' border mode from other libraries).
 * <br>
 * <br>
 * <b>Strict</b>: Output size for Convolutional and Subsampling layers are calculated as follows, in each dimension:
 * outputSize = (inputSize - kernelSize + 2*padding) / stride + 1. If outputSize is not an integer, an exception will
 * be thrown during network initialization or forward pass.<br>
 * <br>
 * <b>Truncate</b>: Output size for Convolutional and Subsampling layers are calculated in the same way as in Strict (that
 * is, outputSize = (inputSize - kernelSize + 2*padding) / stride + 1) in each dimension.<br>
 * If outputSize is an integer, then Strict and Truncate are identical. However, if outputSize is <i>not</i> an integer,
 * the output size will be rounded down to an integer value.<br>
 * Specifically, ConvolutionMode.Truncate implements outSize = floor((inputSize - kernelSize + 2*padding) / stride) + 1.<br>
 * Consequence: a border/edge effect will be seen when rounding down is required. In effect, some number of inputs along
 * the given dimension (height or width) will not be used as input and hence information may be lost. This can be problematic
 * higher in the network (where the cropped activations may represent a significant proportion of the original input),
 * or with large kernel size.<br>
 * In the given dimension (height or width) the number of truncated/cropped input values is equal to
 * (inputSize - kernelSize + 2*padding) % stride. (where % is the modulus/remainder operation).<br>
 * <br>
 * For further information on output sizes for convolutional neural networks, see the "Spatial arrangement" section at
 * <a href="http://cs231n.github.io/convolutional-networks/">http://cs231n.github.io/convolutional-networks/</a>
 *
 * @author Alex Black
 */
public enum ConvolutionMode {

    Strict,
    Truncate,
    Same

}

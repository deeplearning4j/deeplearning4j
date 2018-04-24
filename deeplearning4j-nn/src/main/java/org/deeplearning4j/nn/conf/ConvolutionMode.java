package org.deeplearning4j.nn.conf;

/**
 * ConvolutionMode defines how convolution operations should be executed for Convolutional and Subsampling layers,
 * for a given input size and network configuration (specifically stride/padding/kernel sizes).<br>
 * Currently, 3 modes are provided:
 * <br>
 * <br>
 * <b>Strict</b>: Output size for Convolutional and Subsampling layers are calculated as follows, in each dimension:
 * outputSize = (inputSize - kernelSize + 2*padding) / stride + 1. If outputSize is not an integer, an exception will
 * be thrown during network initialization or forward pass.
 * <br>
 * <br>
 * <br>
 * <b>Truncate</b>: Output size for Convolutional and Subsampling layers are calculated in the same way as in Strict (that
 * is, outputSize = (inputSize - kernelSize + 2*padding) / stride + 1) in each dimension.<br>
 * If outputSize is an integer, then Strict and Truncate are identical. However, if outputSize is <i>not</i> an integer,
 * the output size will be rounded down to an integer value.<br>
 * Specifically, ConvolutionMode.Truncate implements the following:<br>
 * output height = floor((inputHeight - kernelHeight + 2*paddingHeight) / strideHeight) + 1.<br>
 * output width = floor((inputWidth - kernelWidth + 2*paddingWidth) / strideWidth) + 1.<br>
 * where 'floor' is the floor operation (i.e., round down to the nearest integer).<br>
 * <br>
 * The major consequence of this rounding down: a border/edge effect will be seen if/when rounding down is required.
 * In effect, some number of inputs along the given dimension (height or width) will not be used as input and hence
 * some input activations can be lost/ignored. This can be problematic higher in the network (where the cropped activations
 * may represent a significant proportion of the original input), or with large kernel sizes and strides.<br>
 * In the given dimension (height or width) the number of truncated/cropped input values is equal to
 * (inputSize - kernelSize + 2*padding) % stride. (where % is the modulus/remainder operation).<br>
 * <br>
 * <br>
 * <br>
 * <b>Same</b>: Same mode operates differently to Strict/Truncate, in three key ways:<br>
 * (a) Manual padding values in convolution/subsampling layer configuration is not used; padding values are instead calculated
 *     automatically based on the input size, kernel size and strides.<br>
 * (b) The output sizes are calculated differently (see below) compared to Strict/Truncate. Most notably, when stride = 1
 *     the output size is the same as the input size.<br>
 * (c) The calculated padding values may different for top/bottom, and left/right (when they do differ: right and bottom
 *     may have 1 pixel/row/column more than top/left padding)<br>
 * The output size of a Convolutional/Subsampling layer using ConvolutionMode.Same is calculated as follows:<br>
 * output height = ceil( inputHeight / strideHeight )<br>
 * output width = ceil( inputWidth / strideWidth )<br>
 * where 'ceil' is the ceiling operation (i.e., round up to the nearest integer).<br>
 * <br>
 * The padding for top/bottom and left/right are automatically calculated as follows:<br>
 * totalHeightPadding = (outputHeight - 1) * strideHeight + filterHeight - inputHeight<br>
 * totalWidthPadding =  (outputWidth - 1) * strideWidth + filterWidth - inputWidth<br>
 * topPadding = totalHeightPadding / 2      (note: integer division)<br>
 * bottomPadding = totalHeightPadding - topPadding<br>
 * leftPadding = totalWidthPadding / 2      (note: integer division)<br>
 * rightPadding = totalWidthPadding - leftPadding<br>
 * Note that if top/bottom padding differ, then bottomPadding = topPadding + 1
 * <br>
 * <br>
 * <br>
 * For further information on output sizes for convolutional neural networks, see the "Spatial arrangement" section at
 * <a href="http://cs231n.github.io/convolutional-networks/">http://cs231n.github.io/convolutional-networks/</a>
 *
 * @author Alex Black
 */
public enum ConvolutionMode {

    Strict, Truncate, Same

}

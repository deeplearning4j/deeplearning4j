package org.deeplearning4j.nn.layers.convolution;

import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;


/**
 * Confirm calculations to reduce the shape of the input based on convolution or subsampling transformation
 */
public class KernelValidationUtil {

    public static void validateShapes(int inHeight, int inWidth, int kernelHeight, int kernelWidth, int strideHeight,
                               int strideWidth, int padHeight, int padWidth) {

        //Check filter > size + padding
        if (kernelHeight >= (inHeight + 2*padHeight))
            throw new InvalidInputTypeException("Invalid input: activations into layer are h=" + inHeight
                    + " but kernel size is " + kernelHeight + " with padding " + padHeight);

        if (kernelWidth >= (inWidth + 2*padWidth))
            throw new InvalidInputTypeException("Invalid input: activations into layer are w=" + inWidth +
                    " but kernel size is " + kernelWidth + " with padding " + padWidth);

        // Below is to confirm an integer comes out of the calculation but this is taken care of in nd4j
        //Check proposed filter/padding size actually works:
//        if ((inHeight - kernelHeight + 2 * padHeight) % strideHeight != 0) {
//            throw new InvalidInputTypeException("Invalid input/configuration: activations into layer are inputHeight=" + inHeight + ", heightPadding=" + padHeight
//                    + ", kernelHeight = " + kernelHeight + ", strideHeight = " + strideHeight + ". (inputHeight-kernelHeight+2*heightPadding)/strideHeight is not an integer");
//        }
//        if ((inWidth - kernelWidth + 2 * padWidth) % strideWidth != 0)
//            throw new InvalidInputTypeException("Invalid input/configuration: activations into layer are inputWidth=" + inWidth + ", widthPadding=" + padWidth
//                    + ", kernelWidth = " + kernelWidth + ", strideWidth = " + strideWidth + ". (inputWidth-kernelWidth+2*widthPadding)/strideWidth is not an integer");

    }
}

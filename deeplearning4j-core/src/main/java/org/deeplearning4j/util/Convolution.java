package org.deeplearning4j.util;


import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.util.FastMath;
import org.jblas.ComplexDouble;
import org.jblas.ComplexDoubleMatrix;
import org.jblas.DoubleMatrix;
import static org.deeplearning4j.util.MatrixUtil.exp;
import static org.deeplearning4j.util.MatrixUtil.length;

import org.jblas.SimpleBlas;

import org.jblas.ranges.RangeUtils;


import java.awt.*;
import java.awt.image.*;
import java.net.*;
import java.util.*;
import java.io.*;
import java.lang.Math.*;
import java.awt.Color.*;

/**
 * Convolution is the code for applying the convolution operator.
 *  http://www.inf.ufpr.br/danielw/pos/ci724/20102/HIPR2/flatjavasrc/Convolution.java
 * @author: Simon Horne
 */
public class Convolution {

    /**
     * Default no-arg constructor.
     */
    private Convolution() {
    }

    public static enum Type {
        FULL,VALID,SAME
    }


    public static DoubleMatrix conv2d(DoubleMatrix input,DoubleMatrix kernel,Type type) {
        int retRows = input.rows + kernel.rows - 1;
        int retCols = input.columns + kernel.columns - 1;
        DoubleMatrix ret = inverseDisceteFourierTransform(disceteFourierTransform(input).reshape(retRows,retCols).mmul(disceteFourierTransform(kernel).reshape(retRows, retCols)));
        return ret;
    }

    public static ComplexDoubleMatrix complexInverseDisceteFourierTransform(ComplexDoubleMatrix inputC) {
        double len = MatrixUtil.length(inputC);
        ComplexDouble c2 = new ComplexDouble(0,-2).muli(FastMath.PI).divi(len);
        ComplexDoubleMatrix complexRet = complexDisceteFourierTransform(inputC);
        complexRet = complexRet.negi();
        complexRet.divi(len);
        return complexRet;

    }



    public static ComplexDoubleMatrix complexDisceteFourierTransform(ComplexDoubleMatrix inputC) {
        double len = MatrixUtil.length(inputC);
        ComplexDouble c2 = new ComplexDouble(0,-2).muli(FastMath.PI).divi(len);
        ComplexDoubleMatrix range = MatrixUtil.complexRangeVector(0, (int) len);
        ComplexDoubleMatrix div2 = range.transpose().mul(c2);
        ComplexDoubleMatrix div3 = range.mmul(div2);
        ComplexDoubleMatrix matrix = exp(div3);
        ComplexDoubleMatrix complexRet = inputC.isRowVector() ? matrix.mmul(inputC) : inputC.mmul(matrix);

        return complexRet;

    }

    public static ComplexDoubleMatrix complexInverseDisceteFourierTransform(DoubleMatrix input) {
        double len = MatrixUtil.length(input);
        ComplexDouble c2 = new ComplexDouble(0,-2).muli(FastMath.PI).divi(len);
        ComplexDoubleMatrix inputC = new ComplexDoubleMatrix(input);
        ComplexDoubleMatrix range = MatrixUtil.complexRangeVector(0, (int) len);
        ComplexDoubleMatrix div2 = range.transpose().mul(c2);
        ComplexDoubleMatrix div3 = range.mmul(div2).negi();
        ComplexDoubleMatrix matrix = exp(div3).div(len);
        ComplexDoubleMatrix complexRet = inputC.isRowVector() ? matrix.mmul(inputC) : inputC.mmul(matrix);

        return complexRet;
    }



    public static ComplexDoubleMatrix complexDisceteFourierTransform(DoubleMatrix input) {
        double len = MatrixUtil.length(input);
        ComplexDouble c2 = new ComplexDouble(0,-2).muli(FastMath.PI).divi(len);
        ComplexDoubleMatrix inputC = new ComplexDoubleMatrix(input);
        ComplexDoubleMatrix range = MatrixUtil.complexRangeVector(0, (int) len);
        ComplexDoubleMatrix div2 = range.transpose().mul(c2);
        ComplexDoubleMatrix div3 = range.mmul(div2);
        ComplexDoubleMatrix matrix = exp(div3);
        ComplexDoubleMatrix complexRet = inputC.isRowVector() ? matrix.mmul(inputC) : inputC.mmul(matrix);

        return complexRet;

    }


    public static DoubleMatrix inverseDisceteFourierTransform(DoubleMatrix input) {
        return complexInverseDisceteFourierTransform(input).getReal();

    }



    public static DoubleMatrix disceteFourierTransform(DoubleMatrix input) {
        return complexDisceteFourierTransform(input).getReal();

    }


    /**
     * Takes an image (grey-levels) and a kernel and a position,
     * applies the convolution at that position and returns the
     * new pixel value.
     *
     * @param input The 2D double array representing the image.
     * @param x The x coordinate for the position of the convolution.
     * @param y The y coordinate for the position of the convolution.
     * @param k The 2D array representing the kernel.
     * @param kernelWidth The width of the kernel.
     * @param kernelHeight The height of the kernel.
     * @return The new pixel value after the convolution.
     */
    public static double singlePixelConvolution(DoubleMatrix input,
                                                int x, int y,
                                                DoubleMatrix k,
                                                int kernelWidth,
                                                int kernelHeight){
        double output = 0;
        for(int i = 0;i < kernelWidth;++i){
            for(int j = 0; j < kernelHeight;++j){
                output += (input.get(x+i,y+j) * k.get(i,j));
            }
        }
        return output;
    }

    public static int applyConvolution(int [][] input,
                                       int x, int y,
                                       DoubleMatrix k,
                                       int kernelWidth,
                                       int kernelHeight){
        int output = 0;
        for(int i = 0;i < kernelWidth;++i) {
            for(int j = 0;j < kernelHeight;++j) {
                output = output + (int) Math.round(input[x+i][y+j] * k.get(i,j));
            }
        }
        return output;
    }

    /**
     * Takes a 2D array of grey-levels and a kernel and applies the convolution
     * over the area of the image specified by width and height.
     *
     * @param input the 2D double array representing the image
     * @param width the width of the image
     * @param height the height of the image
     * @param kernel the 2D array representing the kernel
     * @param kernelWidth the width of the kernel
     * @param kernelHeight the height of the kernel
     * @return the 2D array representing the new image
     */
    public static DoubleMatrix convolution2D(DoubleMatrix input,
                                             int width, int height,
                                             DoubleMatrix kernel,
                                             int kernelWidth,
                                             int kernelHeight){
        int smallWidth = width - kernelWidth + 1;
        int smallHeight = height - kernelHeight + 1;
        DoubleMatrix output = new DoubleMatrix(smallWidth,smallHeight);

        for(int i=0;i < smallWidth;++i){
            for(int j=0;j < smallHeight;++j){
                output.put(i,j,singlePixelConvolution(input,i,j,kernel,kernelWidth,kernelHeight));
            }
        }
        return output;
    }

    /**
     * Takes a 2D array of grey-levels and a kernel, applies the convolution
     * over the area of the image specified by width and height and returns
     * a part of the final image.
     * @param input the 2D double array representing the image
     * @param width the width of the image
     * @param height the height of the image
     * @param kernel the 2D array representing the kernel
     * @param kernelWidth the width of the kernel
     * @param kernelHeight the height of the kernel
     * @return the 2D array representing the new image
     */
    public static DoubleMatrix convolution2DPadded(DoubleMatrix input,
                                                   int width, int height,
                                                   DoubleMatrix kernel,
                                                   int kernelWidth,
                                                   int kernelHeight){
        int smallWidth = width - kernelWidth + 1;
        int smallHeight = height - kernelHeight + 1;
        int top = kernelHeight / 2;
        int left = kernelWidth / 2;
        DoubleMatrix small  = new DoubleMatrix(smallWidth,smallHeight);
        small = convolution2D(input,width,height,
                kernel,kernelWidth,kernelHeight);
        DoubleMatrix large  = new DoubleMatrix(width,height);

        for(int j = 0;j < smallHeight;++j){
            for(int i = 0;i < smallWidth;++i){
                large.put(i+left,j+top,small.get(i,j));
            }
        }
        return large;
    }

    /**
     * Takes a 2D array of grey-levels and a kernel and applies the convolution
     * over the area of the image specified by width and height.
     *
     * @param input the 2D double array representing the image
     * @param width the width of the image
     * @param height the height of the image
     * @param kernel the 2D array representing the kernel
     * @param kernelWidth the width of the kernel
     * @param kernelHeight the height of the kernel
     * @return the 1D array representing the new image
     */
    public static DoubleMatrix convolutionDouble(DoubleMatrix input,
                                                 int width, int height,
                                                 DoubleMatrix kernel,
                                                 int kernelWidth, int kernelHeight){
        int smallWidth = width - kernelWidth + 1;
        int smallHeight = height - kernelHeight + 1;
        DoubleMatrix small = convolution2D(input,width,height,kernel,kernelWidth,kernelHeight);
        DoubleMatrix result = new  DoubleMatrix(smallWidth * smallHeight);
        for(int j=0;j<smallHeight;++j){
            for(int i=0;i<  smallWidth;++i){
                result.put(j * smallWidth +i,small.get(i,j));
            }
        }
        return result;
    }

    /**
     * Takes a 2D array of grey-levels and a kernel and applies the convolution
     * over the area of the image specified by width and height.
     *
     * @param input the 2D double array representing the image
     * @param width the width of the image
     * @param height the height of the image
     * @param kernel the 2D array representing the kernel
     * @param kernelWidth the width of the kernel
     * @param kernelHeight the height of the kernel
     * @return the 1D array representing the new image
     */
    public static DoubleMatrix convolutionDoublePadded(DoubleMatrix input,
                                                       int width, int height,
                                                       DoubleMatrix kernel,
                                                       int kernelWidth,
                                                       int kernelHeight){
        DoubleMatrix result2D = new DoubleMatrix(width,height);
        result2D = convolution2DPadded(input,width,height,
                kernel,kernelWidth,kernelHeight);
        DoubleMatrix result = new DoubleMatrix(width * height);
        for(int j=0;j < height;++j){
            for(int i=0;i < width;++i){
                result.put(j*width +i,result2D.get(i,j));
            }
        }
        return result;
    }

    /**
     * Converts a greylevel array into a pixel array.
     * @param greys 1D array of greylevels.
     * @return the 1D array of RGB pixels.
     */
    public static int [] doublesToValidPixels (double [] greys){
        int [] result = new int [greys.length];
        int grey;
        for(int i=0;i<greys.length;++i){
            if(greys[i]>255){
                grey = 255;
            }else if(greys[i]<0){
                grey = 0;
            }else{
                grey = (int) Math.round(greys[i]);
            }
            result[i] = (new Color(grey,grey,grey)).getRGB();
        }
        return result;
    }

    /**
     * Applies the convolution2D algorithm to the input array as many as
     * iterations.
     * @param input the 2D double array representing the image
     * @param width the width of the image
     * @param height the height of the image
     * @param kernel the 2D array representing the kernel
     * @param kernelWidth the width of the kernel
     * @param kernelHeight the height of the kernel
     * @param iterations the number of iterations to apply the convolution
     * @return the 2D array representing the new image
     */
    public static DoubleMatrix convolutionType1(DoubleMatrix input,
                                                int width, int height,
                                                DoubleMatrix kernel,
                                                int kernelWidth, int kernelHeight,
                                                int iterations){
        DoubleMatrix newInput =  input.dup();
        DoubleMatrix output =  input.dup();
        for(int i = 0;i < iterations; ++i){
            int smallWidth = width-kernelWidth + 1;
            int smallHeight = height-kernelHeight + 1;
            output = new DoubleMatrix(smallWidth,smallHeight);
            output = convolution2D(newInput,width,height,
                    kernel,kernelWidth,kernelHeight);
            width = smallWidth;
            height = smallHeight;
            newInput = output.dup();
        }
        return output;
    }
    /**
     * Applies the convolution2DPadded  algorithm to the input array as many as
     * iterations.
     * @param input the 2D double array representing the image
     * @param width the width of the image
     * @param height the height of the image
     * @param kernel the 2D array representing the kernel
     * @param kernelWidth the width of the kernel
     * @param kernelHeight the height of the kernel
     * @param iterations the number of iterations to apply the convolution
     * @return the 2D array representing the new image
     */
    public static DoubleMatrix convolutionType2(DoubleMatrix input,
                                                int width, int height,
                                                DoubleMatrix kernel,
                                                int kernelWidth, int kernelHeight,
                                                int iterations){
        DoubleMatrix newInput = input.dup();
        DoubleMatrix output =  input.dup();

        for(int i=0;i<iterations;++i){
            output = new DoubleMatrix(width,height);
            output = convolution2DPadded(newInput,width,height,
                    kernel,kernelWidth,kernelHeight);
            newInput = output.dup();
        }
        return output;
    }

}

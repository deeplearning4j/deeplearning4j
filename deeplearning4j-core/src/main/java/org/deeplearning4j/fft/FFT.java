package org.deeplearning4j.fft;

import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.nn.linalg.NDArray;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.ComplexDouble;
import org.jblas.ComplexDoubleMatrix;
import org.jblas.DoubleMatrix;
import org.jblas.ranges.RangeUtils;

import static org.deeplearning4j.util.MatrixUtil.exp;

/**
 * FFT and IFFT
 * @author Adam Gibson
 */
public class FFT {


    public static ComplexDoubleMatrix fft(NDArray transform,int numElements) {
        return complexDisceteFourierTransform(transform,numElements,transform.columns);
    }

    /**
     * Discrete fourier transform 2d
     * @param input the input to transform
     * @param rows the number of rows in the transformed output matrix
     * @param cols the number of columns in the transformed output matrix
     * @return the discrete fourier transform of the input
     */
    public static ComplexDoubleMatrix complexDisceteFourierTransform(DoubleMatrix input,int rows,int cols) {
        ComplexDoubleMatrix base;

        //pad
        if(input.rows < rows || input.columns < cols)
            base = MatrixUtil.complexPadWithZeros(input, rows, cols);
            //truncation
        else if(input.rows > rows || input.columns > cols) {
            base = new ComplexDoubleMatrix(input);
            base = base.get(MatrixUtil.toIndices(RangeUtils.interval(0, rows)),MatrixUtil.toIndices(RangeUtils.interval(0,cols)));
        }
        else
            base = new ComplexDoubleMatrix(input);

        ComplexDoubleMatrix temp = new ComplexDoubleMatrix(base.rows,base.columns);
        ComplexDoubleMatrix ret = new ComplexDoubleMatrix(base.rows,base.columns);
        for(int i = 0; i < base.columns; i++) {
            ComplexDoubleMatrix column = base.getColumn(i);
            temp.putColumn(i,complexDiscreteFourierTransform1d(column));
        }

        for(int i = 0; i < ret.rows; i++) {
            ComplexDoubleMatrix row = temp.getRow(i);
            ret.putRow(i,complexDiscreteFourierTransform1d(row));
        }
        return ret;

    }


    /**
     * 1d discrete fourier transform, note that this will
     * throw an exception if the passed in input
     * isn't a vector.
     * See matlab's fft2 for more information
     * @param inputC the input to transform
     * @return the the discrete fourier transform of the passed in input
     */
    public static  ComplexDoubleMatrix complexDiscreteFourierTransform1d(ComplexDoubleMatrix inputC) {
        return fft(inputC, inputC.length);
    }


    /**
     * 1d discrete fourier transform, note that this will
     * throw an exception if the passed in input
     * isn't a vector.
     * See matlab's fft2 for more information
     * @param inputC the input to transform
     * @return the the discrete fourier transform of the passed in input
     */
    public static  ComplexDoubleMatrix fft(ComplexDoubleMatrix inputC, int n) {
        if(inputC.rows != 1 && inputC.columns != 1)
            throw new IllegalArgumentException("Illegal input: Must be a vector");

        double len = inputC.length;
        ComplexDouble c2 = new ComplexDouble(0,-2).muli(FastMath.PI).divi(len);
        ComplexDoubleMatrix range = MatrixUtil.complexRangeVector(0, len);
        ComplexDoubleMatrix matrix = exp(range.mmul(range.transpose().mul(c2)));
        ComplexDoubleMatrix complexRet = inputC.isRowVector() ? matrix.mmul(inputC) : inputC.mmul(matrix);
        if(n < complexRet.length) {
            ComplexDoubleMatrix newRet = new ComplexDoubleMatrix(1,n);
            for(int i = 0; i < n; i++)
                newRet.put(i,complexRet.get(i));
            return newRet;
        }


        return complexRet;
    }



}
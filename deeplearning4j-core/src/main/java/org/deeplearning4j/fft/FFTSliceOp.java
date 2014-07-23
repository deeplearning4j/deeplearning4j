package org.deeplearning4j.fft;

import org.deeplearning4j.nn.linalg.ComplexNDArray;
import org.deeplearning4j.nn.linalg.DimensionSlice;
import org.deeplearning4j.nn.linalg.NDArray;
import org.deeplearning4j.nn.linalg.SliceOp;
import org.jblas.ComplexDoubleMatrix;
import org.jblas.DoubleMatrix;

/**
 * FFT Slice operation
 * @author Adam Gibson
 */
public class FFTSliceOp implements SliceOp {
    private int n; //number of elements to operate on per dimension
    private NDArray  operateOn;
    private ComplexNDArray operateOnComplex;


    /**
     * FFT operation on a given slice.
     * Will throw an {@link java.lang.IllegalArgumentException}
     * if n < 1
     * @param operateOn the ndarray to operate on
     * @param n the number of elements per dimension in fft
     */
    public FFTSliceOp(NDArray operateOn,int n) {
        if(n < 1)
            throw new IllegalArgumentException("Number of elements per dimension must be at least 1");

        this.operateOn = operateOn;
        this.n = n;
    }

    /**
     * FFT operation on a given slice
     *
     * Will throw an {@link java.lang.IllegalArgumentException}
     * if n < 1
     *
     * @param operateOnComplex the ndarray to operate on
     * @param n the number of elements per dimension in fft
     */
    public FFTSliceOp(ComplexNDArray operateOnComplex,int n) {
        if(n < 1)
           throw new IllegalArgumentException("Number of elements per dimension must be at least 1");
        this.operateOnComplex = operateOnComplex;
        this.n = n;
    }

    /**
     * FFT along the whole ndarray
     * @param operateOn the ndarray to operate on
     */
    public FFTSliceOp(NDArray operateOn) {
         this(operateOn,operateOn.length);
    }

    /**
     * FFT along the whole ndarray
     * @param operateOnComplex the ndarray to operate on
     */
    public FFTSliceOp(ComplexNDArray operateOnComplex) {
        this(operateOnComplex,operateOnComplex.length);
    }

    /**
     * Operates on an ndarray slice
     *
     * @param nd the result to operate on
     */
    @Override
    public void operate(DimensionSlice nd) {
        if(nd.getResult() instanceof NDArray) {
            NDArray a = (NDArray) nd.getResult();
            int n = this.n < 1 ? a.length : this.n;

            DoubleMatrix result = FFT.fft(a,n).getReal();
            for(int i = 0; i < n; i++) {
                operateOn.data[nd.getIndices()[i]] = result.get(i);
            }

        }
        else if(nd.getResult() instanceof ComplexNDArray) {
            ComplexNDArray a = (ComplexNDArray) nd.getResult();
            int n = this.n < 1 ? a.length : this.n;
            ComplexDoubleMatrix result = FFT.fft(a,n);
            for(int i = 0; i < n; i++) {
                operateOnComplex.put(nd.getIndices()[i],result.get(i));
            }
        }
    }


}

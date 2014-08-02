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


    /**
     * FFT operation on a given slice.
     * Will throw an {@link java.lang.IllegalArgumentException}
     * if n < 1
     * @param n the number of elements per dimension in fft
     */
    public FFTSliceOp(int n) {
        if(n < 1)
            throw new IllegalArgumentException("Number of elements per dimension must be at least 1");
        this.n = n;
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

            DoubleMatrix result = FFT.fft(a,a.length).getReal();
            if(nd.getIndices() == null) {
                for(int i = 0; i < result.length; i++)
                    a.put(i,result.get(i));
            }

            else
                for(int i = 0; i < n; i++) {
                    a.data[nd.getIndices()[i]] = result.get(i);
                }

        }
        else if(nd.getResult() instanceof ComplexNDArray) {
            ComplexNDArray a = (ComplexNDArray) nd.getResult();
            int n =  a.length;
            ComplexDoubleMatrix result = FFT.fft(a,n);
            if(nd.getIndices() == null) {
                for(int i = 0; i < a.length; i++)
                    a.put(i,result.get(i));
            }
            else {
                int count = 0;
                for(int i = 0; i < n; i++) {
                    a.data[nd.getIndices()[count]] = result.get(i).real();
                    a.data[nd.getIndices()[count] + 1] = result.get(i).imag();
                    count++;
                }
            }

        }
    }


}

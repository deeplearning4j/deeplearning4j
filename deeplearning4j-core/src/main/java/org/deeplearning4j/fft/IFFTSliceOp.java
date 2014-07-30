package org.deeplearning4j.fft;

import org.deeplearning4j.nn.linalg.ComplexNDArray;
import org.deeplearning4j.nn.linalg.DimensionSlice;
import org.deeplearning4j.nn.linalg.NDArray;
import org.deeplearning4j.nn.linalg.SliceOp;
import org.jblas.ComplexDoubleMatrix;
import org.jblas.DoubleMatrix;

/**
 * Dimension wise IFFT
 *
 * @author Adam Gibson
 */
public class IFFTSliceOp implements SliceOp {


    private int n; //number of elements per dimension


    /**
     * IFFT on the given nd array
     * @param n number of elements per dimension for the ifft
     */
    public IFFTSliceOp(int n) {
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

            DoubleMatrix result = FFT.ifft(a,n).getReal();
            for(int i = 0; i < n; i++) {
                if(i >= a.length)
                    break;
                a.data[nd.getIndices()[i]] = result.get(i);
            }

        }
        else if(nd.getResult() instanceof ComplexNDArray) {
            ComplexNDArray a = (ComplexNDArray) nd.getResult();
            int n = this.n < 1 ? a.length : this.n;
            ComplexDoubleMatrix result = FFT.ifft(a,n);
            for(int i = 0; i < n; i++) {
                if(i >= a.length)
                    break;
                a.put(nd.getIndices()[i],result.get(i));
            }
        }
    }



}

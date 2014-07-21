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
    private int n;
    private NDArray  operateOn;
    private ComplexNDArray operateOnComplex;

    public FFTSliceOp(NDArray operateOn) {
        this.operateOn = operateOn;
    }

    public FFTSliceOp(ComplexNDArray operateOnComplex) {
        this.operateOnComplex = operateOnComplex;
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

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


    private int n;
    private NDArray operateOn;
    private ComplexNDArray operateOnComplex;



    public IFFTSliceOp(NDArray operateOn,int n) {
        this.operateOn = operateOn;
        this.n = n;
    }

    public IFFTSliceOp(ComplexNDArray operateOnComplex,int n) {
        this.operateOnComplex = operateOnComplex;
        this.n = n;
    }

    public IFFTSliceOp(NDArray operateOn) {
        this(operateOn,operateOn.length);
    }

    public IFFTSliceOp(ComplexNDArray operateOnComplex) {
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

            DoubleMatrix result = FFT.ifft(a,n).getReal();
            for(int i = 0; i < n; i++) {
                operateOn.data[nd.getIndices()[i]] = result.get(i);
            }

        }
        else if(nd.getResult() instanceof ComplexNDArray) {
            ComplexNDArray a = (ComplexNDArray) nd.getResult();
            int n = this.n < 1 ? a.length : this.n;
            ComplexDoubleMatrix result = FFT.ifft(a,n);
            for(int i = 0; i < n; i++) {
                operateOnComplex.put(nd.getIndices()[i],result.get(i));
            }
        }
    }



}

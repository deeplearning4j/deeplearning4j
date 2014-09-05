package org.nd4j.linalg.fft;

import com.google.common.base.Function;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.factory.NDArrays;
import org.nd4j.linalg.util.ComplexNDArrayUtil;

/**
 * Single ifft operation
 *
 * @author Adam Gibson
 */
public class VectorIFFT implements Function<IComplexNDArray,IComplexNDArray> {


    private int n;
    private int originalN = -1;
    /**
     * Create a vector fft operation.
     * If initialized with  a nonzero number, this will
     * find the next power of 2 for the element and truncate the
     * return matrix to the original n
     * @param n
     */
    public VectorIFFT(int n) {
        this.n = n;
    }


    @Override
    public IComplexNDArray apply(IComplexNDArray ndArray) {
        //ifft(x) = conj(fft(conj(x)) / length(x)
        IComplexNDArray ret = new VectorFFT(n).apply(ndArray.conj()).conj().divi(NDArrays.scalar(n));
        return originalN > 0 ? ComplexNDArrayUtil.truncate(ret, originalN, 0) : ret;

    }
}

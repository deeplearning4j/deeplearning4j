package org.deeplearning4j.linalg.fft;

import com.google.common.base.Function;
import org.deeplearning4j.nn.linalg.ComplexNDArray;
import org.deeplearning4j.nn.linalg.NDArray;
import org.deeplearning4j.util.ComplexNDArrayUtil;
import org.deeplearning4j.util.MathUtils;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.ComplexDouble;
import org.jblas.ComplexDoubleMatrix;

import static org.deeplearning4j.util.MatrixUtil.exp;

/**
 * Single ifft operation
 *
 * @author Adam Gibson
 */
public class VectorIFFT implements Function<ComplexNDArray,ComplexNDArray> {


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

    /**
     * Returns the result of applying this function to {@code input}. This method is <i>generally
     * expected</i>, but not absolutely required, to have the following properties:
     * <p/>
     * <ul>
     * <li>Its execution does not cause any observable side effects.
     * <li>The computation is <i>consistent with equals</i>; that is, {@link Objects#equal
     * Objects.equal}{@code (a, b)} implies that {@code Objects.equal(function.apply(a),
     * function.apply(b))}.
     * </ul>
     *
     * @param input
     * @throws NullPointerException if {@code input} is null and this function does not accept null
     *                              arguments
     */
    @Override
    public ComplexNDArray apply(ComplexNDArray ndArray) {
        //ifft(x) = conj(fft(conj(x)) / length(x)
        ComplexNDArray ret = new VectorFFT(n).apply(ndArray.conj()).conj().divi(n);
        return originalN > 0 ? ComplexNDArrayUtil.truncate(ret,originalN,0) : ret;

    }
}

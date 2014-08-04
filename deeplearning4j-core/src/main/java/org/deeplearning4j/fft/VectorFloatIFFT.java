package org.deeplearning4j.fft;

import com.google.common.base.Function;
import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.nn.linalg.ComplexNDArray;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.ComplexDouble;
import org.jblas.ComplexDoubleMatrix;
import org.jblas.ComplexFloat;
import org.jblas.ComplexFloatMatrix;

import static org.deeplearning4j.util.MatrixUtil.exp;

/**
 * Single ifft operation
 *
 * @author Adam Gibson
 */
public class VectorFloatIFFT implements Function<ComplexFloatMatrix,ComplexFloatMatrix> {


    private int n;

    public VectorFloatIFFT(int n) {
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
    public ComplexFloatMatrix apply(ComplexFloatMatrix input) {
        float len = MatrixUtil.length(input);
        ComplexFloat c2 = new ComplexFloat(0,-2).muli((float) FastMath.PI).divi(len);
        ComplexFloatMatrix range = MatrixUtil.complexRangeVectorFloat(0,len);
        ComplexFloatMatrix div2 = range.transpose().mul(c2);
        ComplexFloatMatrix div3 = range.mmul(div2).negi();
        ComplexFloatMatrix matrix = exp(div3).div(len);
        ComplexFloatMatrix complexRet = input.mmul(matrix);


        if(n != complexRet.length) {
            ComplexFloatMatrix newRet = new ComplexFloatMatrix(1,n);
            for(int i = 0; i < n; i++) {
                if(i >= complexRet.length)
                    break;

                newRet.put(i, complexRet.get(i));
            }
            return newRet;
        }

        return complexRet;
    }
}

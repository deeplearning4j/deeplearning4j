package org.deeplearning4j.fft;

import com.google.common.base.Function;
import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.nn.linalg.ComplexNDArray;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.ComplexDouble;
import org.jblas.ComplexFloat;
import org.jblas.ComplexFloatMatrix;

/**
 * Encapsulated vector operation
 *
 * @author Adam Gibson
 */
public class VectorFloatFFT implements Function<ComplexFloatMatrix,ComplexFloatMatrix> {
    private int n;

    public VectorFloatFFT(int n) {
        this.n = n;
    }

    @Override
    public ComplexFloatMatrix apply(ComplexFloatMatrix ndArray) {
        float len = n;
        ComplexFloat c2 = new ComplexFloat(0,-2).muli((float) FastMath.PI).divi(len);
        ComplexFloatMatrix range = MatrixUtil.complexRangeVectorFloat(0, len);
        ComplexFloatMatrix rangeTimesC2 = range.mul(c2);
        ComplexFloatMatrix matrix = MatrixUtil.exp(range.transpose().mmul(rangeTimesC2));
        ComplexFloatMatrix complexRet =  ndArray.mmul(matrix);
        return complexRet;
    }


}

package org.deeplearning4j.fft;

import com.google.common.base.Function;
import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.nn.linalg.ComplexNDArray;
import org.deeplearning4j.util.ComplexNDArrayUtil;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.ComplexDouble;

/**
 * Encapsulated vector operation
 *
 * @author Adam Gibson
 */
public class VectorFFT implements Function<ComplexNDArray,ComplexNDArray> {
    private int n;

    public VectorFFT(int n) {
        this.n = n;
    }

    @Override
    public ComplexNDArray apply(ComplexNDArray ndArray) {
        double len = n;
        int desiredElementsAlongDimension = ndArray.length;

        if(len > desiredElementsAlongDimension) {
            ndArray = ComplexNDArrayUtil.padWithZeros(ndArray,new int[]{n});
        }

        else if(len < desiredElementsAlongDimension) {
            ndArray = ComplexNDArrayUtil.truncate(ndArray,n,0);
        }

        ComplexDouble c2 = new ComplexDouble(0,-2).muli(FastMath.PI).divi(len);
        ComplexNDArray range = ComplexNDArray.wrap(MatrixUtil.complexRangeVector(0, len));
        ComplexNDArray rangeTimesC2 = range.mul(c2);
        ComplexNDArray matrix = ComplexNDArray.wrap(ComplexNDArrayUtil.exp(range.transpose().mmul(rangeTimesC2)));
        ComplexNDArray complexRet =  ndArray.mmul(matrix);
        return ComplexNDArray.wrap(complexRet);
    }


}

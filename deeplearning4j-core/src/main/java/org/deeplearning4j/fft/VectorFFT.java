package org.deeplearning4j.fft;

import static  org.deeplearning4j.util.ComplexNDArrayUtil.exp;

import com.google.common.base.Function;
import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.nn.linalg.ComplexNDArray;
import org.deeplearning4j.util.ComplexNDArrayUtil;
import org.deeplearning4j.util.MathUtils;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.ComplexDouble;

/**
 * Encapsulated vector operation
 *
 * @author Adam Gibson
 */
public class VectorFFT implements Function<ComplexNDArray,ComplexNDArray> {
    private int n;
    private int originalN = -1;
    /**
     * Create a vector fft operation.
     * If initialized with  a nonzero number, this will
     * find the next power of 2 for the element and truncate the
     * return matrix to the original n
     * @param n
     */
    public VectorFFT(int n) {
        this.n = n;
        if (Math.log(n) % 1 != 0) {
            this.n = (int) MathUtils.nextPowOf2(n);
            this.originalN = n;
        }
    }

    @Override
    public ComplexNDArray apply(ComplexNDArray ndArray) {
        double len = n;

        int desiredElementsAlongDimension = ndArray.length;

        if(len > desiredElementsAlongDimension) {
            ndArray = ComplexNDArrayUtil.padWithZeros(ndArray,new int[]{n});
        }

        else if(len < desiredElementsAlongDimension) {
            ndArray = ComplexNDArrayUtil.truncate(ndArray, n, 0);
        }


        ComplexDouble c2 = new ComplexDouble(0,-2).muli(FastMath.PI);
        //row vector
        ComplexNDArray n = ComplexNDArray.wrap(MatrixUtil.complexRangeVector(0d, this.n));
        //column vector
        ComplexNDArray k = n.reshape(new int[]{n.length,1});
        ComplexNDArray nk = k.mmul(n);
        ComplexNDArray nkc2 = k.mmul(n).muli(c2);
        ComplexNDArray nkc2div = k.mmul(n).muli(c2).divi(len);

        ComplexNDArray M = exp(k.mmul(n).muli(c2).divi(len));
        ComplexNDArray matrix = M.mmul(ndArray.reshape(new int[]{ndArray.length,1}));
        if(originalN > 0) {
            matrix = ComplexNDArrayUtil.truncate(matrix, originalN, 0);

        }

        return matrix;
    }


}

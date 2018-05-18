package org.nd4j.linalg.api.blas.params;

import lombok.Data;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JArraySizeException;

/**
 * Gemv parameters:
 * The parameters for general matrix
 * vector operations
 *
 * @author Adam Gibson
 */
public @Data class GemvParameters {
    private int m, n, lda, incx, incy;
    private INDArray a, x, y;
    private char aOrdering = 'N';

    public GemvParameters(INDArray a, INDArray x, INDArray y) {
        a = copyIfNecessary(a);
        x = copyIfNecessaryVector(x);
        this.a = a;
        this.x = x;
        this.y = y;

        if (a.columns() > Integer.MAX_VALUE || a.rows() > Integer.MAX_VALUE)
            throw new ND4JArraySizeException();

        if (x.columns() > Integer.MAX_VALUE || x.rows() > Integer.MAX_VALUE)
            throw new ND4JArraySizeException();


        if (a.ordering() == 'f' && a.isMatrix()) {
            this.m = (int) a.rows();
            this.n = (int) a.columns();
            this.lda = (int) a.rows();
        } else if (a.ordering() == 'c' && a.isMatrix()) {
            this.m = (int) a.columns();
            this.n = (int) a.rows();
            this.lda = (int) a.columns();
            aOrdering = 'T';
        }

        else {
            this.m = (int) a.rows();
            this.n = (int) a.columns();
            this.lda = (int) a.size(0);
        }


        if (x.rank() == 1) {
            incx = 1;
        } else if (x.isColumnVector()) {
            incx = x.stride(0);
        } else {
            incx = x.stride(1);
        }

        this.incy = y.elementWiseStride();

        if (x instanceof IComplexNDArray)
            this.incx /= 2;
        if (y instanceof IComplexNDArray)
            this.incy /= 2;

    }

    private INDArray copyIfNecessary(INDArray arr) {
        //See also: Shape.toMmulCompatible - want same conditions here and there
        //Check if matrix values are contiguous in memory. If not: dup
        //Contiguous for c if: stride[0] == shape[1] and stride[1] = 1
        //Contiguous for f if: stride[0] == 1 and stride[1] == shape[0]
        if (arr.ordering() == 'c' && (arr.stride(0) != arr.size(1) || arr.stride(1) != 1))
            return arr.dup();
        else if (arr.ordering() == 'f' && (arr.stride(0) != 1 || arr.stride(1) != arr.size(0)))
            return arr.dup();
        else if (arr.elementWiseStride() < 1)
            return arr.dup();
        return arr;
    }

    private INDArray copyIfNecessaryVector(INDArray vec) {
        if (vec.offset() != 0)
            return vec.dup();
        return vec;
    }

}

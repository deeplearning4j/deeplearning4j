package org.nd4j.linalg.api.blas.params;

import lombok.Data;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Iterator;

/**
 * Gemv parameters:
 * The parameters for general matrix
 * vector operations
 *
 * @author Adam Gibson
 */
public @Data class GemvParameters {
    private int m,n,lda,incx,incy;
    private INDArray a,x,y;
    private char aOrdering = 'N';

    public GemvParameters(INDArray a,INDArray x,INDArray y) {

        this.a = a;
        this.x = x;
        this.y = y;

        if(a.ordering() == 'f' && a.isMatrix()) {
            this.m = a.rows();
            this.n = y.columns();
            this.lda = a.rows();
        }
        else if(a.ordering() == 'c' && a.isMatrix()) {
            this.m = a.rows();
            this.n = x.length();
            this.lda = a.columns();
            aOrdering = 'T';
        }

        else  {
            this.m = a.rows();
            this.n = a.columns();
            this.lda = a.size(0);
        }


        if(x.isColumnVector()) {
            if(x.ordering() == 'f')
                incx = x.stride(0);
            else
                incx = x.stride(1);
        }
        else {
            if(x.ordering() == 'f')
                incx = x.stride(1);
            else
                incx = x.stride(0);
        }

        this.incy = y.elementStride();

        if(x instanceof IComplexNDArray)
            this.incx /= 2;
        if(y instanceof IComplexNDArray)
            this.incy /= 2;

    }


}

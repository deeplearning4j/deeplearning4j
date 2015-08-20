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

    public GemvParameters(INDArray a,INDArray x,INDArray y) {
        this.a = a;
        this.x = x;
        this.y = y;
        this.m = a.rows();
        this.n = a.columns();
        this.lda = Math.max(1, m);

        if(a.ordering() == 'c') {
            INDArray newOrder = Nd4j.create(a.shape(),'f');
            Iterator<int[]> copy = new NdIndexIterator(a.shape());
            while(copy.hasNext()) {
                int[] next = copy.next();
                newOrder.putScalar(next,a.getDouble(next));
            }
            a = newOrder;
        }


        if(a.isVector() && a.ordering() == NDArrayFactory.FORTRAN) {
            this.lda = a.stride(0);
            if(a instanceof IComplexNDArray)
                this.lda /= 2;
        }

        this.incx = x.elementStride();
        this.incy = y.elementStride();

        if(x instanceof IComplexNDArray)
            this.incx /= 2;
        if(y instanceof IComplexNDArray)
            this.incy /= 2;

    }


}

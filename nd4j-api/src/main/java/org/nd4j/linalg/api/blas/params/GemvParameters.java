package org.nd4j.linalg.api.blas.params;

import lombok.Data;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDArrayFactory;

/**
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
        this.lda = a.rows();
        if(a.isVector() && a.ordering() == NDArrayFactory.FORTRAN) {
            this.lda = a.stride(0);
            if(a instanceof IComplexNDArray)
                this.lda /= 2;
        }
        this.incx = x.majorStride();
        this.incy = y.majorStride();
        if(x instanceof IComplexNDArray)
            this.incx /= 2;
        if(y instanceof IComplexNDArray)
            this.incy /= 2;

    }


}

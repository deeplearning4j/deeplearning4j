package org.nd4j.linalg.api.blas.params;

import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;

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
        this.incx = x.majorStride();
        this.incy = y.majorStride();

    }


}

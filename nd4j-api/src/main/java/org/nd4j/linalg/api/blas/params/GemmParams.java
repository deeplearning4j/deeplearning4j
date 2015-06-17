package org.nd4j.linalg.api.blas.params;

import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDArrayFactory;

/**
 * Used for setting the gemm parameters
 * Separates blas logic from
 * the run time itself.
 *
 * @author Adam Gibson
 */
public @Data class GemmParams {
    private int lda,ldb,ldc,m,n,k;
    private INDArray a,b,c;

    public GemmParams(INDArray a,INDArray b,INDArray c) {
        if(b.columns() != c.columns())
            throw new IllegalArgumentException("B columns must match c columns");
        if(a.rows() != c.rows())
            throw new IllegalArgumentException("A rows must equal c rows");
        this.a = a;
        this.b = b;
        this.c = c;
        this.m = a.rows();
        this.n = b.columns();
        this.k = a.columns();
        this.lda = a.size(0);
        this.ldb = b.size(0);
        this.ldc = c.size(0);
        if(a.ordering() == NDArrayFactory.C) {
            int oldN = n;
            int oldM = m;
            this.m = oldN;
            this.n = oldM;
            this.lda = n;
            this.ldb = k;

        }

        validate();
    }


    private void validate() {

        if(m < 0)
            throw new IllegalStateException("M must be >= 0");
        if(n < 0)
            throw new IllegalStateException("N must be >= 0");
        if(k < 0)
            throw new IllegalStateException("K must be at least 0");

    }




}

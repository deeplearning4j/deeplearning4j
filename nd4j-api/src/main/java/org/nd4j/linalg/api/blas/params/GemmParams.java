package org.nd4j.linalg.api.blas.params;

import lombok.Data;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;

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
    private char aOrdering = 'N';
    private char bOrdering = 'N';

    /**
     *
     * @param a
     * @param b
     * @param c
     */
    public GemmParams(INDArray a,INDArray b,INDArray c) {
        if(b.columns() != c.columns())
            throw new IllegalArgumentException("B columns must match c columns");
        if(a.rows() != c.rows())
            throw new IllegalArgumentException("A rows must equal c rows");


        //automatically assume fortran ordering
        //multiple backends force us to be
        //in fortran ordering only
        this.a = a;
        this.b = b;
        this.c = c;

        this.m = a.ordering() == 'f' ? a.size(0) : a.size(1);
        this.n = b.ordering() == 'f' ? b.size(1) : b.size(0);
        this.k = c.ordering() == 'f' ? a.size(1) : a.size(0);



        this.lda = a.size(0) > 1 ? a.size(0) : 1;
        this.ldb = b.size(0) > 1 ? b.size(0) : 1;
        this.ldc = c.size(0) > 1 ? c.size(0) : 1;

        if(a.ordering() == 'c') {
            aOrdering = 'T';
            lda = a.size(1) > 1 ? a.size(1) : 1;
        }

        if(b.ordering() == 'c') {
            bOrdering = 'T';
            ldb = b.size(1) > 1 ? b.size(1) : 1;
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

package org.nd4j.linalg.api.blas.params;

import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;

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

        //multiple backends force us to be
        //in fortran ordering only
        this.a = a;
        this.b = b;
        this.c = c;

        if(a.ordering() == 'f' && b.ordering() == 'f'){
            this.m = a.rows();
            this.n = b.columns();
            this.k = a.columns();
            this.lda = a.rows();
            this.ldb = b.rows();
            this.ldc = a.rows();
        }
        else if(a.ordering() == 'c' && b.ordering() == 'c') {
            this.m = c.rows();
            this.n = c.columns();
            this.k = b.rows();
            this.lda = a.columns();
            this.ldb = b.columns();
            this.ldc = c.rows();
            aOrdering = 'T';
            bOrdering = 'T';
        }
        else
            throw new RuntimeException();

        ldc = c.size(0);

    }





}

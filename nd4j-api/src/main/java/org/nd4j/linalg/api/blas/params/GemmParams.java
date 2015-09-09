package org.nd4j.linalg.api.blas.params;

import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;

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

        if(a.ordering() == 'f' && b.ordering() == 'f'){
            this.m = a.rows();
            this.n = b.columns();
            this.k = a.columns();
            this.lda = a.rows();
            this.ldb = b.rows();
            this.ldc = a.rows();
        } else if(a.ordering() == 'c' && b.ordering() == 'c') {
            this.m = c.rows();
            this.n = c.columns();
            this.k = b.rows();

            this.lda = a.columns();
            this.ldb = b.columns();
            this.ldc = c.rows();
            aOrdering = 'T';
            bOrdering = 'T';
        } else throw new RuntimeException();

        /*
        this.m = a.ordering() == 'f' ? a.size(0) : a.size(1);
        this.n = b.ordering() == 'f' ? b.size(1) : b.size(0);
        this.k = c.ordering() == 'f' ? a.size(1) : a.size(0);



        this.lda = a.ordering() == 'f' ? (a.size(0) > 1 ? a.size(0) : 1) : (a.size(1) > 1 ? a.size(1) : 1);
        this.ldb = b.ordering() == 'f' ? (b.size(0) > 1 ? b.size(0) : 1) : (b.size(1) > 1 ? b.size(1) : 1);
        this.ldc = c.size(0) > 1 ? c.size(0) : 1;

//        this.ldc = c.size(1);

        if(a.ordering() == 'c') {
            aOrdering = 'T';
        }

        if(b.ordering() == 'c') {
            bOrdering = 'T';
        }



        ldc = c.size(0);*/


        validate();
    }




    private void validate() {

        if( c.ordering() != 'f' ) throw new IllegalStateException("C is not order f");

        /*
        if(aOrdering == 'N') {
            if(a.columns() != k)
                throw new IllegalStateException("When trans(a) == n a columns must be equal to k");
            if(lda < Math.max(1,m))
                throw new IllegalStateException("When trans(a) == n lda must be >= max(1,m)");

        }
        else {
            if(a.columns() != m)
                throw new IllegalStateException("When trans(a) == t a columns must be m");
            if(lda < Math.max(1,k))
                throw new IllegalStateException("When trans(a) == t lda must be >= max(1,k)");
        }

        if(bOrdering == 'N') {
            if(b.columns() != n)
                throw new IllegalStateException("When trans(b) == n b columns must be n");
            if(ldb < Math.max(1,k))
                throw new IllegalStateException("When trans(b) == n ldb must be >= max(1,k)");
        }
        else {
            if(b.rows() != k)
                throw new IllegalStateException("When trans(b) == t b columns must be k");
            if(ldb < Math.max(1,n))
                throw new IllegalStateException("When trans(b) == t ldb must be >= max(1,n)");
        }


        if(ldc < Math.max(1,m))
            throw new IllegalStateException("ldc must be >= max(1,m)");

        if(m < 0)
            throw new IllegalStateException("M must be >= 0");
        if(n < 0)
            throw new IllegalStateException("N must be >= 0");
        if(k < 0)
            throw new IllegalStateException("K must be at least 0");*/

    }




}

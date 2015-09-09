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
        this.a = copyIfNeccessary(a);
        this.b = copyIfNeccessary(b);
        this.c = copyIfNeccessary(c);

        this.m = c.rows();
        this.n = c.columns();
        if(this.a.ordering() == 'f' && this.b.ordering() == 'f') {
            this.k = this.a.columns();   //common dimension: = this.b.rows()
            this.lda = this.a.rows();
            this.ldb = this.b.rows();
            this.ldc = this.a.rows();
        } else if(this.a.ordering() == 'c' && this.b.ordering() == 'c') {
            this.k = this.b.rows();

            this.lda = this.a.columns();
            this.ldb = this.b.columns();
            this.ldc = this.c.rows();
            aOrdering = 'T';
            bOrdering = 'T';
        } else if(this.a.ordering() == 'f' && this.b.ordering() == 'c') {
            this.k = this.a.columns();   //common dimension of a and b

            this.lda = this.a.rows();
            this.ldb = this.b.columns();
            this.ldc = this.c.rows();
            bOrdering = 'T';

        } else if(this.a.ordering() == 'c' && this.b.ordering() == 'f' ){
            this.k = this.b.rows();  //common dimension of a and b

            this.lda = this.a.columns(); //normally a.rows() but swap for c->f
            this.ldb = this.b.rows();
            this.ldc = this.c.rows();
            aOrdering = 'T';

        } else throw new RuntimeException();



        validate();
    }



    private INDArray copyIfNeccessary(INDArray arr) {
        if(arr.isMatrix()) {
            //Check if matrix values are contiguous in memory. If not: dup
            //Contiguous for c if: stride[0] == shape[1] and stride[1] = 1
            //Contiguous for f if: stride[0] == 1 and stride[1] == shape[0]
            if(arr.ordering() == 'c' && (arr.stride(0) != arr.size(1) || arr.stride(1) != 1) ) return arr.dup();
            else if(arr.stride(0) != 1 || arr.stride(1) != arr.size(0)) return arr.dup();
        }
        return arr;
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

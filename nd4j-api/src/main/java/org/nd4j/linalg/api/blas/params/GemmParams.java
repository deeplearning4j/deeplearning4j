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
        //homogenize row/column major
        if(a.ordering() != b.ordering()) {
            if(a.ordering() != 'f') {
                INDArray rearrangedA = Nd4j.create(a.shape(),'f');
                NdIndexIterator iter = new NdIndexIterator('c',rearrangedA.shape());
                while(iter.hasNext()) {
                    int[] next = iter.next();
                    rearrangedA.putScalar(next, a.getDouble(next));
                }

                a = rearrangedA;
            }

            if(b.ordering() != 'f') {
                INDArray rearrangedB = Nd4j.create(b.shape(),'f');
                NdIndexIterator iter = new NdIndexIterator('c',rearrangedB.shape());
                while(iter.hasNext()) {
                    int[] next = iter.next();
                    rearrangedB.putScalar(next,b.getDouble(next));
                }

                b = rearrangedB;
            }


        }


        this.a = a;
        this.b = b;
        this.c = c;
        this.m = a.rows();
        this.n = b.columns();
        this.k = a.columns();

        if(a.ordering() == NDArrayFactory.C && b.ordering() == NDArrayFactory.C) {
            int oldN = n;
            int oldM = m;
            this.m = oldN;
            this.n = oldM;
            //invert the operation
            this.a = b;
            this.b = a;
        }

        this.lda = Math.max(1, m);
        this.ldb = Math.max(1, k);
        this.ldc = Math.max(1, m);
        if(unevenStrides(a))
            this.a = a.dup();
        if(unevenStrides(b))
            this.b = b.dup();

        validate();
    }

    protected boolean unevenStrides(INDArray arr) {
        return arr.ordering() == 'f' && arr.offset() > 0;
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

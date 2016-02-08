package org.nd4j.linalg.api.blas.params;

import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
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
    private char transA = 'N';
    private char transB = 'N';
    private char ordering = 'f';


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



        if(Nd4j.allowsSpecifyOrdering()) {
            if(a.ordering() == b.ordering()) {
                //both will be same ordering for cblas
                this.ordering = a.ordering();
                //automatically assume fortran ordering
                //multiple backends force us to be
                //in fortran ordering only
                this.a = copyIfNeccessary(a);
                this.b = copyIfNeccessary(b);
                this.c = c;
                if(ordering == 'c') {
                    this.m = c.columns();
                    this.n = c.rows();
                    this.k = a.columns();
                }
                else {
                    this.m = c.rows();
                    this.n = c.columns();
                    this.k = b.columns();
                }

                this.lda = a.rows();
                this.ldb = b.rows();
                this.ldc = c.rows();

                this.transA = 'N';
                this.transB = 'N';
            }
            else {
                //automatically assume fortran ordering
                //multiple backends force us to be
                //in fortran ordering only
                this.a = copyIfNeccessary(a);
                this.b = b.dup(a.ordering());
                this.c = c;

                this.m = c.rows();
                this.n = c.columns();
                this.k = a.columns();

                this.ordering = a.ordering();

                this.lda = a.rows();
                this.ldb = b.rows();
                this.ldc = c.rows();

                this.transA = 'N';
                this.transB = 'N';
            }


        }
        else {

            //automatically assume fortran ordering
            //multiple backends force us to be
            //in fortran ordering only
            this.a = copyIfNeccessary(a);
            this.b = copyIfNeccessary(b);
            this.c = c;

            this.m = c.rows();
            this.n = c.columns();
            this.k = a.columns();

            //always fortran ordering
            this.lda = (this.a.ordering() == 'f' ? this.a.rows() : this.a.columns());  //Leading dimension of a, as declared. But swap if 'c' order
            this.ldb = (this.b.ordering() == 'f' ? this.b.rows() : this.b.columns());  //Leading dimension of b, as declared. But swap if 'c' order
            this.ldc = c.rows();

            this.transA = (this.a.ordering() == 'c' ? 'T' : 'N');
            this.transB = (this.b.ordering() == 'c' ? 'T' : 'N');

        }

        ///validate();
    }

    public GemmParams(INDArray a, INDArray b, INDArray c, boolean transposeA, boolean transposeB) {
        if(transposeA && a.columns() != c.rows()) throw new IllegalArgumentException("transposeA but a.columns != c.rows");
        else if(!transposeA && a.rows() != c.rows() ) throw new IllegalArgumentException("a.rows != c.rows");
        if(transposeB && b.rows() != c.columns()) throw new IllegalArgumentException("transposeB but b.rows != c.columns");
        else if(!transposeB && b.columns() != c.columns()) throw new IllegalArgumentException("b.columns != c.columns");
        if(c.ordering() != 'f' || c.offset() != 0 || c.length() != c.data().length() )
            throw new IllegalArgumentException("c must be f order, offset 0 and have length == c.data.length");

        //automatically assume fortran ordering
        //multiple backends force us to be
        //in fortran ordering only
        this.a = copyIfNeccessary(a);
        this.b = copyIfNeccessary(b);
        this.c = c;

        this.m = c.rows();
        this.n = c.columns();
        this.k = a.columns();

        //Might transpose because (a) it's the op we want, and (b) because order is c.
        //But 2 transposes == no transpose
        boolean transposeAOut = transposeA ^ this.a.ordering() == 'c';
        boolean transposeBOut = transposeB ^ this.b.ordering() == 'c';

        this.lda = (this.a.ordering() == 'f' ? this.a.rows() : this.a.columns());  //Leading dimension of a, as declared. But swap if 'c' order
        this.ldb = (this.b.ordering() == 'f' ? this.b.rows() : this.b.columns());  //Leading dimension of b, as declared. But swap if 'c' order
        this.ldc = c.rows();

        this.transA = (transposeAOut ? 'T' : 'N');
        this.transB = (transposeBOut ? 'T' : 'N');
    }



    private INDArray copyIfNeccessary(INDArray arr) {
        //See also: Shape.toMmulCompatible - want same conditions here and there
        if(arr.isMatrix()) {
            //Check if matrix values are contiguous in memory. If not: dup
            //Contiguous for c if: stride[0] == shape[1] and stride[1] = 1
            //Contiguous for f if: stride[0] == 1 and stride[1] == shape[0]
            if(!Nd4j.allowsSpecifyOrdering() && arr.ordering() == 'c' && (arr.stride(0) != arr.size(1) || arr.stride(1) != 1))
                return arr.dup();
            else if(arr.ordering() == 'f' && (arr.stride(0) != 1 || arr.stride(1) != arr.size(0)))
                return arr.dup();
            else if(arr.elementWiseStride() < 0)
                return arr.dup();
        }
        return arr;
    }



    private void validate() {
        if(ordering == 'c') {
            if(transA == 'T' || transA == 't') {
               if(m != a.rows())
                   throw new IllegalArgumentException("M under transpose and c ordering must be a.columns()");
                if(k != a.columns())
                    throw new IllegalArgumentException("K under transpose and c ordering must be a.rows()");
            }
            //N
            else  {
               if(m != a.columns())
                   throw new IllegalArgumentException("M under no transpose and c ordering must be a.rows()");
                if(k != a.rows())
                    throw new IllegalArgumentException("K under no transpose and c ordering must be a.columns()");
            }
        }
        else {
            if(transB == 't' || transB == 'T') {
                if(n != b.columns())
                    throw new IllegalArgumentException("N under transpose and c ordering ust be b.rows()");
                if(k != b.rows())
                    throw new IllegalArgumentException("K under tranpose and c ordering must be b.columns()");
            }
            //N
            else {
                if(n != b.rows())
                    throw new IllegalArgumentException("N under no transpose and c ordering must be b.columns()");
                if(k != b.columns())
                    throw new IllegalArgumentException("K under no transpose and c ordering must be b.rows()");
            }
        }


    }




}
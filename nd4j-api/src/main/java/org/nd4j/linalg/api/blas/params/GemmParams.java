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
        if(Nd4j.allowsSpecifyOrdering() && a.ordering() == b.ordering()) {

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

            this.lda = (this.a.ordering() == 'f' ? this.a.rows() : this.a.columns());  //Leading dimension of a, as declared. But swap if 'c' order
            this.ldb = (this.b.ordering() == 'f' ? this.b.rows() : this.b.columns());  //Leading dimension of b, as declared. But swap if 'c' order
            this.ldc = c.rows();

            this.aOrdering = (this.a.ordering() == 'c' ? 'T' : 'N');
            this.bOrdering = (this.b.ordering() == 'c' ? 'T' : 'N');

            validate();
        }


    }

    public GemmParams(INDArray a, INDArray b, INDArray c, boolean transposeA, boolean transposeB) {
        if(transposeA && a.columns() != c.rows()) throw new IllegalArgumentException("transposeA but a.columns != c.rows");
        else if(!transposeA && a.rows() != c.rows() ) throw new IllegalArgumentException("a.rows != c.rows");
        if(transposeB && b.rows() != c.columns()) throw new IllegalArgumentException("transposeB but b.rows != c.columns");
        else if(!transposeB && b.columns() != c.columns()) throw new IllegalArgumentException("b.columns != c.columns");
        if(c.ordering() != 'f' || c.offset() != 0 || c.length() != c.data().length() )
            throw new IllegalArgumentException("c must be f order, offset 0 and have length == c.data.length");
        if(Nd4j.allowsSpecifyOrdering() && a.ordering() == b.ordering()) {
            this.a = a;
            this.b = b;
            this.c = c;


            this.m = c.rows();
            this.n = c.columns();
            this.k = (transposeA ? a.rows() : a.columns());

            this.lda = this.a.rows();  //Leading dimension of a, as declared. But swap if 'c' order
            this.ldb = b.rows();  //Leading dimension of b, as declared. But swap if 'c' order
            this.ldc = c.rows();
            //Might transpose because (a) it's the op we want, and (b) because order is c.
            //But 2 transposes == no transpose
            boolean transposeAOut = transposeA ^ this.a.ordering() == 'c';
            boolean transposeBOut = transposeB ^ this.b.ordering() == 'c';

            this.aOrdering = (transposeAOut ? 'T' : 'N');
            this.bOrdering = (transposeBOut ? 'T' : 'N');

        }
        else if(Nd4j.allowsSpecifyOrdering()) {
            this.a = a;
            this.b = b;
            this.c = c;


            this.m = c.rows();
            this.n = c.columns();
            this.k = (transposeA ? a.rows() : a.columns());

            this.lda = this.a.rows();  //Leading dimension of a, as declared. But swap if 'c' order
            this.ldb = b.rows();  //Leading dimension of b, as declared. But swap if 'c' order
            this.ldc = c.rows();
            //Might transpose because (a) it's the op we want, and (b) because order is c.
            //But 2 transposes == no transpose
            boolean transposeAOut = transposeA ^ this.a.ordering() == 'c';
            boolean transposeBOut = transposeB ^ this.b.ordering() == 'c';

            this.aOrdering = (transposeAOut ? 'T' : 'N');
            this.bOrdering = (transposeBOut ? 'T' : 'N');
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
            this.k = (transposeA ? a.rows() : a.columns());

            //Might transpose because (a) it's the op we want, and (b) because order is c.
            //But 2 transposes == no transpose
            boolean transposeAOut = transposeA ^ this.a.ordering() == 'c';
            boolean transposeBOut = transposeB ^ this.b.ordering() == 'c';

            this.lda = (this.a.ordering() == 'f' ? this.a.rows() : this.a.columns());  //Leading dimension of a, as declared. But swap if 'c' order
            this.ldb = (this.b.ordering() == 'f' ? this.b.rows() : this.b.columns());  //Leading dimension of b, as declared. But swap if 'c' order
            this.ldc = c.rows();

            this.aOrdering = (transposeAOut ? 'T' : 'N');
            this.bOrdering = (transposeBOut ? 'T' : 'N');

        }
    }



    private INDArray copyIfNeccessary(INDArray arr) {
        //See also: Shape.toMmulCompatible - want same conditions here and there
        if(arr.isMatrix()) {
            //Check if matrix values are contiguous in memory. If not: dup
            //Contiguous for c if: stride[0] == shape[1] and stride[1] = 1
            //Contiguous for f if: stride[0] == 1 and stride[1] == shape[0]
            if(arr.ordering() == 'c' && (arr.stride(0) != arr.size(1) || arr.stride(1) != 1) )
                return arr.dup();
            else if(arr.ordering() == 'f' && (arr.stride(0) != 1 || arr.stride(1) != arr.size(0)))
                return arr.dup();
        }

        return arr;
    }



    private void validate() {

       // if( c.ordering() != 'f' ) throw new IllegalStateException("C is not order f");

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

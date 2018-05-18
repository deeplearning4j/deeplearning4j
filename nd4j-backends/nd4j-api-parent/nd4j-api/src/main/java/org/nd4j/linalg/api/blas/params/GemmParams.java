package org.nd4j.linalg.api.blas.params;

import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JArraySizeException;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

/**
 * Used for setting the gemm parameters
 * Separates blas logic from
 * the run time itself.
 *
 * @author Adam Gibson
 */
public @Data class GemmParams {
    private int lda, ldb, ldc, m, n, k;
    private INDArray a, b, c;
    private char transA = 'N';
    private char transB = 'N';
    private char ordering = 'f';


    /**
     *
     * @param a
     * @param b
     * @param c
     */
    public GemmParams(INDArray a, INDArray b, INDArray c) {
        if (a.columns() != b.rows()) {
            throw new IllegalArgumentException("A columns must equal B rows. MMul attempt: "
                            + Arrays.toString(a.shape()) + "x" + Arrays.toString(b.shape()));
        }
        if (b.columns() != c.columns()) {
            throw new IllegalArgumentException("B columns must match C columns. MMul attempt: "
                            + Arrays.toString(a.shape()) + "x" + Arrays.toString(b.shape())
                            + "; result array provided: " + Arrays.toString(c.shape()));
        }
        if (a.rows() != c.rows()) {
            throw new IllegalArgumentException("A rows must equal C rows. MMul attempt: " + Arrays.toString(a.shape())
                            + "x" + Arrays.toString(b.shape()) + "; result array provided: "
                            + Arrays.toString(c.shape()));
        }

        if (a.columns() > Integer.MAX_VALUE || a.rows() > Integer.MAX_VALUE)
            throw new ND4JArraySizeException();

        if (b.columns() > Integer.MAX_VALUE || b.rows() > Integer.MAX_VALUE)
            throw new ND4JArraySizeException();

        if (c.columns() > Integer.MAX_VALUE || c.rows() > Integer.MAX_VALUE)
            throw new ND4JArraySizeException();


        if (Nd4j.allowsSpecifyOrdering()) {
            if (a.ordering() == b.ordering()) {
                //both will be same ordering for cblas
                this.ordering = a.ordering();
                //automatically assume fortran ordering
                //multiple backends force us to be
                //in fortran ordering only
                this.a = copyIfNeccessary(a);
                this.b = copyIfNeccessary(b);
                this.c = c;
                if (ordering == 'c') {
                    this.m = (int) c.columns();
                    this.n = (int) c.rows();
                    this.k = (int) a.columns();
                } else {
                    this.m = (int) c.rows();
                    this.n = (int) c.columns();
                    this.k = (int) b.columns();
                }

                this.lda = (int) a.rows();
                this.ldb = (int) b.rows();
                this.ldc = (int) c.rows();

                this.transA = 'N';
                this.transB = 'N';
            } else {
                //automatically assume fortran ordering
                //multiple backends force us to be
                //in fortran ordering only
                this.a = copyIfNeccessary(a);
                this.b = b.dup(a.ordering());
                this.c = c;

                this.m = (int) c.rows();
                this.n = (int) c.columns();
                this.k = (int) a.columns();

                this.ordering = a.ordering();

                this.lda = (int) a.rows();
                this.ldb = (int) b.rows();
                this.ldc = (int) c.rows();

                this.transA = 'N';
                this.transB = 'N';
            }


        } else {
            //automatically assume fortran ordering
            //multiple backends force us to be
            //in fortran ordering only
            this.a = copyIfNeccessary(a);
            this.b = copyIfNeccessary(b);
            this.c = c;

            this.m = (int) c.rows();
            this.n = (int) c.columns();
            this.k = (int) a.columns();

            //always fortran ordering
            this.lda = (int) (this.a.ordering() == 'f' ? this.a.rows() : this.a.columns()); //Leading dimension of a, as declared. But swap if 'c' order
            this.ldb = (int) (this.b.ordering() == 'f' ? this.b.rows() : this.b.columns()); //Leading dimension of b, as declared. But swap if 'c' order
            this.ldc = (int) c.rows();

            this.transA = (this.a.ordering() == 'c' ? 'T' : 'N');
            this.transB = (this.b.ordering() == 'c' ? 'T' : 'N');

        }

        ///validate();
    }

    public GemmParams(INDArray a, INDArray b, INDArray c, boolean transposeA, boolean transposeB) {
        this(transposeA ? a.transpose() : a, transposeB ? b.transpose() : b, c);
    }



    private INDArray copyIfNeccessary(INDArray arr) {
        //See also: Shape.toMmulCompatible - want same conditions here and there
        //Check if matrix values are contiguous in memory. If not: dup
        //Contiguous for c if: stride[0] == shape[1] and stride[1] = 1
        //Contiguous for f if: stride[0] == 1 and stride[1] == shape[0]
        if (!Nd4j.allowsSpecifyOrdering() && arr.ordering() == 'c'
                && (arr.stride(0) != arr.size(1) || arr.stride(1) != 1))
            return arr.dup();
        else if (arr.ordering() == 'f' && (arr.stride(0) != 1 || arr.stride(1) != arr.size(0)))
            return arr.dup();
        else if (arr.elementWiseStride() < 0)
            return arr.dup();
        return arr;
    }
}

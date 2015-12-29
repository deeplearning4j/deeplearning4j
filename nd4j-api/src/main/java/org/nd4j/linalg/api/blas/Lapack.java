package org.nd4j.linalg.api.blas;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Lapack interface
 *
 * @author Adam Gibson
 */
public interface Lapack {

    // LU decomoposition of a general matrix

    /**
     * LU decomposiiton of a matrix
     * @param M
     * @param N
     * @param A
     * @param lda
     * @param IPIV
     * @param INFO
     */
    void getrf(int M, int N, INDArray A, int lda, int[] IPIV, int INFO);

    // generate inverse of a matrix given its LU decomposition

    /**
     * Generate inverse ggiven LU decomp
     * @param N
     * @param A
     * @param lda
     * @param IPIV
     * @param WORK
     * @param lwork
     * @param INFO
     */
    void getri(int N, INDArray A, int lda, int[] IPIV, INDArray WORK, int lwork, int INFO);

}

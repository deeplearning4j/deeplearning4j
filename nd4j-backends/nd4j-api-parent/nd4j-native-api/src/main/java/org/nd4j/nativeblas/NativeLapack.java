package org.nd4j.nativeblas;


/**
 * Created by agibsonccc on 2/20/16.
 */
public class NativeLapack {

    public NativeLapack() {}
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
    public native void dgetrf(long[] extraPointers, int M, int N, long A, int lda, int[] IPIV, int INFO);

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
    public native void dgetri(long[] extraPointers, int N, long A, int lda, int[] IPIV, long WORK, int lwork, int INFO);

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
    public native void sgetrf(long[] extraPointers, int M, int N, long A, int lda, int[] IPIV, int INFO);

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
    public native void sgetri(long[] extraPointers, int N, long A, int lda, int[] IPIV, long WORK, int lwork, int INFO);
}

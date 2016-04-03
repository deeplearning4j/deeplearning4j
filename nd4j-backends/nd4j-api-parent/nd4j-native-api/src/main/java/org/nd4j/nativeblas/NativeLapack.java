package org.nd4j.nativeblas;


import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Platform;

/**
 * Created by agibsonccc on 2/20/16.
 *
 * the preload="libnd4j" is there because MinGW puts a "lib" in front of the filename for the DLL, but at load time,
 * we are using the default Windows platform naming scheme, which doesn't put "lib" in front, so that's a bit of a hack,
 * but easier than forcing MinGW into not renaming the library name -- @saudet on 3/21/16
 */
@Platform(include="NativeLapack.h",preload = "libnd4j", link = "nd4j")
public class NativeLapack extends Pointer {
    public NativeLapack() {
    }
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
    public native void dgetrf(long[] extraPointers,int M, int N, long A, int lda, int[] IPIV, int INFO);

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
    public native void dgetri(long[] extraPointers,int N, long A, int lda, int[] IPIV, long WORK, int lwork, int INFO);

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
    public native void sgetrf(long[] extraPointers,int M, int N, long A, int lda, int[] IPIV, int INFO);

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
    public native void sgetri(long[] extraPointers,int N, long A, int lda, int[] IPIV, long WORK, int lwork, int INFO);
}

package org.nd4j.linalg.api.blas.impl;

import org.nd4j.linalg.api.blas.Lapack;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Base lapack define float and oduble vversions.
 *
 * @author Adam Gibson
 */
public  abstract  class BaseLapack implements Lapack {
    @Override
    public void getrf(int M, int N, INDArray A, int lda, int[] IPIV, int INFO) {
        if(A.data().dataType() == DataBuffer.Type.DOUBLE) {
            dgetrf(M,N,A.data().asDouble(),lda,IPIV,INFO);
        }
        else {
            sgetrf(M,N,A.data().asFloat(),lda,IPIV,INFO);
        }
    }

    @Override
    public void getri(int N, INDArray A, int lda, int[] IPIV, INDArray WORK, int lwork, int INFO) {
        if(A.data().dataType() == DataBuffer.Type.DOUBLE) {
            dgetri(N,A.data().asDouble(),lda,IPIV, WORK.data().asDouble(),lwork,INFO);
        }
        else {
            sgetri(N,A.data().asFloat(),lda,IPIV, WORK.data().asFloat(),lwork,INFO);

        }
    }

    // LU decomoposition of a general matrix
    public abstract  void dgetrf(int M, int N, double[] A, int lda, int[] IPIV, int INFO);

    // generate inverse of a matrix given its LU decomposition
    public abstract void dgetri(int N, double[] A, int lda, int[] IPIV, double[] WORK, int lwork, int INFO);

    // LU decomoposition of a general matrix
    public abstract  void sgetrf(int M, int N, float[] A, int lda, int[] IPIV, int INFO);

    // generate inverse of a matrix given its LU decomposition
    public abstract void sgetri(int N, float[] A, int lda, int[] IPIV, float[] WORK, int lwork, int INFO);

}

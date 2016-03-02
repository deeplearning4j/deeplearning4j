package org.nd4j.linalg.jcublas.blas;

import org.nd4j.linalg.api.blas.impl.BaseLapack;

/**
 * Created by agibsoncccc on 12/6/15.
 */
public class JcublasLapack extends BaseLapack {
    @Override
    public void dgetrf(int M, int N, double[] A, int lda, int[] IPIV, int INFO) {
         throw new UnsupportedOperationException();
    }

    @Override
    public void dgetri(int N, double[] A, int lda, int[] IPIV, double[] WORK, int lwork, int INFO) {
        throw new UnsupportedOperationException();

    }

    @Override
    public void sgetrf(int M, int N, float[] A, int lda, int[] IPIV, int INFO) {
        throw new UnsupportedOperationException();

    }

    @Override
    public void sgetri(int N, float[] A, int lda, int[] IPIV, float[] WORK, int lwork, int INFO) {
        throw new UnsupportedOperationException();

    }


}

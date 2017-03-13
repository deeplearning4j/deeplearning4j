package org.nd4j.linalg.api.blas.impl;

import org.nd4j.linalg.api.blas.Lapack;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by audrey on 3/13/17.
 */
public class SparseBaseLapack implements Lapack {
    @Override
    public INDArray getrf(INDArray A) {
        return null;
    }

    @Override
    public void sgesvd(INDArray A, INDArray S, INDArray U, INDArray VT) {

    }

    @Override
    public INDArray getPFactor(int M, INDArray ipiv) {
        return null;
    }

    @Override
    public INDArray getLFactor(INDArray A) {
        return null;
    }

    @Override
    public INDArray getUFactor(INDArray A) {
        return null;
    }

    @Override
    public void getri(int N, INDArray A, int lda, int[] IPIV, INDArray WORK, int lwork, int INFO) {

    }
}

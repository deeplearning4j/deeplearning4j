package org.nd4j.linalg.cpu.blas;

import com.github.fommil.netlib.LAPACK;
import org.nd4j.linalg.api.blas.impl.BaseLapack;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.CBLAS;
import org.netlib.util.intW;

/**
 * CPU lapack implementation
 */
public class CpuLapack extends BaseLapack {

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

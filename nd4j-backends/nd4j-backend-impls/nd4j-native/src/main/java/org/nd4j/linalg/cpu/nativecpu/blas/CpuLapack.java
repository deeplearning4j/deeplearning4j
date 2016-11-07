package org.nd4j.linalg.cpu.nativecpu.blas;

import org.nd4j.linalg.api.blas.impl.BaseLapack;
import org.nd4j.linalg.api.ndarray.INDArray;

import static org.bytedeco.javacpp.openblas.*;

/**
 * CPU lapack implementation
 */
public class CpuLapack extends BaseLapack {

    @Override
    public void sgetrf(int M, int N, INDArray A, int lda, INDArray IPIV, INDArray INFO) {
	
	//public static native int LAPACKE_sgetrf( int matrix_layout, int m, int n,
        //                   FloatPointer a, int lda, IntPointer ipiv );

	int status = LAPACKE_sgetrf(LAPACK_ROW_MAJOR, M, N, A.data().asNioFloat(), lda, IPIV.data().asNioInt() ) ;

    }


    /**
     * Generate inverse ggiven LU decomp
     *
     * @param N
     * @param A
     * @param lda
     * @param IPIV
     * @param WORK
     * @param lwork
     * @param INFO
     */
    @Override
    public void getri(int N, INDArray A, int lda, int[] IPIV, INDArray WORK, int lwork, int INFO) {

    }
}

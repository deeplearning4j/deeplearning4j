package org.nd4j.linalg.cpu.nativecpu.blas;

import org.nd4j.linalg.api.blas.impl.BaseLapack;
import org.nd4j.linalg.api.ndarray.INDArray;

import static org.bytedeco.javacpp.openblas.*;

/**
 * CPU lapack implementation
 */
public class CpuLapack extends BaseLapack {
    protected static int getColumnOrder( INDArray A ) {
	return A.ordering() == 'f' ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR ;
    }
    protected static int getLda( INDArray A ) {
	return A.ordering() == 'f' ? A.rows() : A.columns() ;
    }

    @Override
    public void sgetrf(int M, int N, INDArray A, INDArray IPIV, INDArray INFO) {
	
	int status = org.bytedeco.javacpp.openblas.LAPACKE_sgetrf(
			getColumnOrder(A), M, N, A.data().asNioFloat(), getLda(A), IPIV.data().asNioInt() ) ;
    }

    @Override
    public void dgetrf(int M, int N, INDArray A, INDArray IPIV, INDArray INFO) {
	
	int status = org.bytedeco.javacpp.openblas.LAPACKE_dgetrf(
			getColumnOrder(A), M, N, A.data().asNioDouble(), getLda(A), IPIV.data().asNioInt() ) ;
    }

    @Override
    public void sgesvd( byte jobu, byte jobvt, int M, int N, INDArray A, INDArray S, INDArray U, INDArray VT, INDArray INFO ) {
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

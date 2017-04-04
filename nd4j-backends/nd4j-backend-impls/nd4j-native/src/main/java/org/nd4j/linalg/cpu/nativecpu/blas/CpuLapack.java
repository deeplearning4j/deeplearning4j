package org.nd4j.linalg.cpu.nativecpu.blas;

import org.nd4j.linalg.api.blas.impl.BaseLapack;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;

import static org.bytedeco.javacpp.openblas.*;
import org.nd4j.linalg.api.blas.BlasException ;

/**
 * CPU lapack implementation
 */
public class CpuLapack extends BaseLapack {
    protected static int getColumnOrder(INDArray A) {
        return A.ordering() == 'f' ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;
    }

    protected static int getLda(INDArray A) {
        return A.ordering() == 'f' ? A.rows() : A.columns();
    }
//=========================    
// L U DECOMP
    @Override
    public void sgetrf(int M, int N, INDArray A, INDArray IPIV, INDArray INFO) {
        int status = LAPACKE_sgetrf(getColumnOrder(A), M, N, 
            (FloatPointer)A.data().addressPointer(), 
            getLda(A), (IntPointer)IPIV.data().addressPointer()
            );
        if( status != 0 ) {
            throw new BlasException( "Failed to execute sgetrf", status ) ;
        }
    }

    @Override
    public void dgetrf(int M, int N, INDArray A, INDArray IPIV, INDArray INFO) {
        int status = LAPACKE_dgetrf(getColumnOrder(A), M, N, (DoublePointer)A.data().addressPointer(), 
            getLda(A), (IntPointer)IPIV.data().addressPointer()
            );
        if( status != 0 ) {
            throw new BlasException( "Failed to execute dgetrf", status ) ;
        }
    }

//=========================    
// Q R DECOMP
    @Override
    public void sgeqrf(int M, int N, INDArray A, INDArray R, INDArray INFO) {
        INDArray tau = Nd4j.create( N ) ;

        int status = LAPACKE_sgeqrf(getColumnOrder(A), M, N, 
             (FloatPointer)A.data().addressPointer(), getLda(A),
             (FloatPointer)tau.data().addressPointer()
             );
        if( status != 0 ) {
            throw new BlasException( "Failed to execute sgeqrf", status ) ;
        }

        // Copy R ( upper part of Q ) into result
        if( R != null ) {
            for( int ro=0 ; ro<M ; ro++ ) {
                for( int c=ro ; c<N ; c++ ) {
                    R.putScalar( ro, c, A.getDouble(ro,c) ) ;
                }
            }
        }

        status = LAPACKE_sorgqr( getColumnOrder(A), M, N, N, 
             (FloatPointer)A.data().addressPointer(), getLda(A),
             (FloatPointer)tau.data().addressPointer()
             );
        if( status != 0 ) {
            throw new BlasException( "Failed to execute sorgqr", status ) ;
        }
    }

    @Override
    public void dgeqrf(int M, int N, INDArray A, INDArray R, INDArray INFO)  {
        INDArray tau = Nd4j.create( N ) ;

        int status = LAPACKE_dgeqrf(getColumnOrder(A), M, N,
             (DoublePointer)A.data().addressPointer(), getLda(A),
             (DoublePointer)tau.data().addressPointer()
             );
        if( status != 0 ) {
            throw new BlasException( "Failed to execute dgeqrf", status ) ;
        }

        // Copy R ( upper part of Q ) into result
        if( R != null ) {
            for( int ro=0 ; ro<M ; ro++ ) {
                for( int c=ro ; c<N ; c++ ) {
                    R.putScalar( ro, c, A.getDouble(ro,c) ) ;
                }
            }
        }

        status = LAPACKE_dorgqr( getColumnOrder(A), M, N, N, 
             (DoublePointer)A.data().addressPointer(), getLda(A),
             (DoublePointer)tau.data().addressPointer()
             );
        if( status != 0 ) {
            throw new BlasException( "Failed to execute dorgqr", status ) ;
        }
    }


//=========================    
// CHOLESKY DECOMP
    @Override
    public void spotrf(byte uplo, int N, INDArray A, INDArray INFO) {
        int status = LAPACKE_spotrf(getColumnOrder(A), uplo, N, 
                        (FloatPointer)A.data().addressPointer(), getLda(A) );
        if( status != 0 ) {
            throw new BlasException( "Failed to execute spotrf", status ) ;
        }
        if( uplo == 'U' ) {
            for( int ro=1 ; ro<N ; ro++ ) {
                for( int c=0 ; c<ro ; c++ ) {
                    A.putScalar( ro, c, 0 ) ;
                }
            }
        } else {
            for( int c=1 ; c<N ; c++ ) {
                for( int ro=0 ; ro<c ; ro++ ) {
                    A.putScalar( ro, c, 0 ) ;
                }
            }
        }
    }

    @Override
    public void dpotrf(byte uplo, int N, INDArray A, INDArray INFO) {
        int status = LAPACKE_dpotrf(getColumnOrder(A), uplo, N, 
                    (DoublePointer)A.data().addressPointer(), getLda(A) );
        if( status != 0 ) {
            throw new BlasException( "Failed to execute dpotrf", status ) ;
        }
        if( uplo == 'U' ) {
            for( int ro=1 ; ro<N ; ro++ ) {
                for( int c=0 ; c<ro ; c++ ) {
                    A.putScalar( ro, c, 0 ) ;
                }
            }
            //A = A.transpose() ;
        } else {
            for( int c=1 ; c<N ; c++ ) {
                for( int ro=0 ; ro<c ; ro++ ) {
                    A.putScalar( ro, c, 0 ) ;
                }
            }
        }
    }



//=========================    
// U S V' DECOMP  (aka SVD)
    @Override
    public void sgesvd(byte jobu, byte jobvt, int M, int N, INDArray A, INDArray S, INDArray U, INDArray VT,
                    INDArray INFO) {
        INDArray superb = Nd4j.create( M < N ? M : N ) ;
        int status = LAPACKE_sgesvd(getColumnOrder(A), jobu, jobvt, M, N, 
                        (FloatPointer)A.data().addressPointer(), getLda(A),
                        (FloatPointer)S.data().addressPointer(), 
                        U == null ? null : (FloatPointer)U.data().addressPointer(), U == null ? 1 : getLda(U),
                        VT == null ? null : (FloatPointer)VT.data().addressPointer(), VT == null ? 1 : getLda(VT), 
                        (FloatPointer)superb.data().addressPointer() 
                        );
        if( status != 0 ) {
            throw new BlasException( "Failed to execute sgesvd", status ) ;
        }
    }

    @Override
    public void dgesvd(byte jobu, byte jobvt, int M, int N, INDArray A, INDArray S, INDArray U, INDArray VT,
                    INDArray INFO) {
        INDArray superb = Nd4j.create( M < N ? M : N ) ;
        int status = LAPACKE_dgesvd(getColumnOrder(A), jobu, jobvt, M, N, 
                        (DoublePointer)A.data().addressPointer(), getLda(A),
                        (DoublePointer)S.data().addressPointer(), 
                        U == null ? null : (DoublePointer)U.data().addressPointer(), U == null ? 1 : getLda(U),
                        VT == null ? null : (DoublePointer)VT.data().addressPointer(), VT == null ? 1 : getLda(VT), 
                        (DoublePointer)superb.data().addressPointer() 
                        ) ;
        if( status != 0 ) {
            throw new BlasException( "Failed to execute dgesvd", status ) ;
        }
    }

    /**
     * Generate inverse given LU decomp
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

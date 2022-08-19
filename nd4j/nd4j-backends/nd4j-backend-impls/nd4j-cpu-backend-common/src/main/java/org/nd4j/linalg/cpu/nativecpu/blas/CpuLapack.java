/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.cpu.nativecpu.blas;

import org.nd4j.linalg.api.blas.impl.BaseLapack;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JArraySizeException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;

import org.nd4j.linalg.api.blas.BlasException ;




public class CpuLapack extends BaseLapack {

    public static final int OPENBLAS_OS_WINNT = 1;
    public static final int OPENBLAS_ARCH_X86_64 = 1;
    public static final int OPENBLAS_C_GCC = 1;
    public static final int OPENBLAS___64BIT__ = 1;
    public static final int OPENBLAS_HAVE_C11 = 1;
    public static final int OPENBLAS_NEEDBUNDERSCORE = 1;
    public static final int OPENBLAS_L1_DATA_SIZE = 32768;
    public static final int OPENBLAS_L1_DATA_LINESIZE = 64;
    public static final int OPENBLAS_L2_SIZE = 262144;
    public static final int OPENBLAS_L2_LINESIZE = 64;
    public static final int OPENBLAS_DTB_DEFAULT_ENTRIES = 64;
    public static final int OPENBLAS_DTB_SIZE = 4096;
    public static final String OPENBLAS_CHAR_CORENAME = "NEHALEM";
    public static final int OPENBLAS_SLOCAL_BUFFER_SIZE = 65536;
    public static final int OPENBLAS_DLOCAL_BUFFER_SIZE = 32768;
    public static final int OPENBLAS_CLOCAL_BUFFER_SIZE = 65536;
    public static final int OPENBLAS_ZLOCAL_BUFFER_SIZE = 32768;
    public static final int OPENBLAS_GEMM_MULTITHREAD_THRESHOLD = 4;
    public static final String OPENBLAS_VERSION = " OpenBLAS 0.3.19 ";
    public static final int OPENBLAS_SEQUENTIAL = 0;
    public static final int OPENBLAS_THREAD = 1;
    public static final int OPENBLAS_OPENMP = 2;
    public static final int CblasRowMajor = 101;
    public static final int CblasColMajor = 102;
    public static final int CblasNoTrans = 111;
    public static final int CblasTrans = 112;
    public static final int CblasConjTrans = 113;
    public static final int CblasConjNoTrans = 114;
    public static final int CblasUpper = 121;
    public static final int CblasLower = 122;
    public static final int CblasNonUnit = 131;
    public static final int CblasUnit = 132;
    public static final int CblasLeft = 141;
    public static final int CblasRight = 142;
    public static final int LAPACK_ROW_MAJOR = 101;
    public static final int LAPACK_COL_MAJOR = 102;
    public static final int LAPACK_WORK_MEMORY_ERROR = -1010;
    public static final int LAPACK_TRANSPOSE_MEMORY_ERROR = -1011;


    private BLASDelegator blasDelegator;

    protected static int getColumnOrder(INDArray A) {
        return A.ordering() == 'f' ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;
    }

    protected static int getLda(INDArray A) {
        if (A.rows() > Integer.MAX_VALUE || A.columns() > Integer.MAX_VALUE) {
            throw new ND4JArraySizeException();
        }
        return A.ordering() == 'f' ? (int) A.rows() : (int) A.columns();
    }
//=========================
// L U DECOMP
    @Override
    public void sgetrf(int M, int N, INDArray A, INDArray IPIV, INDArray INFO) {
        int status = blasDelegator.LAPACKE_sgetrf(getColumnOrder(A), M, N,
            (FloatPointer)A.data().addressPointer(),
            getLda(A), (IntPointer)IPIV.data().addressPointer()
            );
        if( status < 0 ) {
            throw new BlasException( "Failed to execute sgetrf", status ) ;
        }
    }

    @Override
    public void dgetrf(int M, int N, INDArray A, INDArray IPIV, INDArray INFO) {
        int status = blasDelegator.LAPACKE_dgetrf(getColumnOrder(A), M, N, (DoublePointer)A.data().addressPointer(),
            getLda(A), (IntPointer)IPIV.data().addressPointer()
            );
        if( status < 0 ) {
            throw new BlasException( "Failed to execute dgetrf", status ) ;
        }
    }

//=========================
// Q R DECOMP
    @Override
    public void sgeqrf(int M, int N, INDArray A, INDArray R, INDArray INFO) {
        INDArray tau = Nd4j.create(DataType.FLOAT, N ) ;

        int status = blasDelegator.LAPACKE_sgeqrf(getColumnOrder(A), M, N,
             (FloatPointer)A.data().addressPointer(), getLda(A),
             (FloatPointer)tau.data().addressPointer()
             );
        if( status != 0 ) {
            throw new BlasException( "Failed to execute sgeqrf", status ) ;
        }

        // Copy R ( upper part of Q ) into result
        if( R != null ) {
            R.assign( A.get( NDArrayIndex.interval( 0, A.columns() ), NDArrayIndex.all() ) ) ;
			INDArrayIndex ix[] = new INDArrayIndex[ 2 ] ;

			for( int i=1 ; i<Math.min( A.rows(), A.columns() ) ; i++ ) {
				ix[0] = NDArrayIndex.point( i ) ;
				ix[1] = NDArrayIndex.interval( 0, i ) ;
				R.put(ix, 0) ;
			}
        }

        status = blasDelegator.LAPACKE_sorgqr( getColumnOrder(A), M, N, N,
             (FloatPointer)A.data().addressPointer(), getLda(A),
             (FloatPointer)tau.data().addressPointer()
             );
        if( status != 0 ) {
            throw new BlasException( "Failed to execute sorgqr", status ) ;
        }
    }

    @Override
    public void dgeqrf(int M, int N, INDArray A, INDArray R, INDArray INFO)  {
        INDArray tau = Nd4j.create(DataType.DOUBLE, N ) ;

        int status = blasDelegator.LAPACKE_dgeqrf(getColumnOrder(A), M, N,
             (DoublePointer)A.data().addressPointer(), getLda(A),
             (DoublePointer)tau.data().addressPointer()
             );
        if( status != 0 ) {
            throw new BlasException( "Failed to execute dgeqrf", status ) ;
        }

        // Copy R ( upper part of Q ) into result
        if( R != null ) {
            R.assign( A.get( NDArrayIndex.interval( 0, A.columns() ), NDArrayIndex.all() ) ) ;
			INDArrayIndex ix[] = new INDArrayIndex[ 2 ] ;

			for( int i=1 ; i<Math.min( A.rows(), A.columns() ) ; i++ ) {
				ix[0] = NDArrayIndex.point( i ) ;
				ix[1] = NDArrayIndex.interval( 0, i ) ;
				R.put(ix, 0) ;
			}
        }

        status = blasDelegator.LAPACKE_dorgqr( getColumnOrder(A), M, N, N,
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
        int status = blasDelegator.LAPACKE_spotrf(getColumnOrder(A), uplo, N,
                        (FloatPointer)A.data().addressPointer(), getLda(A) );
        if( status != 0 ) {
            throw new BlasException( "Failed to execute spotrf", status ) ;
        }
        if( uplo == 'U' ) {
			INDArrayIndex ix[] = new INDArrayIndex[ 2 ] ;
			for( int i=1 ; i<Math.min( A.rows(), A.columns() ) ; i++ ) {
				ix[0] = NDArrayIndex.point( i ) ;
				ix[1] = NDArrayIndex.interval( 0, i ) ;
				A.put(ix, 0) ;
			}
        } else {
            INDArrayIndex ix[] = new INDArrayIndex[ 2 ] ;
            for( int i=0 ; i<Math.min( A.rows(), A.columns()-1 ) ; i++ ) {
                ix[0] = NDArrayIndex.point( i ) ;
                ix[1] = NDArrayIndex.interval( i+1, A.columns() ) ;
                A.put(ix, 0) ;
            }
        }
    }

    @Override
    public void dpotrf(byte uplo, int N, INDArray A, INDArray INFO) {
        int status = blasDelegator.LAPACKE_dpotrf(getColumnOrder(A), uplo, N,
                    (DoublePointer)A.data().addressPointer(), getLda(A) );
        if( status != 0 ) {
            throw new BlasException( "Failed to execute dpotrf", status ) ;
        }
        if( uplo == 'U' ) {
			INDArrayIndex ix[] = new INDArrayIndex[ 2 ] ;
			for( int i=1 ; i<Math.min( A.rows(), A.columns() ) ; i++ ) {
				ix[0] = NDArrayIndex.point( i ) ;
				ix[1] = NDArrayIndex.interval( 0, i ) ;
				A.put(ix, 0) ;
			}
        } else {
            INDArrayIndex ix[] = new INDArrayIndex[ 2 ] ;
            for( int i=0 ; i<Math.min( A.rows(), A.columns()-1 ) ; i++ ) {
                ix[0] = NDArrayIndex.point( i ) ;
                ix[1] = NDArrayIndex.interval( i+1, A.columns() ) ;
                A.put(ix, 0) ;
            }
        }
    }



//=========================
// U S V' DECOMP  (aka SVD)
    @Override
    public void sgesvd(byte jobu, byte jobvt, int M, int N, INDArray A, INDArray S, INDArray U, INDArray VT,
                    INDArray INFO) {
        INDArray superb = Nd4j.create(DataType.FLOAT, M < N ? M : N ) ;
        int status = blasDelegator.LAPACKE_sgesvd(getColumnOrder(A), jobu, jobvt, M, N,
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
        INDArray superb = Nd4j.create(DataType.DOUBLE, M < N ? M : N ) ;
        int status = blasDelegator.LAPACKE_dgesvd(getColumnOrder(A), jobu, jobvt, M, N,
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


//=========================
// syev EigenValue/Vectors
//
    @Override
    public int ssyev( char jobz, char uplo, int N, INDArray A, INDArray R ) {
	FloatPointer fp = new FloatPointer(1) ;
	int status = blasDelegator.LAPACKE_ssyev_work( getColumnOrder(A), (byte)jobz, (byte)uplo,
					N, (FloatPointer)A.data().addressPointer(), getLda(A),
					(FloatPointer)R.data().addressPointer(), fp, -1 ) ;
	if( status == 0 ) {
		int lwork = (int)fp.get() ;
		INDArray work = Nd4j.createArrayFromShapeBuffer(Nd4j.getDataBufferFactory().createFloat(lwork),
		                Nd4j.getShapeInfoProvider().createShapeInformation(new long[] {lwork}, A.dataType()).getFirst());

		status = blasDelegator.LAPACKE_ssyev( getColumnOrder(A), (byte)jobz, (byte)uplo, N,
		            (FloatPointer)A.data().addressPointer(), getLda(A),
			    (FloatPointer)work.data().addressPointer() ) ;
		if( status == 0 ) {
			R.assign(work.get(NDArrayIndex.interval(0,N))) ;
		}
	}
	return status ;
    }


    public int dsyev( char jobz, char uplo, int N, INDArray A, INDArray R ) {

	DoublePointer dp = new DoublePointer(1) ;
	int status = blasDelegator.LAPACKE_dsyev_work( getColumnOrder(A), (byte)jobz, (byte)uplo,
					N, (DoublePointer)A.data().addressPointer(), getLda(A),
					(DoublePointer)R.data().addressPointer(), dp, -1 ) ;
	if( status == 0 ) {
		int lwork = (int)dp.get() ;
		INDArray work = Nd4j.createArrayFromShapeBuffer(Nd4j.getDataBufferFactory().createDouble(lwork),
		                Nd4j.getShapeInfoProvider().createShapeInformation(new long[] {lwork}, A.dataType()).getFirst());

		status = blasDelegator.LAPACKE_dsyev( getColumnOrder(A), (byte)jobz, (byte)uplo, N,
		            (DoublePointer)A.data().addressPointer(), getLda(A),
			    (DoublePointer)work.data().addressPointer() ) ;

		if( status == 0 ) {
			R.assign( work.get( NDArrayIndex.interval(0,N) ) ) ;
		}
	}
	return status ;
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

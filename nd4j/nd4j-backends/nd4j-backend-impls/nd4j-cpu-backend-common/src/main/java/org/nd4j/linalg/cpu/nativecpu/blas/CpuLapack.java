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

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.nd4j.linalg.api.blas.BlasException;
import org.nd4j.linalg.api.blas.impl.BaseLapack;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JArraySizeException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;


public class CpuLapack extends BaseLapack {
    public static final int LAPACK_ROW_MAJOR = 101;
    public static final int LAPACK_COL_MAJOR = 102;

    protected static int getColumnOrder(INDArray A) {
        return A.ordering() == 'f' ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;
    }

    protected static int getLda(INDArray A) {
        if (A.rows() > Integer.MAX_VALUE || A.columns() > Integer.MAX_VALUE) {
            throw new ND4JArraySizeException();
        }
        return A.ordering() == 'f' ? (int) A.rows() : (int) A.columns();
    }

    private FloatPointer floatPtr(INDArray arr) {
        return ((FloatPointer) arr.data().addressPointer()).getPointer(arr.offset());
    }

    private DoublePointer doublePtr(INDArray arr) {
        return ((DoublePointer) arr.data().addressPointer()).getPointer(arr.offset());
    }

    private IntPointer intPtr(INDArray arr) {
        return ((IntPointer) arr.data().addressPointer()).getPointer(arr.offset());
    }

    //=========================
// L U DECOMP
    @Override
    public void sgetrf(int M, int N, INDArray A, INDArray IPIV, INDArray INFO) {
        int status = Nd4j.getBlasLapackDelegator().LAPACKE_sgetrf(getColumnOrder(A), M, N,
                floatPtr(A),
                getLda(A), intPtr(IPIV)
        );
        if( status < 0 ) {
            throw new BlasException( "Failed to execute sgetrf", status ) ;
        }
    }

    @Override
    public void dgetrf(int M, int N, INDArray A, INDArray IPIV, INDArray INFO) {
        int status = Nd4j.getBlasLapackDelegator().LAPACKE_dgetrf(getColumnOrder(A), M, N, doublePtr(A),
                getLda(A), intPtr(IPIV)
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

        int status = Nd4j.getBlasLapackDelegator().LAPACKE_sgeqrf(getColumnOrder(A), M, N,
                floatPtr(A), getLda(A),
                floatPtr(tau)
        );
        if( status != 0 ) {
            throw new BlasException( "Failed to execute sgeqrf", status ) ;
        }

        // Copy R (upper triangular part of A) into result using element-wise access
        if( R != null ) {
            for( int i = 0 ; i < N ; i++ ) {
                for( int j = 0 ; j < N ; j++ ) {
                    if( j >= i ) {
                        R.putScalar(i, j, A.getDouble(i, j));
                    } else {
                        R.putScalar(i, j, 0.0);
                    }
                }
            }
        }

        status = Nd4j.getBlasLapackDelegator().LAPACKE_sorgqr( getColumnOrder(A), M, N, N,
                floatPtr(A), getLda(A),
                floatPtr(tau)
        );
        if( status != 0 ) {
            throw new BlasException( "Failed to execute sorgqr", status ) ;
        }
    }

    @Override
    public void dgeqrf(int M, int N, INDArray A, INDArray R, INDArray INFO)  {
        INDArray tau = Nd4j.create(DataType.DOUBLE, N ) ;

        int status = Nd4j.getBlasLapackDelegator().LAPACKE_dgeqrf(getColumnOrder(A), M, N,
                doublePtr(A), getLda(A),
                doublePtr(tau)
        );
        if( status != 0 ) {
            throw new BlasException( "Failed to execute dgeqrf", status ) ;
        }

        // Copy R (upper triangular part of A) into result using element-wise access
        if( R != null ) {
            for( int i = 0 ; i < N ; i++ ) {
                for( int j = 0 ; j < N ; j++ ) {
                    if( j >= i ) {
                        R.putScalar(i, j, A.getDouble(i, j));
                    } else {
                        R.putScalar(i, j, 0.0);
                    }
                }
            }
        }

        status = Nd4j.getBlasLapackDelegator().LAPACKE_dorgqr( getColumnOrder(A), M, N, N,
                doublePtr(A), getLda(A),
                doublePtr(tau)
        );
        if( status != 0 ) {
            throw new BlasException( "Failed to execute dorgqr", status ) ;
        }
    }


    //=========================
// CHOLESKY DECOMP
    @Override
    public void spotrf(byte uplo, int N, INDArray A, INDArray INFO) {
        int status = Nd4j.getBlasLapackDelegator().LAPACKE_spotrf(getColumnOrder(A), uplo, N,
                floatPtr(A), getLda(A) );
        if( status != 0 ) {
            throw new BlasException( "Failed to execute spotrf", status ) ;
        }
        // Zero the non-factor triangle using direct element access
        if( uplo == 'U' ) {
            for( int i = 1 ; i < N ; i++ ) {
                for( int j = 0 ; j < i ; j++ ) {
                    A.putScalar(i, j, 0.0);
                }
            }
        } else {
            for( int i = 0 ; i < N ; i++ ) {
                for( int j = i + 1 ; j < N ; j++ ) {
                    A.putScalar(i, j, 0.0);
                }
            }
        }
    }

    @Override
    public void dpotrf(byte uplo, int N, INDArray A, INDArray INFO) {
        int status = Nd4j.getBlasLapackDelegator().LAPACKE_dpotrf(getColumnOrder(A), uplo, N,
                doublePtr(A), getLda(A) );
        if( status != 0 ) {
            throw new BlasException( "Failed to execute dpotrf", status ) ;
        }
        // Zero the non-factor triangle using direct element access
        // LAPACKE modifies the raw buffer; NDArrayIndex zeroing can conflict with array strides
        if( uplo == 'U' ) {
            // Factor is in upper triangle; zero strictly lower triangle
            for( int i = 1 ; i < N ; i++ ) {
                for( int j = 0 ; j < i ; j++ ) {
                    A.putScalar(i, j, 0.0);
                }
            }
        } else {
            // Factor is in lower triangle; zero strictly upper triangle
            for( int i = 0 ; i < N ; i++ ) {
                for( int j = i + 1 ; j < N ; j++ ) {
                    A.putScalar(i, j, 0.0);
                }
            }
        }
    }



    //=========================
// U S V' DECOMP  (aka SVD)
    @Override
    public void sgesvd(byte jobu, byte jobvt, int M, int N, INDArray A, INDArray S, INDArray U, INDArray VT,
                       INDArray INFO) {
        INDArray superb = Nd4j.create(DataType.FLOAT, M < N ? M : N ) ;
        int status = Nd4j.getBlasLapackDelegator().LAPACKE_sgesvd(getColumnOrder(A), jobu, jobvt, M, N,
                floatPtr(A), getLda(A),
                floatPtr(S),
                U == null ? null : floatPtr(U), U == null ? 1 : getLda(U),
                VT == null ? null : floatPtr(VT), VT == null ? 1 : getLda(VT),
                floatPtr(superb)
        );
        if( status != 0 ) {
            throw new BlasException( "Failed to execute sgesvd", status ) ;
        }
    }

    @Override
    public void dgesvd(byte jobu, byte jobvt, int M, int N, INDArray A, INDArray S, INDArray U, INDArray VT,
                       INDArray INFO) {
        INDArray superb = Nd4j.create(DataType.DOUBLE, M < N ? M : N ) ;
        int status = Nd4j.getBlasLapackDelegator().LAPACKE_dgesvd(getColumnOrder(A), jobu, jobvt, M, N,
                doublePtr(A), getLda(A),
                doublePtr(S),
                U == null ? null : doublePtr(U), U == null ? 1 : getLda(U),
                VT == null ? null : doublePtr(VT), VT == null ? 1 : getLda(VT),
                doublePtr(superb)
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
        int status = Nd4j.getBlasLapackDelegator().LAPACKE_ssyev_work( getColumnOrder(A), (byte)jobz, (byte)uplo,
                N, floatPtr(A), getLda(A),
                floatPtr(R), fp, -1 ) ;
        if( status == 0 ) {
            int lwork = (int)fp.get() ;
            INDArray work = Nd4j.createArrayFromShapeBuffer(Nd4j.getDataBufferFactory().createFloat(lwork),
                    Nd4j.getShapeInfoProvider().createShapeInformation(new long[] {lwork}, A.dataType()).getFirst());

            status = Nd4j.getBlasLapackDelegator().LAPACKE_ssyev( getColumnOrder(A), (byte)jobz, (byte)uplo, N,
                    floatPtr(A), getLda(A),
                    floatPtr(work) ) ;
            if( status == 0 ) {
                R.assign(work.get(NDArrayIndex.interval(0,N))) ;
            }
        }
        return status ;
    }


    public int dsyev( char jobz, char uplo, int N, INDArray A, INDArray R ) {

        DoublePointer dp = new DoublePointer(1) ;
        int status = Nd4j.getBlasLapackDelegator().LAPACKE_dsyev_work( getColumnOrder(A), (byte)jobz, (byte)uplo,
                N, doublePtr(A), getLda(A),
                doublePtr(R), dp, -1 ) ;
        if( status == 0 ) {
            int lwork = (int)dp.get() ;
            INDArray work = Nd4j.createArrayFromShapeBuffer(Nd4j.getDataBufferFactory().createDouble(lwork),
                    Nd4j.getShapeInfoProvider().createShapeInformation(new long[] {lwork}, A.dataType()).getFirst());

            status = Nd4j.getBlasLapackDelegator().LAPACKE_dsyev( getColumnOrder(A), (byte)jobz, (byte)uplo, N,
                    doublePtr(A), getLda(A),
                    doublePtr(work) ) ;

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
        throw new UnsupportedOperationException();
    }
}

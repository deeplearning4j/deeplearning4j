/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.jcublas.blas;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.jita.allocator.pointers.cuda.cusolverDnHandle_t;
import org.nd4j.linalg.api.blas.BlasException;
import org.nd4j.linalg.api.blas.impl.BaseLapack;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.jcublas.CublasPointer;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import static org.bytedeco.javacpp.cuda.CUstream_st;
import static org.bytedeco.javacpp.cusolver.*;
import static org.bytedeco.javacpp.cublas.* ;


/**
 * JCublas lapack
 *
 * @author Adam Gibson
 * @author Richard Corbishley
 */
@Slf4j
public class JcublasLapack extends BaseLapack {

    private NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
    private Allocator allocator = AtomicAllocator.getInstance();

    @Override
    public void sgetrf(int M, int N, INDArray A, INDArray IPIV, INDArray INFO) {
        INDArray a = A;
        if (Nd4j.dataType() != DataBuffer.Type.FLOAT)
            log.warn("FLOAT getrf called in DOUBLE environment");

        if (A.ordering() == 'c')
            a = A.dup('f');


        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        // Get context for current thread
        CudaContext ctx = (CudaContext) allocator.getDeviceContext().getContext();

        // setup the solver handles for cuSolver calls
        cusolverDnHandle_t handle = ctx.getSolverHandle();
        cusolverDnContext solverDn = new cusolverDnContext(handle);

        // synchronized on the solver
        synchronized (handle) {
            int result = cusolverDnSetStream(new cusolverDnContext(handle), new CUstream_st(ctx.getOldStream()));
            if (result != 0)
                throw new BlasException("solverSetStream failed");

            // transfer the INDArray into GPU memory
            CublasPointer xAPointer = new CublasPointer(a, ctx);

            // this output - indicates how much memory we'll need for the real operation
            DataBuffer worksizeBuffer = Nd4j.getDataBufferFactory().createInt(1);

            int stat = cusolverDnSgetrf_bufferSize(solverDn, M, N, (FloatPointer) xAPointer.getDevicePointer(), M,
                            (IntPointer) worksizeBuffer.addressPointer() // we intentionally use host pointer here
            );

            if (stat != CUSOLVER_STATUS_SUCCESS) {
                throw new BlasException("cusolverDnSgetrf_bufferSize failed", stat);
            }

            int worksize = worksizeBuffer.getInt(0);
            // Now allocate memory for the workspace, the permutation matrix and a return code
            Pointer workspace = new Workspace(worksize * Nd4j.sizeOfDataType());

            // Do the actual LU decomp
            stat = cusolverDnSgetrf(solverDn, M, N, (FloatPointer) xAPointer.getDevicePointer(), M,
                            new CudaPointer(workspace).asFloatPointer(),
                            new CudaPointer(allocator.getPointer(IPIV, ctx)).asIntPointer(),
                            new CudaPointer(allocator.getPointer(INFO, ctx)).asIntPointer());

            // we do sync to make sure getrf is finished
            //ctx.syncOldStream();

            if (stat != CUSOLVER_STATUS_SUCCESS) {
                throw new BlasException("cusolverDnSgetrf failed", stat);
            }
        }
        allocator.registerAction(ctx, a);
        allocator.registerAction(ctx, INFO);
        allocator.registerAction(ctx, IPIV);

        if (a != A)
            A.assign(a);
    }



    @Override
    public void dgetrf(int M, int N, INDArray A, INDArray IPIV, INDArray INFO) {
        INDArray a = A;

        if (Nd4j.dataType() != DataBuffer.Type.DOUBLE)
            log.warn("FLOAT getrf called in FLOAT environment");

        if (A.ordering() == 'c')
            a = A.dup('f');

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        // Get context for current thread
        CudaContext ctx = (CudaContext) allocator.getDeviceContext().getContext();

        // setup the solver handles for cuSolver calls
        cusolverDnHandle_t handle = ctx.getSolverHandle();
        cusolverDnContext solverDn = new cusolverDnContext(handle);

        // synchronized on the solver
        synchronized (handle) {
            int result = cusolverDnSetStream(new cusolverDnContext(handle), new CUstream_st(ctx.getOldStream()));
            if (result != 0)
                throw new BlasException("solverSetStream failed");

            // transfer the INDArray into GPU memory
            CublasPointer xAPointer = new CublasPointer(a, ctx);

            // this output - indicates how much memory we'll need for the real operation
            DataBuffer worksizeBuffer = Nd4j.getDataBufferFactory().createInt(1);

            int stat = cusolverDnDgetrf_bufferSize(solverDn, M, N, (DoublePointer) xAPointer.getDevicePointer(), M,
                            (IntPointer) worksizeBuffer.addressPointer() // we intentionally use host pointer here
            );

            if (stat != CUSOLVER_STATUS_SUCCESS) {
                throw new BlasException("cusolverDnDgetrf_bufferSize failed", stat);
            }
            int worksize = worksizeBuffer.getInt(0);

            // Now allocate memory for the workspace, the permutation matrix and a return code
            Pointer workspace = new Workspace(worksize * Nd4j.sizeOfDataType());

            // Do the actual LU decomp
            stat = cusolverDnDgetrf(solverDn, M, N, (DoublePointer) xAPointer.getDevicePointer(), M,
                            new CudaPointer(workspace).asDoublePointer(),
                            new CudaPointer(allocator.getPointer(IPIV, ctx)).asIntPointer(),
                            new CudaPointer(allocator.getPointer(INFO, ctx)).asIntPointer());

            if (stat != CUSOLVER_STATUS_SUCCESS) {
                throw new BlasException("cusolverDnSgetrf failed", stat);
            }
        }
        allocator.registerAction(ctx, a);
        allocator.registerAction(ctx, INFO);
        allocator.registerAction(ctx, IPIV);

        if (a != A)
            A.assign(a);
    }


//=========================    
// Q R DECOMP
    @Override
    public void sgeqrf(int M, int N, INDArray A, INDArray R, INDArray INFO) {
        INDArray a = A;
        INDArray r = R;

        if (Nd4j.dataType() != DataBuffer.Type.FLOAT)
            log.warn("FLOAT getrf called in DOUBLE environment");

        if (A.ordering() == 'c') 
            a = A.dup('f');
        if ( R!=null && R.ordering() == 'c')
            r = R.dup('f');

        INDArray tau = Nd4j.createArrayFromShapeBuffer(Nd4j.getDataBufferFactory().createFloat(N),
                Nd4j.getShapeInfoProvider().createShapeInformation(new int[] {1, N}).getFirst());

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        // Get context for current thread
        CudaContext ctx = (CudaContext) allocator.getDeviceContext().getContext();

        // setup the solver handles for cuSolver calls
        cusolverDnHandle_t handle = ctx.getSolverHandle();
        cusolverDnContext solverDn = new cusolverDnContext(handle);

        // synchronized on the solver
        synchronized (handle) {
            int result = cusolverDnSetStream(new cusolverDnContext(handle), new CUstream_st(ctx.getOldStream()));
            if (result != 0)
                throw new IllegalStateException("solverSetStream failed");

            // transfer the INDArray into GPU memory
            CublasPointer xAPointer = new CublasPointer(a, ctx);
            CublasPointer xTauPointer = new CublasPointer(tau, ctx);

            // this output - indicates how much memory we'll need for the real operation
            DataBuffer worksizeBuffer = Nd4j.getDataBufferFactory().createInt(1);

            int stat = cusolverDnSgeqrf_bufferSize(solverDn, M, N, 
                        (FloatPointer) xAPointer.getDevicePointer(), M,
                        (IntPointer) worksizeBuffer.addressPointer() // we intentionally use host pointer here
            );


            if (stat != CUSOLVER_STATUS_SUCCESS) {
                throw new BlasException("cusolverDnSgeqrf_bufferSize failed", stat);
            }
            int worksize = worksizeBuffer.getInt(0);
            // Now allocate memory for the workspace, the permutation matrix and a return code
            Pointer workspace = new Workspace(worksize * Nd4j.sizeOfDataType());

            // Do the actual QR decomp
            stat = cusolverDnSgeqrf(solverDn, M, N, 
                            (FloatPointer) xAPointer.getDevicePointer(), M,
                            (FloatPointer) xTauPointer.getDevicePointer(),
                            new CudaPointer(workspace).asFloatPointer(),
                            worksize,
                            new CudaPointer(allocator.getPointer(INFO, ctx)).asIntPointer()
                            );
            if (stat != CUSOLVER_STATUS_SUCCESS) {
                throw new BlasException("cusolverDnSgeqrf failed", stat);
            }
            
            allocator.registerAction(ctx, a);
            //allocator.registerAction(ctx, tau);
            allocator.registerAction(ctx, INFO);
            if (INFO.getInt(0) != 0 ) {
                throw new BlasException("cusolverDnSgeqrf failed on INFO", INFO.getInt(0));
            }

            // Copy R ( upper part of Q ) into result
            if( r != null ) {
                r.assign( a.get( NDArrayIndex.interval( 0, a.columns() ), NDArrayIndex.all() ) ) ; 
	            
                INDArrayIndex ix[] = new INDArrayIndex[ 2 ] ;
                for( int i=1 ; i<Math.min( a.rows(), a.columns() ) ; i++ ) {
                    ix[0] = NDArrayIndex.point( i ) ;
                    ix[1] = NDArrayIndex.interval( 0, i ) ;				
                    r.put(ix, 0) ;
                }
            }

            stat = cusolverDnSorgqr_bufferSize(solverDn, M, N, N,
                        (FloatPointer) xAPointer.getDevicePointer(), M,
                        (FloatPointer) xTauPointer.getDevicePointer(),
                        (IntPointer) worksizeBuffer.addressPointer()
                );
            worksize = worksizeBuffer.getInt(0);
            workspace = new Workspace(worksize * Nd4j.sizeOfDataType());

            stat = cusolverDnSorgqr(solverDn, M, N, N,
                        (FloatPointer) xAPointer.getDevicePointer(), M,
                        (FloatPointer) xTauPointer.getDevicePointer(),
                        new CudaPointer(workspace).asFloatPointer(),
                        worksize,
                        new CudaPointer(allocator.getPointer(INFO, ctx)).asIntPointer()
            );
            if (stat != CUSOLVER_STATUS_SUCCESS) {
                throw new BlasException("cusolverDnSorgqr failed", stat);
            }            
        }
        allocator.registerAction(ctx, a);
        allocator.registerAction(ctx, INFO);
        //    allocator.registerAction(ctx, tau);

        if (a != A)
            A.assign(a);
        if ( r!=null && r != R )
            R.assign(r);

        log.info("A: {}", A);
        if( R != null ) log.info("R: {}", R);
    }

    @Override
    public void dgeqrf(int M, int N, INDArray A, INDArray R, INDArray INFO) {
        INDArray a = A;
        INDArray r = R;

        if (Nd4j.dataType() != DataBuffer.Type.DOUBLE)
            log.warn("DOUBLE getrf called in FLOAT environment");

        if (A.ordering() == 'c')
            a = A.dup('f');
        if ( R!=null && R.ordering() == 'c')
            r = R.dup('f');

        INDArray tau = Nd4j.createArrayFromShapeBuffer(Nd4j.getDataBufferFactory().createDouble(N),
                Nd4j.getShapeInfoProvider().createShapeInformation(new int[] {1, N}));

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        // Get context for current thread
        CudaContext ctx = (CudaContext) allocator.getDeviceContext().getContext();

        // setup the solver handles for cuSolver calls
        cusolverDnHandle_t handle = ctx.getSolverHandle();
        cusolverDnContext solverDn = new cusolverDnContext(handle);

        // synchronized on the solver
        synchronized (handle) {
            int result = cusolverDnSetStream(new cusolverDnContext(handle), new CUstream_st(ctx.getOldStream()));
            if (result != 0)
                throw new BlasException("solverSetStream failed");

            // transfer the INDArray into GPU memory
            CublasPointer xAPointer = new CublasPointer(a, ctx);
            CublasPointer xTauPointer = new CublasPointer(tau, ctx);

            // this output - indicates how much memory we'll need for the real operation
            DataBuffer worksizeBuffer = Nd4j.getDataBufferFactory().createInt(1);

            int stat = cusolverDnDgeqrf_bufferSize(solverDn, M, N, 
                        (DoublePointer) xAPointer.getDevicePointer(), M,
                        (IntPointer) worksizeBuffer.addressPointer() // we intentionally use host pointer here
            );

            if (stat != CUSOLVER_STATUS_SUCCESS) {
                throw new BlasException("cusolverDnDgeqrf_bufferSize failed", stat);
            }
            int worksize = worksizeBuffer.getInt(0);
            // Now allocate memory for the workspace, the permutation matrix and a return code
            Pointer workspace = new Workspace(worksize * Nd4j.sizeOfDataType());

            // Do the actual QR decomp
            stat = cusolverDnDgeqrf(solverDn, M, N, 
                            (DoublePointer) xAPointer.getDevicePointer(), M,
                            (DoublePointer) xTauPointer.getDevicePointer(),
                            new CudaPointer(workspace).asDoublePointer(),
                            worksize,
                            new CudaPointer(allocator.getPointer(INFO, ctx)).asIntPointer()
                            );
            if (stat != CUSOLVER_STATUS_SUCCESS) {
                throw new BlasException("cusolverDnDgeqrf failed", stat);
            }
            
            allocator.registerAction(ctx, a);
            allocator.registerAction(ctx, tau);
            allocator.registerAction(ctx, INFO);
            if (INFO.getInt(0) != 0 ) {
                throw new BlasException("cusolverDnDgeqrf failed with info", INFO.getInt(0));
            }

            // Copy R ( upper part of Q ) into result
            if( r != null ) {
                r.assign( a.get( NDArrayIndex.interval( 0, a.columns() ), NDArrayIndex.all() ) ) ; 
	            
                INDArrayIndex ix[] = new INDArrayIndex[ 2 ] ;
                for( int i=1 ; i<Math.min( a.rows(), a.columns() ) ; i++ ) {
                    ix[0] = NDArrayIndex.point( i ) ;
                    ix[1] = NDArrayIndex.interval( 0, i ) ;				
                    r.put(ix, 0) ;
                }
            }
            stat = cusolverDnDorgqr_bufferSize(solverDn, M, N, N,
                            (DoublePointer) xAPointer.getDevicePointer(), M,
                            (DoublePointer) xTauPointer.getDevicePointer(),
                            (IntPointer) worksizeBuffer.addressPointer()
                );
            worksize = worksizeBuffer.getInt(0);
            workspace = new Workspace(worksize * Nd4j.sizeOfDataType());

            stat = cusolverDnDorgqr(solverDn, M, N, N,
                                (DoublePointer) xAPointer.getDevicePointer(), M,
                                (DoublePointer) xTauPointer.getDevicePointer(),
                                new CudaPointer(workspace).asDoublePointer(),
                                worksize,
                                new CudaPointer(allocator.getPointer(INFO, ctx)).asIntPointer()
                                );
            if (stat != CUSOLVER_STATUS_SUCCESS) {
                throw new BlasException("cusolverDnDorgqr failed", stat);
            }            
        }
        allocator.registerAction(ctx, a);
        allocator.registerAction(ctx, INFO);

        if (a != A)
            A.assign(a);
        if ( r!=null && r != R )
            R.assign(r);

        log.info("A: {}", A);
        if( R != null ) log.info("R: {}", R);
    }

//=========================    
// CHOLESKY DECOMP
    @Override
    public void spotrf(byte uplo, int N, INDArray A, INDArray INFO) {
        INDArray a = A;

        if (Nd4j.dataType() != DataBuffer.Type.FLOAT)
            log.warn("DOUBLE potrf called in FLOAT environment");

        if (A.ordering() == 'c')
            a = A.dup('f');

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        // Get context for current thread
        CudaContext ctx = (CudaContext) allocator.getDeviceContext().getContext();

        // setup the solver handles for cuSolver calls
        cusolverDnHandle_t handle = ctx.getSolverHandle();
        cusolverDnContext solverDn = new cusolverDnContext(handle);

        // synchronized on the solver
        synchronized (handle) {
            int result = cusolverDnSetStream(new cusolverDnContext(handle), new CUstream_st(ctx.getOldStream()));
            if (result != 0)
                throw new BlasException("solverSetStream failed");

            // transfer the INDArray into GPU memory
            CublasPointer xAPointer = new CublasPointer(a, ctx);

            // this output - indicates how much memory we'll need for the real operation
            DataBuffer worksizeBuffer = Nd4j.getDataBufferFactory().createInt(1);

            int stat = cusolverDnSpotrf_bufferSize(solverDn, uplo, N, 
                        (FloatPointer) xAPointer.getDevicePointer(), N,
                        (IntPointer) worksizeBuffer.addressPointer() // we intentionally use host pointer here
            );

            if (stat != CUSOLVER_STATUS_SUCCESS) {
                throw new BlasException("cusolverDnSpotrf_bufferSize failed", stat);
            }

            int worksize = worksizeBuffer.getInt(0);
            // Now allocate memory for the workspace, the permutation matrix and a return code
            Pointer workspace = new Workspace(worksize * Nd4j.sizeOfDataType());

            // Do the actual decomp
            stat = cusolverDnSpotrf(solverDn, uplo, N, 
                            (FloatPointer) xAPointer.getDevicePointer(), N,
                            new CudaPointer(workspace).asFloatPointer(),
                            worksize,
                            new CudaPointer(allocator.getPointer(INFO, ctx)).asIntPointer()
                            );

            if (stat != CUSOLVER_STATUS_SUCCESS) {
                throw new BlasException("cusolverDnSpotrf failed", stat);
            }
        }
        allocator.registerAction(ctx, a);
        allocator.registerAction(ctx, INFO);

        if (a != A)
            A.assign(a);

        if( uplo == 'U' ) {		
            A.assign( A.transpose() ) ;	
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

        log.info("A: {}", A);
    }

    @Override
    public void dpotrf(byte uplo, int N, INDArray A, INDArray INFO) {
        INDArray a = A;

        if (Nd4j.dataType() != DataBuffer.Type.DOUBLE)
            log.warn("FLOAT potrf called in DOUBLE environment");

        if (A.ordering() == 'c')
            a = A.dup('f');

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        // Get context for current thread
        CudaContext ctx = (CudaContext) allocator.getDeviceContext().getContext();

        // setup the solver handles for cuSolver calls
        cusolverDnHandle_t handle = ctx.getSolverHandle();
        cusolverDnContext solverDn = new cusolverDnContext(handle);

        // synchronized on the solver
        synchronized (handle) {
            int result = cusolverDnSetStream(new cusolverDnContext(handle), new CUstream_st(ctx.getOldStream()));
            if (result != 0)
                throw new BlasException("solverSetStream failed");

            // transfer the INDArray into GPU memory
            CublasPointer xAPointer = new CublasPointer(a, ctx);

            // this output - indicates how much memory we'll need for the real operation
            DataBuffer worksizeBuffer = Nd4j.getDataBufferFactory().createInt(1);

            int stat = cusolverDnDpotrf_bufferSize(solverDn, uplo, N, 
                        (DoublePointer) xAPointer.getDevicePointer(), N,
                        (IntPointer) worksizeBuffer.addressPointer() // we intentionally use host pointer here
            );

            if (stat != CUSOLVER_STATUS_SUCCESS) {
                throw new BlasException("cusolverDnDpotrf_bufferSize failed", stat);
            }

            int worksize = worksizeBuffer.getInt(0);
            // Now allocate memory for the workspace, the permutation matrix and a return code
            Pointer workspace = new Workspace(worksize * Nd4j.sizeOfDataType());

            // Do the actual decomp
            stat = cusolverDnDpotrf(solverDn, uplo, N, 
                            (DoublePointer) xAPointer.getDevicePointer(), N,
                            new CudaPointer(workspace).asDoublePointer(),
                            worksize,
                            new CudaPointer(allocator.getPointer(INFO, ctx)).asIntPointer()
                            );

            if (stat != CUSOLVER_STATUS_SUCCESS) {
                throw new BlasException("cusolverDnDpotrf failed", stat);
            }
        }
        allocator.registerAction(ctx, a);
        allocator.registerAction(ctx, INFO);

        if (a != A)
            A.assign(a);

        if( uplo == 'U' ) {		
            A.assign( A.transpose() ) ;	
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

        log.info("A: {}", A);
    }


    /**
     * Generate inverse ggiven LU decomp
     *
     * @param N
     * @param A
     * @param IPIV
     * @param WORK
     * @param lwork
     * @param INFO
     */
    @Override
    public void getri(int N, INDArray A, int lda, int[] IPIV, INDArray WORK, int lwork, int INFO) {

    }


    @Override
    public void sgesvd(byte jobu, byte jobvt, int M, int N, INDArray A, INDArray S, INDArray U, INDArray VT,
                    INDArray INFO) {

        if (Nd4j.dataType() != DataBuffer.Type.FLOAT)
            log.warn("FLOAT gesvd called in DOUBLE environment");

        INDArray a = A;
        INDArray u = U;
        INDArray vt = VT;

 	// we should transpose & adjust outputs if M<N
	// cuda has a limitation, but it's OK we know
	// 	A = U S V'
	// transpose multiply rules give us ...
	// 	A' = V S' U'
	boolean hadToTransposeA = false ;
	if( M<N ) {
		hadToTransposeA = true ;

		int tmp1 = N ;
		N = M ;
		M = tmp1 ;
        byte tmp2 = jobu;
        jobu = jobvt;
        jobvt = tmp2;

		a = A.transpose().dup('f') ;
		u = (VT == null) ? null : VT.transpose().dup('f');
        vt = (U == null) ? null : U.transpose().dup('f');
	} else {
	        // cuda requires column ordering - we'll register a warning in case
       	 	if (A.ordering() == 'c')
       		     a = A.dup('f');

		if (U != null && U.ordering() == 'c')
			u = U.dup('f');

		if (VT != null && VT.ordering() == 'c')
			vt = VT.dup('f');
	}

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        // Get context for current thread
        CudaContext ctx = (CudaContext) allocator.getDeviceContext().getContext();

        // setup the solver handles for cuSolver calls
        cusolverDnHandle_t handle = ctx.getSolverHandle();
        cusolverDnContext solverDn = new cusolverDnContext(handle);

        // synchronized on the solver
        synchronized (handle) {
            int result = cusolverDnSetStream(new cusolverDnContext(handle), new CUstream_st(ctx.getOldStream()));
            if (result != 0)
                throw new BlasException("solverSetStream failed");

            // transfer the INDArray into GPU memory
            CublasPointer xAPointer = new CublasPointer(a, ctx);

            // this output - indicates how much memory we'll need for the real operation
            DataBuffer worksizeBuffer = Nd4j.getDataBufferFactory().createInt(1);

            int stat = cusolverDnSgesvd_bufferSize(solverDn, M, N, (IntPointer) worksizeBuffer.addressPointer() // we intentionally use host pointer here
            );
            if (stat != CUSOLVER_STATUS_SUCCESS) {
                throw new BlasException("cusolverDnSgesvd_bufferSize failed", stat);
            }
            int worksize = worksizeBuffer.getInt(0);

            Pointer workspace = new Workspace(worksize * Nd4j.sizeOfDataType());
            DataBuffer rwork = Nd4j.getDataBufferFactory().createFloat((M < N ? M : N) - 1);

            // Do the actual decomp
            stat = cusolverDnSgesvd(solverDn, jobu, jobvt, M, N, (FloatPointer) xAPointer.getDevicePointer(), M,
                            new CudaPointer(allocator.getPointer(S, ctx)).asFloatPointer(),
                            u == null ? null : new CudaPointer(allocator.getPointer(u, ctx)).asFloatPointer(), M,
                            vt == null ? null : new CudaPointer(allocator.getPointer(vt, ctx)).asFloatPointer(), N,
                            new CudaPointer(workspace).asFloatPointer(), worksize,
                            new CudaPointer(allocator.getPointer(rwork, ctx)).asFloatPointer(),
                            new CudaPointer(allocator.getPointer(INFO, ctx)).asIntPointer());
            if (stat != CUSOLVER_STATUS_SUCCESS) {
                throw new BlasException("cusolverDnSgesvd failed", stat);
            }
        }
        allocator.registerAction(ctx, INFO);
        allocator.registerAction(ctx, S);

        if (u != null)
            allocator.registerAction(ctx, u);
        if (vt != null)
            allocator.registerAction(ctx, vt);

        // if we transposed A then swap & transpose U & V'
        if( hadToTransposeA ) {
            if (vt != null)
                U.assign(vt.transpose());
            if (u != null)
                VT.assign(u.transpose());
        } else {
            if (u != U)
                U.assign(u);
            if (vt != VT)
                VT.assign(vt);
        }
    }


    @Override
    public void dgesvd(byte jobu, byte jobvt, int M, int N, INDArray A, INDArray S, INDArray U, INDArray VT,
                    INDArray INFO) {

        INDArray a = A;
        INDArray u = U;
        INDArray vt = VT;

 	// we should transpose & adjust outputs if M<N
	// cuda has a limitation, but it's OK we know
	// 	A = U S V'
	// transpose multiply rules give us ...
	// 	A' = V S' U'
	boolean hadToTransposeA = false ;
	if( M<N ) {
		hadToTransposeA = true ;

		int tmp1 = N ;
		N = M ;
		M = tmp1 ;
        byte tmp2 = jobu;
        jobu = jobvt;
        jobvt = tmp2;

		a = A.transpose().dup('f') ;
		u = (VT == null) ? null : VT.transpose().dup('f');
        vt = (U == null) ? null : U.transpose().dup('f');
	} else {
	        // cuda requires column ordering - we'll register a warning in case
       	 	if (A.ordering() == 'c')
       		     a = A.dup('f');

		if (U != null && U.ordering() == 'c')
			u = U.dup('f');

		if (VT != null && VT.ordering() == 'c')
			vt = VT.dup('f');
	}

        if (Nd4j.dataType() != DataBuffer.Type.DOUBLE)
            log.warn("DOUBLE gesvd called in FLOAT environment");

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        // Get context for current thread
        CudaContext ctx = (CudaContext) allocator.getDeviceContext().getContext();

        // setup the solver handles for cuSolver calls
        cusolverDnHandle_t handle = ctx.getSolverHandle();
        cusolverDnContext solverDn = new cusolverDnContext(handle);

        // synchronized on the solver
        synchronized (handle) {
            int result = cusolverDnSetStream(new cusolverDnContext(handle), new CUstream_st(ctx.getOldStream()));
            if (result != 0)
                throw new BlasException("solverSetStream failed");

            // transfer the INDArray into GPU memory
            CublasPointer xAPointer = new CublasPointer(a, ctx);

            // this output - indicates how much memory we'll need for the real operation
            DataBuffer worksizeBuffer = Nd4j.getDataBufferFactory().createInt(1);

            int stat = cusolverDnSgesvd_bufferSize(solverDn, M, N, (IntPointer) worksizeBuffer.addressPointer() // we intentionally use host pointer here
            );

            if (stat != CUSOLVER_STATUS_SUCCESS) {
                throw new BlasException("cusolverDnSgesvd_bufferSize failed", stat);
            }
            int worksize = worksizeBuffer.getInt(0);

            // Now allocate memory for the workspace, the non-converging row buffer and a return code
            Pointer workspace = new Workspace(worksize * Nd4j.sizeOfDataType());
            DataBuffer rwork = Nd4j.getDataBufferFactory().createDouble((M < N ? M : N) - 1);

            // Do the actual decomp
            stat = cusolverDnDgesvd(solverDn, jobu, jobvt, M, N, (DoublePointer) xAPointer.getDevicePointer(), M,
                            new CudaPointer(allocator.getPointer(S, ctx)).asDoublePointer(),
                            u == null ? null : new CudaPointer(allocator.getPointer(u, ctx)).asDoublePointer(), M,
                            vt == null ? null : new CudaPointer(allocator.getPointer(vt, ctx)).asDoublePointer(), N,
                            new CudaPointer(workspace).asDoublePointer(), worksize,
                            new CudaPointer(allocator.getPointer(rwork, ctx)).asDoublePointer(),
                            new CudaPointer(allocator.getPointer(INFO, ctx)).asIntPointer());

            if (stat != CUSOLVER_STATUS_SUCCESS) {
                throw new BlasException("cusolverDnDgesvd failed" + stat);
            }
        }
        allocator.registerAction(ctx, INFO);
        allocator.registerAction(ctx, S);
        allocator.registerAction(ctx, a);

        if (u != null)
            allocator.registerAction(ctx, u);

        if (vt != null)
            allocator.registerAction(ctx, vt);

        // if we transposed A then swap & transpose U & V'
        if( hadToTransposeA ) {
            if (vt != null)
                U.assign(vt.transpose());
            if (u != null)
                VT.assign(u.transpose());
        } else {
            if (u != U)
                U.assign(u);
            if (vt != VT)
                VT.assign(vt);
        }
    }

    public int ssyev( char _jobz, char _uplo, int N, INDArray A, INDArray R ) {

	int status = -1 ;

	int jobz = _jobz == 'V' ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR ;
	int uplo = _uplo == 'L' ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER ;

        if (Nd4j.dataType() != DataBuffer.Type.FLOAT)
            log.warn("FLOAT ssyev called in DOUBLE environment");

        INDArray a = A;

        if (A.ordering() == 'c')
            a = A.dup('f');

        // FIXME: int cast
	int M = (int) A.rows() ;

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        // Get context for current thread
        CudaContext ctx = (CudaContext) allocator.getDeviceContext().getContext();

        // setup the solver handles for cuSolver calls
        cusolverDnHandle_t handle = ctx.getSolverHandle();
        cusolverDnContext solverDn = new cusolverDnContext(handle);

        // synchronized on the solver
        synchronized (handle) {
            status = cusolverDnSetStream(new cusolverDnContext(handle), new CUstream_st(ctx.getOldStream()));
            if( status == 0 ) {
		    // transfer the INDArray into GPU memory
		    CublasPointer xAPointer = new CublasPointer(a, ctx);
		    CublasPointer xRPointer = new CublasPointer(R, ctx);

		    // this output - indicates how much memory we'll need for the real operation
		    DataBuffer worksizeBuffer = Nd4j.getDataBufferFactory().createInt(1);
		    status = cusolverDnSsyevd_bufferSize (
				solverDn, jobz, uplo, M, 
				(FloatPointer) xAPointer.getDevicePointer(), M,
				(FloatPointer) xRPointer.getDevicePointer(),
				(IntPointer)worksizeBuffer.addressPointer() ) ;

		    if (status == CUSOLVER_STATUS_SUCCESS) {
			    int worksize = worksizeBuffer.getInt(0);

			    // allocate memory for the workspace, the non-converging row buffer and a return code
			    Pointer workspace = new Workspace(worksize * Nd4j.sizeOfDataType());

			    INDArray INFO = Nd4j.createArrayFromShapeBuffer(Nd4j.getDataBufferFactory().createInt(1),
		  	    Nd4j.getShapeInfoProvider().createShapeInformation(new int[] {1, 1}));


			    // Do the actual decomp
			    status = cusolverDnSsyevd(solverDn, jobz, uplo, M, 
					(FloatPointer) xAPointer.getDevicePointer(), M,
					(FloatPointer) xRPointer.getDevicePointer(), 
					new CudaPointer(workspace).asFloatPointer(), worksize,
					new CudaPointer(allocator.getPointer(INFO, ctx)).asIntPointer());

			    allocator.registerAction(ctx, INFO);
			    if( status == 0 ) status = INFO.getInt(0) ;
		    }
		}
        }
	if( status == 0 ) {
		allocator.registerAction(ctx, R);
		allocator.registerAction(ctx, a);

		if (a != A)
		    A.assign(a);
	}
	return status ;
    }


    public int dsyev( char _jobz, char _uplo, int N, INDArray A, INDArray R ) {

	int status = -1 ;

	int jobz = _jobz == 'V' ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR ;
	int uplo = _uplo == 'L' ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER ;

        if (Nd4j.dataType() != DataBuffer.Type.DOUBLE)
            log.warn("DOUBLE dsyev called in FLOAT environment");

        INDArray a = A;

        if (A.ordering() == 'c')
            a = A.dup('f');

        // FIXME: int cast
	int M = (int) A.rows() ;

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        // Get context for current thread
        CudaContext ctx = (CudaContext) allocator.getDeviceContext().getContext();

        // setup the solver handles for cuSolver calls
        cusolverDnHandle_t handle = ctx.getSolverHandle();
        cusolverDnContext solverDn = new cusolverDnContext(handle);

        // synchronized on the solver
        synchronized (handle) {
            status = cusolverDnSetStream(new cusolverDnContext(handle), new CUstream_st(ctx.getOldStream()));
            if( status == 0 ) {
		    // transfer the INDArray into GPU memory
		    CublasPointer xAPointer = new CublasPointer(a, ctx);
		    CublasPointer xRPointer = new CublasPointer(R, ctx);

		    // this output - indicates how much memory we'll need for the real operation
		    DataBuffer worksizeBuffer = Nd4j.getDataBufferFactory().createInt(1);
		    status = cusolverDnDsyevd_bufferSize(
				solverDn, jobz, uplo, M, 
				(DoublePointer) xAPointer.getDevicePointer(), M,
				(DoublePointer) xRPointer.getDevicePointer(),
				(IntPointer)worksizeBuffer.addressPointer() ) ;

		    if (status == CUSOLVER_STATUS_SUCCESS) {
			    int worksize = worksizeBuffer.getInt(0);

			    // allocate memory for the workspace, the non-converging row buffer and a return code
			    Pointer workspace = new Workspace(worksize * Nd4j.sizeOfDataType());

			    INDArray INFO = Nd4j.createArrayFromShapeBuffer(Nd4j.getDataBufferFactory().createInt(1),
			    Nd4j.getShapeInfoProvider().createShapeInformation(new int[] {1, 1}));


			    // Do the actual decomp
			    status = cusolverDnDsyevd(solverDn, jobz, uplo, M, 
					(DoublePointer) xAPointer.getDevicePointer(), M,
					(DoublePointer) xRPointer.getDevicePointer(), 
					new CudaPointer(workspace).asDoublePointer(), worksize,
					new CudaPointer(allocator.getPointer(INFO, ctx)).asIntPointer());

			    allocator.registerAction(ctx, INFO);
			    if( status == 0 ) status = INFO.getInt(0) ;
		    }
		}
        }
	if( status == 0 ) {
		allocator.registerAction(ctx, R);
		allocator.registerAction(ctx, a);

		if (a != A)
		    A.assign(a);
	}
	return status ;
    }

    static class Workspace extends Pointer {
        public Workspace(long size) {
            super(NativeOpsHolder.getInstance().getDeviceNativeOps().mallocDevice(size, null, 0));
            deallocator(new Deallocator() {
                @Override
                public void deallocate() {
                    NativeOpsHolder.getInstance().getDeviceNativeOps().freeDevice(Workspace.this, null);
                }
            });
        }
    }
}

package org.nd4j.linalg.jcublas.blas;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.linalg.api.blas.impl.BaseLapack;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;

import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;


import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.linalg.jcublas.CublasPointer;
import org.nd4j.jita.allocator.pointers.cuda.cusolverDnHandle_t;
import org.nd4j.jita.allocator.impl.AtomicAllocator ;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.IntBuffer;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;
import org.nd4j.linalg.jcublas.buffer.CudaIntDataBuffer;
import org.nd4j.linalg.jcublas.buffer.CudaFloatDataBuffer;



import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

import static org.bytedeco.javacpp.cusolver.*;

/**
 * JCublas lapack
 *
 * @author Adam Gibson
 */
public class JcublasLapack extends BaseLapack {

    private NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
    private Allocator allocator = AtomicAllocator.getInstance();
    private static Logger logger = LoggerFactory.getLogger(JcublasLapack.class);



    /**
     * LU decomposiiton of a matrix
     *
     * @param M
     * @param N
     * @param A
     * @param lda
     * @param IPIV
     * @param INFO
     */
    @Override
    public void getrf(int M, int N, INDArray A, int lda, INDArray IPIV, INDArray INFO) {
	
	if (Nd4j.dataType() != DataBuffer.Type.FLOAT)
            logger.warn("FLOAT getrf called");

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

	// Get context for current thread
        CudaContext ctx = (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext() ;

	// setup the solver handles for cuSolver calls
        cusolverDnHandle_t handle = ctx.getSolverHandle();
	cusolverDnContext solverDn = new cusolverDnContext( handle ) ;

	// synchronized on the solver
        synchronized (handle) {
		long result = nativeOps.setSolverStream(handle, ctx.getOldStream());
           	if (result == 0)
	               	throw new IllegalStateException("solverSetStream failed");

		// transfer the INDArray into GPU memory
        	CublasPointer xAPointer = new CublasPointer(A, ctx);
		// this output - indicates how much memory we'll need for the real operation
		DataBuffer worksize = Nd4j.getDataBufferFactory().createInt(1) ;

		int stat = cusolverDnSgetrf_bufferSize( 
			solverDn, 
			M, N, 
			(FloatPointer)xAPointer.getDevicePointer(), 
			lda, 
			(IntPointer) worksize.addressPointer() // we intentionally use host pointer here
			) ;

		if( stat != CUSOLVER_STATUS_SUCCESS ) {
 		    throw new IllegalStateException("cusolverDnSgetrf_bufferSize failed with code: " + stat ) ;
		}
		logger.info("Worksize returned: {}", worksize.getInt(0));

		// Now allocate memory for the workspace, the permutation matrix and a return code
		DataBuffer work = Nd4j.getDataBufferFactory().createFloat(worksize.getInt(0)) ;
		//DataBuffer ipiv = Nd4j.getDataBufferFactory().createInt( IPIV.length() ) ;
		//DataBuffer info = Nd4j.getDataBufferFactory().createInt(1) ;

		//IntPointer ip1 = (IntPointer) AtomicAllocator.getInstance().getPointer(ipiv, ctx) ;
		//IntPointer ip2 = (IntPointer) ipiv.addressPointer() ;

		logger.info("IPIV data before: {}", Arrays.toString(IPIV.data().asInt()));

		// DO the actual LU decomp
		stat = cusolverDnSgetrf(
			solverDn,
			M, N, 
			(FloatPointer)xAPointer.getDevicePointer(), 
			lda, 
			new CudaPointer(AtomicAllocator.getInstance().getHostPointer(work)).asFloatPointer(),
			new CudaPointer(AtomicAllocator.getInstance().getHostPointer(IPIV)).asIntPointer() ,
			new CudaPointer(AtomicAllocator.getInstance().getHostPointer(INFO)).asIntPointer()
			) ;

			// we do sync to make sure getr is finished
			ctx.syncOldStream();

			logger.info("IPIV data after: {}", Arrays.toString(IPIV.data().asInt()));

		if( stat != CUSOLVER_STATUS_SUCCESS ) {
 		    throw new IllegalStateException("cusolverDnSgetrf failed with code: " + stat ) ;
		}
			// Copy the results back to the input vectors//
			// INFO.putScalar(0,info.asInt()[0] ) ;
			// int xxx[] = ipiv.asInt() ;
			// obtain pointers
			// Pointer dst = AtomicAllocator.getInstance().getPointer(IPIV, ctx);
			//	Pointer src = AtomicAllocator.getInstance().getPointer(ipiv, ctx);

			// device to device copy
			// nativeOps.memcpyAsync(dst, src, IPIV.length() * 4, 3, ctx.getSpecialStream());
			// ctx.syncSpecialStream();

			// notify that IPIV was modified on device side
			AtomicAllocator.getInstance().getAllocationPoint(IPIV).tickDeviceWrite();

			// A is modified on device side as well
			AtomicAllocator.getInstance().getAllocationPoint(INFO).tickDeviceWrite();
			AtomicAllocator.getInstance().getAllocationPoint(A).tickDeviceWrite();

			//IPIV.setData( ipiv );
			// now when you'll call getInt(), data will travel back to host
//			if( IPIV.getInt(2) != 4 ) { 
//				System.out.println( "WTF" + xxx[2] ) ; 
//			}

		// After we get an inplace result we should 
		// transpose the array - because of differenes in 
		// column- and row-major ordering between ND4J & CUDA
        	//A.setStride( A.stride()[1], A.stride()[0] );
	}
		// this op call is synchronous, so we don't need register action here
        //allocator.registerAction(ctx, A );
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

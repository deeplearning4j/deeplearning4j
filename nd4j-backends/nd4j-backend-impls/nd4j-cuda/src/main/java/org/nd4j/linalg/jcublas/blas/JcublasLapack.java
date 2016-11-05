package org.nd4j.linalg.jcublas.blas;

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
		IntPointer worksize = new IntPointer(1) ;

		int stat = cusolverDnSgetrf_bufferSize( 
			solverDn, 
			M, N, 
			(FloatPointer)xAPointer.getDevicePointer(), 
			lda, 
			worksize
			) ;

		// Now allocate memory for the workspace, the permutation matrix and a return code
		BaseCudaDataBuffer work = new CudaFloatDataBuffer(worksize.get(0)) ;
		BaseCudaDataBuffer ipiv = new CudaIntDataBuffer(IPIV.length()) ;
		BaseCudaDataBuffer info = new CudaIntDataBuffer(1) ;

		// DO the actual LU decomp
		stat = cusolverDnSgetrf(
			solverDn,
			M, N, 
			(FloatPointer)xAPointer.getDevicePointer(), 
			lda, 
			(FloatPointer)work.addressPointer() ,
			(IntPointer)ipiv.addressPointer() ,
			(IntPointer)info.addressPointer() 
			) ;

		// Copy the results back to the input vectors
		INFO.putScalar(0,info.asInt()[0] ) ;
		IPIV.setData( new IntBuffer( ipiv.asInt() ) );
//		int perm [] = ipiv.asInt() ;
//		for( int i=0 ; i<perm.length ; ++i) {
//			IPIV.putScalar(i,perm[i] ) ;
//		}	

		// After we get an inplace result we should 
		// transpose the array - because of differenes in 
		// column- and row-major ordering between ND4J & CUDA
        	A.setStride( A.stride()[1], A.stride()[0] );
	}
	// copy the result from GPU -> CPU memory
        allocator.registerAction(ctx, A );
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

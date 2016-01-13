package jcuda;

import jcuda.jcublas.JCublas2;
import jcuda.runtime.JCuda;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.blas.BlasBufferUtil;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.CublasPointer;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.buffer.allocation.PageableDirectBufferMemoryStrategy;
import org.nd4j.linalg.jcublas.buffer.allocation.PinnedMemoryStrategy;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Method;

import static org.junit.Assert.*;

/**
 * This set of very basic tests will check for memory leaks in different allocation cases.
 *
 * 1. full array allocation
 * 2. view allocation
 * 3. nested view allocation
 * 4. allocation over the same memory space
 * 5. allocation from buffer directly
 * 6. subarrays with stride > 1 (based on columns taken from 2D arrays)
 *
 * + few additional tests for data integrity check using pre-calculated comparison values
 *
 * On later stages, sparse allocations should be tested here as well. But for cuSparse that shouldn't be an issue, due to dense underlying CSR format.
 *
 * All this tests should be executed against both Pageable and Pinned MemoryStrategies.
 * If any more strategies will be presented, it would be nice to add them here
 *
 * PLEASE NOTE: this test was intentionally left within jcuda package, to allow access to protected method Pointer.getNativePointer()
 *
 * @author raver119@gmail.com
 */
public class CublasPointerRevTests {


    private static Logger log = LoggerFactory.getLogger(CublasPointerRevTests.class);

    @Before
    public void setUp() throws Exception {

    }

    /**
     * This test is most simple check for backend loader ever.
     *
     * @throws Exception
     */
    @Test
    public void testSetup() throws Exception {
            INDArray array = Nd4j.create(new double[]{1.0});
    }


    /**
     * This is primitive test for forced MemoryStrategies
     * @throws Exception
     */
    @Test
    public void testForcedMemoryStrategy() throws Exception {
        ContextHolder.getInstance().forceMemoryStrategyForThread(new PinnedMemoryStrategy());

        assertEquals("PinnedMemoryStrategy", ContextHolder.getInstance().getMemoryStrategy().getClass().getSimpleName());

        ContextHolder.getInstance().forceMemoryStrategyForThread(new PageableDirectBufferMemoryStrategy());

        assertEquals("PageableDirectBufferMemoryStrategy", ContextHolder.getInstance().getMemoryStrategy().getClass().getSimpleName());

        // explicit check for nullified forced strategy, this should get back to default memory strategy for crr
        ContextHolder.getInstance().forceMemoryStrategyForThread(null);

        assertEquals("PinnedMemoryStrategy", ContextHolder.getInstance().getMemoryStrategy().getClass().getSimpleName());

    }

    /**
     *
     */
    @Test
    public void testPageableMemoryRelease() throws Exception {
        // force current thread to use Pageable memory strategy
        ContextHolder.getInstance().forceMemoryStrategyForThread(new PageableDirectBufferMemoryStrategy());

        assertEquals("PageableDirectBufferMemoryStrategy", ContextHolder.getInstance().getMemoryStrategy().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});

        CudaContext ctx = CudaContext.getBlasContext();

        double[] ret = new double[1];
        ret[0] = 15.0d;
        Pointer result = Pointer.to(ret);

        CublasPointer xCPointer = new CublasPointer(array1,ctx);

        BaseCudaDataBuffer buffer1 = (BaseCudaDataBuffer) xCPointer.getBuffer();

        assertEquals(DataBuffer.AllocationMode.DIRECT, buffer1.allocationMode());
        assertTrue(buffer1.copied(Thread.currentThread().getName()));
        assertFalse(buffer1.isPersist());

        // we're pushing whole array to device, not a view; so buffer length should be equal to array length
        assertEquals(15, buffer1.length());

        long addr_buff1 = xCPointer.getDevicePointer().getNativePointer();

        CublasPointer yCPointer = new CublasPointer(array2,ctx);

        BaseCudaDataBuffer buffer2 = (BaseCudaDataBuffer) yCPointer.getBuffer();

        assertFalse(xCPointer.isResultPointer());
        assertFalse(xCPointer.isClosed());


        JCublas2.cublasDdot(
                ctx.getHandle(),
                array1.length(),
                xCPointer.getDevicePointer().withByteOffset(array1.offset() * array1.data().getElementSize()),
                BlasBufferUtil.getBlasStride(array1),
                yCPointer.getDevicePointer().withByteOffset(array2.offset() * array2.data().getElementSize()),
                BlasBufferUtil.getBlasStride(array2),
                result);
        ctx.syncOldStream();


        // in this test copyToHost is handled by JCublas, so there's no need for explicit copyToHost call
        ctx.finishBlasOperation();


        // check that result not equals to 0
        assertNotEquals(15.0d, ret[0], 0.0001d);

        // we emulate AutoCloseable by direct close() call
        // close call should fire freeDevicePointer
        // AND freeHost
        xCPointer.close();
        yCPointer.close();



        // here we check, if device pointer was released
        assertEquals(true, xCPointer.isClosed());
        assertEquals(true, yCPointer.isClosed());

        // now we should check, if host memory was released.
        // if not - freeHost() wasn't called for corresponding buffer and we have memory leak there
        assertTrue(buffer1.isFreed());

        assertTrue(buffer2.isFreed());

        // make sure buffer got 0 references left
        assertEquals(0, buffer1.getPointersToContexts().size());
        assertEquals(0, buffer2.getPointersToContexts().size());
    }

    /**
     * This test addresses subsequent alloc/free for views within pageable memory.
     * + we also test same offset allocation within same thread (sequental access test only)
     *
     * @throws Exception
     */
    @Test
    public void testPageableMemoryReleaseSlicedSubsequently() throws Exception {
        // force current thread to use Pageable memory strategy
        ContextHolder.getInstance().forceMemoryStrategyForThread(new PageableDirectBufferMemoryStrategy());

        assertEquals("PageableDirectBufferMemoryStrategy", ContextHolder.getInstance().getMemoryStrategy().getClass().getSimpleName());

        INDArray baseArray1 = Nd4j.rand(new int[]{1000, 200}, -1.0, 1.0, new DefaultRandom());
        INDArray baseArray2 = Nd4j.rand(new int[]{1000, 200}, -1.0, 1.0, new DefaultRandom());


        INDArray slice1 = baseArray1.slice(1);
        INDArray slice2 = baseArray2.slice(1);

        CudaContext ctx = CudaContext.getBlasContext();

        CublasPointer xCPointer = new CublasPointer(slice1,ctx);
        CublasPointer yCPointer = new CublasPointer(slice2,ctx);

        // at this moment we have 2 buffers allocated, and 2 pointers + offsets set up. time to add new slices to the equation

        INDArray slice3 = baseArray1.slice(3);

        // please note, slice(1) isn't a type. we're testing one more edge here: double offset allocation being used within same thread
        INDArray slice4 = baseArray2.slice(1);

        CublasPointer xDPointer = new CublasPointer(slice3, ctx);
        CublasPointer yDPointer = new CublasPointer(slice4, ctx);

        BaseCudaDataBuffer bufferSlice1 = (BaseCudaDataBuffer) xCPointer.getBuffer();
        BaseCudaDataBuffer bufferSlice2 = (BaseCudaDataBuffer) yCPointer.getBuffer();
        BaseCudaDataBuffer bufferSlice3 = (BaseCudaDataBuffer) xDPointer.getBuffer();
        BaseCudaDataBuffer bufferSlice4 = (BaseCudaDataBuffer) yDPointer.getBuffer();

        JCublas2.cublasSaxpy(
                ctx.getHandle(),
                slice1.length(),
                Pointer.to(new float[]{1.0f}),
                xCPointer.getDevicePointer().withByteOffset(slice1.offset() * slice1.data().getElementSize()),
                BlasBufferUtil.getBlasStride(slice1),
                yCPointer.getDevicePointer().withByteOffset(slice2.offset() * slice2.data().getElementSize()),
                BlasBufferUtil.getBlasStride(slice2));
        ctx.syncOldStream();

        // we have to copyback
        yCPointer.copyToHost();

        ctx.finishBlasOperation();

        // now we'll start closing pointers
        xCPointer.close();
        yCPointer.close();
        // at this point buffers should be NOT freed, since we have 2 more allocations
        assertFalse(bufferSlice1.isFreed());
        assertFalse(bufferSlice2.isFreed());
        assertFalse(bufferSlice3.isFreed());
        assertFalse(bufferSlice4.isFreed());

        ctx = CudaContext.getBlasContext();

        // at this moment we assume that yCPointer contains updated result, and we'll check it's equality to slice4
        assertEquals(slice2.getDouble(1), slice4.getDouble(1), 0.001);

        // now we'll fire axpy on slices 3 & 4
        JCublas2.cublasSaxpy(
                ctx.getHandle(),
                slice3.length(),
                Pointer.to(new float[]{1.0f}),
                xDPointer.getDevicePointer().withByteOffset(slice3.offset() * slice3.data().getElementSize()),
                BlasBufferUtil.getBlasStride(slice3),
                yDPointer.getDevicePointer().withByteOffset(slice4.offset() * slice4.data().getElementSize()),
                BlasBufferUtil.getBlasStride(slice4));
        ctx.syncOldStream();

        // copyback, once again
        yDPointer.copyToHost();

        ctx.finishBlasOperation();

        // once again, we check that memory is updated properly
        assertEquals(slice2.getDouble(1), slice4.getDouble(1), 0.001);


        // now we free slice4, and all buffers should be released now
        xDPointer.close();
        yDPointer.close();

        assertTrue(bufferSlice1.isFreed());
        assertTrue(bufferSlice2.isFreed());
        assertTrue(bufferSlice3.isFreed());
        assertTrue(bufferSlice4.isFreed());

        // make sure buffer got 0 references left
        assertEquals(0, bufferSlice1.getPointersToContexts().size());
        assertEquals(0, bufferSlice2.getPointersToContexts().size());
        assertEquals(0, bufferSlice3.getPointersToContexts().size());
        assertEquals(0, bufferSlice4.getPointersToContexts().size());
    }


    /**
     * This test addresses subsequent alloc/free for views within pinned memory.
     * + we also test same offset allocation within same thread (sequental access test only)
     *
     * @throws Exception
     */
    @Test
    public void testPinnedMemoryReleaseSlicedSubsequently() throws Exception {
        // force current thread to use Pageable memory strategy
        ContextHolder.getInstance().forceMemoryStrategyForThread(new PinnedMemoryStrategy());

        assertEquals("PinnedMemoryStrategy", ContextHolder.getInstance().getMemoryStrategy().getClass().getSimpleName());

        INDArray baseArray1 = Nd4j.rand(new int[]{1000, 200}, -1.0, 1.0, new DefaultRandom());
        INDArray baseArray2 = Nd4j.rand(new int[]{1000, 200}, -1.0, 1.0, new DefaultRandom());


        INDArray slice1 = baseArray1.slice(1);
        INDArray slice2 = baseArray2.slice(1);

        CudaContext ctx = CudaContext.getBlasContext();

        CublasPointer xCPointer = new CublasPointer(slice1,ctx);
        CublasPointer yCPointer = new CublasPointer(slice2,ctx);

        // at this moment we have 2 buffers allocated, and 2 pointers + offsets set up. time to add new slices to the equation


        INDArray slice3 = baseArray1.slice(3);

        // please note, slice(1) isn't a typo. we're testing one more edge here: double offset allocation being used within same thread
        INDArray slice4 = baseArray2.slice(1);

        CublasPointer xDPointer = new CublasPointer(slice3, ctx);
        CublasPointer yDPointer = new CublasPointer(slice4, ctx);

        BaseCudaDataBuffer bufferSlice1 = (BaseCudaDataBuffer) xCPointer.getBuffer();
        BaseCudaDataBuffer bufferSlice2 = (BaseCudaDataBuffer) yCPointer.getBuffer();

        BaseCudaDataBuffer bufferSlice3 = (BaseCudaDataBuffer) xDPointer.getBuffer();
        BaseCudaDataBuffer bufferSlice4 = (BaseCudaDataBuffer) yDPointer.getBuffer();


        JCublas2.cublasSaxpy(
                ctx.getHandle(),
                slice1.length(),
                Pointer.to(new float[]{1.0f}),
                xCPointer.getDevicePointer().withByteOffset(slice1.offset() * slice1.data().getElementSize()),
                BlasBufferUtil.getBlasStride(slice1),
                yCPointer.getDevicePointer().withByteOffset(slice2.offset() * slice2.data().getElementSize()),
                BlasBufferUtil.getBlasStride(slice2));
        ctx.syncOldStream();

        // we have to copyback
        yCPointer.copyToHost();

        ctx.finishBlasOperation();

        // at this moment we assume that yCPointer contains updated result, and we'll check it's equality to slice4
        assertEquals(slice2.getDouble(1), slice4.getDouble(1), 0.001);

        // now we'll start closing pointers
        xCPointer.close();
        yCPointer.close();

        // at this point buffers should be NOT freed, since we have 2 more allocations
        assertFalse(bufferSlice1.isFreed());
        assertFalse(bufferSlice2.isFreed());

        assertFalse(bufferSlice3.isFreed());
        assertFalse(bufferSlice4.isFreed());

        ctx = CudaContext.getBlasContext();



        // now we'll fire axpy on slices 3 & 4
        JCublas2.cublasSaxpy(
                ctx.getHandle(),
                slice3.length(),
                Pointer.to(new float[]{1.0f}),
                xDPointer.getDevicePointer().withByteOffset(slice3.offset() * slice3.data().getElementSize()),
                BlasBufferUtil.getBlasStride(slice3),
                yDPointer.getDevicePointer().withByteOffset(slice4.offset() * slice4.data().getElementSize()),
                BlasBufferUtil.getBlasStride(slice4));
        ctx.syncOldStream();

        assertFalse(xDPointer.isClosed());
        assertFalse(yDPointer.isClosed());

        // right now we have neat case. buffer is NOT marked as free, but device pointer is already discarded
        assertFalse(bufferSlice4.isFreed());

        // copyback, once again
        yDPointer.copyToHost();

        ctx.finishBlasOperation();

        // once again, we check that memory is updated properly
        assertEquals(slice2.getDouble(1), slice4.getDouble(1), 0.001);


        // now we free slice4, and all buffers should be released now
        xDPointer.close();
        yDPointer.close();

        assertTrue(bufferSlice1.isFreed());
        assertTrue(bufferSlice2.isFreed());
        assertTrue(bufferSlice3.isFreed());
        assertTrue(bufferSlice4.isFreed());


        assertEquals(0, bufferSlice1.getPointersToContexts().size());
        assertEquals(0, bufferSlice2.getPointersToContexts().size());
        assertEquals(0, bufferSlice3.getPointersToContexts().size());
        assertEquals(0, bufferSlice4.getPointersToContexts().size());

    }

    @Test
    public void testPinnedMemoryRelease() throws Exception {

        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        // reset to default MemoryStrategy, most probable is Pinned
        ContextHolder.getInstance().forceMemoryStrategyForThread(null);

        assertEquals("PinnedMemoryStrategy", ContextHolder.getInstance().getMemoryStrategy().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});


        double dotWrapped = Nd4j.getBlasWrapper().level1().dot(array1.length(), 1, array1, array2);


        CudaContext ctx = CudaContext.getBlasContext();

        double[] ret = new double[1];
        Pointer result = Pointer.to(ret);

        CublasPointer xCPointer = new CublasPointer(array1,ctx);

        BaseCudaDataBuffer buffer = (BaseCudaDataBuffer) xCPointer.getBuffer();

        assertEquals(DataBuffer.AllocationMode.DIRECT, buffer.allocationMode());
        assertTrue(buffer.copied(Thread.currentThread().getName()));
        assertFalse(buffer.isPersist());

        // we're pushing whole array to device, not a view; so buffer length should be equal to array length
        assertEquals(15, buffer.length());

        long addr_buff1 = xCPointer.getDevicePointer().getNativePointer();

        CublasPointer yCPointer = new CublasPointer(array2,ctx);

        assertFalse(xCPointer.isResultPointer());
        assertFalse(xCPointer.isClosed());

        JCublas2.cublasDdot(
                ctx.getHandle(),
                array1.length(),
                xCPointer.getDevicePointer(),
                1,
                yCPointer.getDevicePointer(),
                1,
                result);
        ctx.syncOldStream();

        // in this test copyToHost is handled by JCublas, so there's no need for explicit copyToHost call
        ctx.finishBlasOperation();


        // we emulate AutoCloseable by direct close() call
        // close call should fire freeDevicePointer
        // AND freeHost
        xCPointer.close();
        yCPointer.close();

        // here we check, if device pointer was released
        assertEquals(true, xCPointer.isClosed());
        assertEquals(true, yCPointer.isClosed());

        // now we should check, if host memory was released.
        // if not - freeHost() wasn't called for corresponding buffer and we have memory leak there
        assertTrue(buffer.isFreed());

        // Please note: we do NOT test result pointer deallocation here,since we assume it's handled by JCuda

        System.out.println("Dot product: " + ret[0] + " Dot wrapped: " + dotWrapped);

        // make sure buffer got 0 references left
        assertEquals(0, buffer.getPointersToContexts().size());
    }


    /**
     * This test addresses memory management for result array passed from ND4j to JcuBlas
     *
     * @throws Exception
     */
    @Test
    public void testPinnedMemoryReleaseResult() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        // reset to default MemoryStrategy, most probable is Pinned
        ContextHolder.getInstance().forceMemoryStrategyForThread(null);

        assertEquals("PinnedMemoryStrategy", ContextHolder.getInstance().getMemoryStrategy().getClass().getSimpleName());

        CudaContext ctx = CudaContext.getBlasContext();

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f});

        CublasPointer xAPointer = new CublasPointer(array1,ctx);

        BaseCudaDataBuffer buffer1 = (BaseCudaDataBuffer) xAPointer.getBuffer();

        assertEquals(DataBuffer.AllocationMode.DIRECT, buffer1.allocationMode());
        assertTrue(buffer1.copied(Thread.currentThread().getName()));
        assertFalse(buffer1.isPersist());

        assertEquals(15, buffer1.length());


        assertFalse(xAPointer.isResultPointer());
        assertFalse(xAPointer.isClosed());

        CublasPointer xBPointer = new CublasPointer(array2,ctx);


        BaseCudaDataBuffer buffer2 = (BaseCudaDataBuffer) xAPointer.getBuffer();


        JCublas2.cublasDaxpy(
                ctx.getHandle(),
                array1.length(),
                Pointer.to(new double[]{1.0}),
                xAPointer.getDevicePointer().withByteOffset(array1.offset() * array1.data().getElementSize()),
                BlasBufferUtil.getBlasStride(array1),
                xBPointer.getDevicePointer().withByteOffset(array2.offset() * array2.data().getElementSize()),
                BlasBufferUtil.getBlasStride(array2));
        ctx.syncOldStream();

        //now, since we have result array, we call for explicit copyback
        double valBefore = array2.getDouble(0);

        xBPointer.copyToHost();

        // we don't care if the result is true here. All we want to know is: if memory was really updated after copyback
        double valAfter = array2.getDouble(0);
        System.out.println("Val0 before: [" + valBefore+ "], after: ["+ valAfter+"]");
        assertNotEquals(valBefore, valAfter, 0.01);

        ctx.finishBlasOperation();

        // we emulate AutoCloseable by direct close() call
        // close call should fire freeDevicePointer
        // AND freeHost
        xAPointer.close();
        xBPointer.close();

        // here we check, if device pointer was released
        assertEquals(true, xAPointer.isClosed());
        assertEquals(true, xBPointer.isClosed());

        // now we should check, if host memory was released.
        // if not - freeHost() wasn't called for corresponding buffer and we have memory leak there
        assertTrue(buffer1.isFreed());

        // we check result buffer
        assertTrue(buffer2.isFreed());


        /*
            so, at this moment we know the followint machine state:
            1. Both cuBlasPointers are closed
            2. Both underlying buffers are freed
        */

        // make sure buffers got 0 references left
        assertEquals(0, buffer1.getPointersToContexts().size());
        assertEquals(0, buffer2.getPointersToContexts().size());
    }

    /**
     * This test addresses GPU memory allocation for sliced views retrieved from larger array/buffer
     *
     * @throws Exception
     */
    @Test
    public void testPinnedMemoryReleaseSliced() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        // reset to default MemoryStrategy, most probable is Pinned
        ContextHolder.getInstance().forceMemoryStrategyForThread(null);

        assertEquals("PinnedMemoryStrategy", ContextHolder.getInstance().getMemoryStrategy().getClass().getSimpleName());

        INDArray baseArray1 = Nd4j.rand(new int[]{1000, 200}, -1.0, 1.0, new DefaultRandom());
        INDArray baseArray2 = Nd4j.rand(new int[]{1000, 200}, -1.0, 1.0, new DefaultRandom());

        INDArray slice1 = baseArray1.slice(1);
        INDArray slice2 = baseArray2.slice(1);

        CudaContext ctx = CudaContext.getBlasContext();

        // We are NOT using try-with-resource here, hence we use explicit call to xAPointer.close() method, as exact emoulation of AutoCloseable behaviour
        CublasPointer xAPointer = new CublasPointer(slice1,ctx);


        BaseCudaDataBuffer buffer1 = (BaseCudaDataBuffer) xAPointer.getBuffer();

        assertEquals(DataBuffer.AllocationMode.DIRECT, buffer1.allocationMode());
        assertTrue(buffer1.copied(Thread.currentThread().getName()));
        assertFalse(buffer1.isPersist());

        // for sliced view we have whole original array allocated
        assertEquals(200000, buffer1.length());

        CublasPointer xBPointer = new CublasPointer(slice2,ctx);

        long addr_buff1 = xAPointer.getDevicePointer().getNativePointer();
        long addr_buff2 = xBPointer.getDevicePointer().getNativePointer();

        System.out.println("Native buffer1 pointer: " + addr_buff1);
        System.out.println("Native buffer2 pointer: " + addr_buff2);



        BaseCudaDataBuffer buffer2 = (BaseCudaDataBuffer) xBPointer.getBuffer();

        // the same here, for sliced view we have whole original buffer allocated using cudaHostAlloc
        assertEquals(200000, buffer2.length());


        JCublas2.cublasSaxpy(
                ctx.getHandle(),
                slice1.length(),
                Pointer.to(new float[]{1.0f}),
                xAPointer.getDevicePointer().withByteOffset(slice1.offset() * slice1.data().getElementSize()),
                BlasBufferUtil.getBlasStride(slice1),
                xBPointer.getDevicePointer().withByteOffset(slice2.offset() * slice2.data().getElementSize()),
                BlasBufferUtil.getBlasStride(slice2));
        ctx.syncOldStream();

        //now, since we have result array, we call for explicit copyback
        double valBefore = slice2.getFloat(0);

        xBPointer.copyToHost();

        // we don't care if the result is true here. All we want to know is: if memory was really updated after copyback
        double valAfter = slice2.getDouble(0);
        System.out.println("Val0 before: [" + valBefore+ "], after: ["+ valAfter+"]");
        assertNotEquals(valBefore, valAfter, 0.01);

        ctx.finishBlasOperation();


        // we emulate AutoCloseable by direct close() call
        // close() call should fire freeDevicePointer
        // AND freeHost
        xAPointer.close();
        xBPointer.close();

        // here we check, if device pointer was released
        assertEquals(true, xAPointer.isClosed());
        assertEquals(true, xBPointer.isClosed());

        // now we should check, if host memory was released.
        // if not - freeHost() wasn't called for corresponding buffer and we have memory leak there
        assertTrue(buffer1.isFreed());

        // we check if result buffer is freed too.
        assertTrue(buffer2.isFreed());

        /*
            PLEASE IGNORE! THIS COMMENT WAS WRITTEN PRIOR TO FIX AND KEPT FOR HISTORIC PURPOSES ONLY:
            As you can see, this test fails here - underlying buffer of size 200000 elements hasn't got cudaFreeHost call
            That happens due to logical flaw in BaseCudaDataBuffer.free() method.
            And since try-with-resource is nothing more then auto-call for close() method, overall idea is flawed by this delegation.

            From now on, this buffer will stay allocated until application is terminated, however all subsequent view allocations will return proper pointers to this buffer + offset.
        */
        /*
            Now we know, that array1 and array2 backing buffers were freed, since there were no more references left.
            And following sile calls will cause new allocation for both baseArray1 and baseArray2, that's not good from performance point of view, but still valid from allocation accuracy.
            And this should be considered as first stop, when it comes to performance improvements over current memory model.
         */

        slice1 = baseArray1.slice(2);
        slice2 = baseArray2.slice(2);

        ctx = CudaContext.getBlasContext();

        /*
            Since our backing buffer allocated for original array1/array2 was NOT freed, we'll obtain offset pointers to the buffer allocated on previous step.
        */
        xAPointer = new CublasPointer(slice1,ctx);
        xBPointer = new CublasPointer(slice2,ctx);

        // at this point we should have equal mem pointers to underlying buffers
        long new_addr_buff1 = xAPointer.getDevicePointer().getNativePointer();
        long new_addr_buff2 = xBPointer.getDevicePointer().getNativePointer();


        // please note, this equation is NOT means bug if it fails, in c world you can have two equal pointers that references exactly same memory address
        // so java buffers reality should be investigated before removing comments
        /*
            assertEquals(addr_buff1, new_addr_buff1);
            assertEquals(addr_buff2, new_addr_buff2);
        */
        xAPointer.close();
        xBPointer.close();

        assertEquals(true, xAPointer.isClosed());
        assertEquals(true, xBPointer.isClosed());

        // now we should check, if host memory was released.
        // if not - freeHost() wasn't called for corresponding buffer and we have memory leak there
        assertTrue(buffer1.isFreed());

        // we check if result buffer is freed too.
        assertTrue(buffer2.isFreed());

        // make sure buffer got 0 references left
        assertEquals(0, buffer1.getPointersToContexts().size());
        assertEquals(0, buffer2.getPointersToContexts().size());

    }


    /**
     * This test makes sure that data is transferred host->device->host path properly.
     * To check that, we use pre-calculated dot product validation
     *
     * @throws Exception
     */
    @Test
    public void testPinnedBlasCallValue1() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        // reset to default MemoryStrategy, most probable is Pinned
        ContextHolder.getInstance().forceMemoryStrategyForThread(new PinnedMemoryStrategy());

        assertEquals("PinnedMemoryStrategy", ContextHolder.getInstance().getMemoryStrategy().getClass().getSimpleName());

        CudaContext ctx = CudaContext.getBlasContext();

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f});


        double dotWrapped = Nd4j.getBlasWrapper().dot(array1, array2);

        CublasPointer xAPointer = new CublasPointer(array1,ctx);
        CublasPointer xBPointer = new CublasPointer(array2,ctx);

        float[] ret = new float[1];
        ret[0] = 0;
        Pointer result = Pointer.to(ret);


        JCublas2.cublasSdot(
                ctx.getHandle(),
                array1.length(),
                xAPointer.getDevicePointer().withByteOffset(array1.offset() * array1.data().getElementSize()),
                BlasBufferUtil.getBlasStride(array1),
                xBPointer.getDevicePointer().withByteOffset(array2.offset() * array2.data().getElementSize()),
                BlasBufferUtil.getBlasStride(array2),
                result);
        ctx.syncOldStream();



        ctx.finishBlasOperation();


        double res = ret[0]; //  / (norm1.doubleValue() * norm2.doubleValue());

        System.out.println("Val before: [0], after: ["+ ret[0]+"], norm: [" + res +"], dotWrapped: [" + dotWrapped + "]");

        xAPointer.close();
        xBPointer.close();

        assertEquals(dotWrapped, res, 0.001d);

        // we compare result against precalculated value
        assertEquals(16.665000915527344, res, 0.001d);
    }


    /**
     * This test makes sure that data is transferred host->device->host path properly.
     * To check that, we use pre-calculated axpy product validation
     *
     * @throws Exception
     */
    @Test
    public void testPinnedBlasCallValue2() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        // reset to default MemoryStrategy, most probable is Pinned
        ContextHolder.getInstance().forceMemoryStrategyForThread(new PinnedMemoryStrategy());

        assertEquals("PinnedMemoryStrategy", ContextHolder.getInstance().getMemoryStrategy().getClass().getSimpleName());

        CudaContext ctx = CudaContext.getBlasContext();

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});

        CublasPointer xAPointer = new CublasPointer(array1,ctx);
        CublasPointer xBPointer = new CublasPointer(array2,ctx);

        JCublas2.cublasSaxpy(
                ctx.getHandle(),
                array1.length(),
                Pointer.to(new float[]{0.75f}),
                xAPointer.getDevicePointer().withByteOffset(array1.offset() * array1.data().getElementSize()),
                BlasBufferUtil.getBlasStride(array1),
                xBPointer.getDevicePointer().withByteOffset(array2.offset() * array2.data().getElementSize()),
                BlasBufferUtil.getBlasStride(array2));
        ctx.syncOldStream();

        xBPointer.copyToHost();

        ctx.finishBlasOperation();

        double result1 = array2.getDouble(0);
        double result2 = array2.getDouble(1);

        xAPointer.close();
        xBPointer.close();

        System.out.println("Value[0]: " + result1);
        System.out.println("Value[1]: " + result2);

        // we're checking two "random" elements of result array against precalculated values
        assertEquals(1.7574999332427979, result1, 0.00001);
        assertEquals(1.7574999332427979, result2, 0.00001);
    }



    /**
     * This test makes sure that data is transferred host->device->host path properly.
     * To check that, we use pre-calculated axpy product validation
     *
     * @throws Exception
     */
    @Test
    public void testPageableBlasCallValue2() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        // setting Pageable memory strategy
        ContextHolder.getInstance().forceMemoryStrategyForThread(new PageableDirectBufferMemoryStrategy());

        assertEquals("PageableDirectBufferMemoryStrategy", ContextHolder.getInstance().getMemoryStrategy().getClass().getSimpleName());

        CudaContext ctx = CudaContext.getBlasContext();

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});

        CublasPointer xAPointer = new CublasPointer(array1,ctx);
        CublasPointer xBPointer = new CublasPointer(array2,ctx);

        JCublas2.cublasSaxpy(
                ctx.getHandle(),
                array1.length(),
                Pointer.to(new float[]{0.75f}),
                xAPointer.getDevicePointer().withByteOffset(array1.offset() * array1.data().getElementSize()),
                BlasBufferUtil.getBlasStride(array1),
                xBPointer.getDevicePointer().withByteOffset(array2.offset() * array2.data().getElementSize()),
                BlasBufferUtil.getBlasStride(array2));
        ctx.syncOldStream();

        xBPointer.copyToHost();

        ctx.finishBlasOperation();

        double result1 = array2.getDouble(0);
        double result2 = array2.getDouble(1);

        xAPointer.close();
        xBPointer.close();

        System.out.println("Value[0]: " + result1);
        System.out.println("Value[1]: " + result2);

        // we checking two elements of result array with precalculated value
        assertEquals(1.7574999332427979, result1, 0.00001);
        assertEquals(1.7574999332427979, result2, 0.00001);
    }


    /**
     * This test makes sure that data is transferred host->device->host path properly.
     * To check that, we use pre-calculated dot product validation
     *
     * @throws Exception
     */
    @Test
    public void testPageableBlasCallValue1() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        // reset to default MemoryStrategy, most probable is Pinned
        ContextHolder.getInstance().forceMemoryStrategyForThread(new PageableDirectBufferMemoryStrategy());

        assertEquals("PageableDirectBufferMemoryStrategy", ContextHolder.getInstance().getMemoryStrategy().getClass().getSimpleName());

        CudaContext ctx = CudaContext.getBlasContext();

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f});


        double dotWrapped = Nd4j.getBlasWrapper().dot(array1, array2);

        CublasPointer xAPointer = new CublasPointer(array1,ctx);
        CublasPointer xBPointer = new CublasPointer(array2,ctx);

        float[] ret = new float[1];
        ret[0] = 0;
        Pointer result = Pointer.to(ret);


        JCublas2.cublasSdot(
                ctx.getHandle(),
                array1.length(),
                xAPointer.getDevicePointer().withByteOffset(array1.offset() * array1.data().getElementSize()),
                BlasBufferUtil.getBlasStride(array1),
                xBPointer.getDevicePointer().withByteOffset(array2.offset() * array2.data().getElementSize()),
                BlasBufferUtil.getBlasStride(array2),
                result);
        ctx.syncOldStream();



        ctx.finishBlasOperation();


        double res = ret[0]; //  / (norm1.doubleValue() * norm2.doubleValue());

        System.out.println("Val before: [0], after: ["+ ret[0]+"], norm: [" + res +"], dotWrapped: [" + dotWrapped + "]");

        xAPointer.close();
        xBPointer.close();

        assertEquals(dotWrapped, res, 0.001d);

        // compare result to precalculated value
        assertEquals(16.665000915527344, res, 0.001d);
    }

    /**
     * This test makes sure that data is transferred host->device->host path properly.
     * To check that, we use pre-calculated dot product validation
     *
     * @throws Exception
     */
    @Test
    public void tesPinnedBlasCallValue1() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        // reset to default MemoryStrategy, most probable is Pinned
        ContextHolder.getInstance().forceMemoryStrategyForThread(new PinnedMemoryStrategy());

        assertEquals("PinnedMemoryStrategy", ContextHolder.getInstance().getMemoryStrategy().getClass().getSimpleName());

        CudaContext ctx = CudaContext.getBlasContext();

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f, 1.10f});


        double dotWrapped = Nd4j.getBlasWrapper().dot(array1, array2);

        CublasPointer xAPointer = new CublasPointer(array1,ctx);
        CublasPointer xBPointer = new CublasPointer(array2,ctx);

        float[] ret = new float[1];
        Pointer result = Pointer.to(ret);

        System.out.println("Offset: ["+ array1.offset()+"], stride: ["+ BlasBufferUtil.getBlasStride(array1)+"]");
        System.out.println("Offset: ["+ array2.offset()+"], stride: ["+ BlasBufferUtil.getBlasStride(array2)+"]");

        JCublas2.cublasSdot(
                    ctx.getHandle(),
                    array1.length(),
                    xAPointer.getDevicePointer().withByteOffset(array1.offset() * array1.data().getElementSize()),
                    BlasBufferUtil.getBlasStride(array1),
                    xBPointer.getDevicePointer().withByteOffset(array2.offset() * array2.data().getElementSize()),
                    BlasBufferUtil.getBlasStride(array2),
                    result);
        ctx.syncOldStream();

        xAPointer.close();
        xBPointer.close();

        ctx.finishBlasOperation();



        double res = ret[0]; //  / (norm1.doubleValue() * norm2.doubleValue());


        System.out.println("Val before: [0], after: ["+ ret[0]+"], norm: [" + res +"], wrapped: [" + dotWrapped + "]");
        assertEquals(dotWrapped, res, 0.001d);

        // compare result to pre-calculated value
        assertEquals(16.665000915527344, res, 0.001d);

    }

    /**
     * This test is suited for test of multiple consequent allocations over the same original buffer.
     *
     * Basic idea: We have large array, we get a slice (view), it receives a pointer to the original buffer + offset.
     * After view is released, original buffer should be released ONLY if it has NO more offset references left
     *
     * @throws Exception
     */
    @Test
    public void testPinnedMemoryNestedAllocation1() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        // reset to default MemoryStrategy, most probable is Pinned
        ContextHolder.getInstance().forceMemoryStrategyForThread(new PinnedMemoryStrategy());

        assertEquals("PinnedMemoryStrategy", ContextHolder.getInstance().getMemoryStrategy().getClass().getSimpleName());

        INDArray baseArray1 = Nd4j.rand(new int[]{1000, 200}, -1.0, 1.0, new DefaultRandom());

        // this is not a typo, we test situation where we process on views allocated on the same offsets
        INDArray slice1 = baseArray1.slice(1);
        INDArray slice2 = baseArray1.slice(1);

        CudaContext ctx = CudaContext.getBlasContext();

        CublasPointer xAPointer = new CublasPointer(slice1,ctx);
        CublasPointer xBPointer = new CublasPointer(slice2,ctx);


        BaseCudaDataBuffer buffer1 = (BaseCudaDataBuffer) xAPointer.getBuffer();
        BaseCudaDataBuffer buffer2 = (BaseCudaDataBuffer) xBPointer.getBuffer();

        // both buffers should NOT be freed, since second pointer to the same array still exists
        xAPointer.close();
        assertFalse(buffer1.isFreed());
        assertFalse(buffer2.isFreed());


        xBPointer.close();

        // now, when second pointer was closed, both buffers should be free too
        assertTrue(buffer1.isFreed());
        assertTrue(buffer2.isFreed());

        // make sure buffer got 0 references left
        assertEquals(0, buffer1.getPointersToContexts().size());
        assertEquals(0, buffer1.getPointersToContexts().size());
    }


    /**
     * This test is suited for test of multiple consequent allocations over the same original buffer.
     *
     * Basic idea: We have large array, we get a slice (view), it receives a pointer to the original buffer + offset.
     * After view is released, original buffer should be released ONLY if it has NO more offset references left
     *
     * @throws Exception
     */
    @Test
    public void testPinnedMemoryNestedAllocation2() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        // reset to default MemoryStrategy, most probable is Pinned
        ContextHolder.getInstance().forceMemoryStrategyForThread(new PinnedMemoryStrategy());

        assertEquals("PinnedMemoryStrategy", ContextHolder.getInstance().getMemoryStrategy().getClass().getSimpleName());

        INDArray baseArray1 = Nd4j.rand(new int[]{1000, 200}, -1.0, 1.0, new DefaultRandom());

        INDArray slice1 = baseArray1.slice(1);
        INDArray slice2 = baseArray1.slice(2);

        CudaContext ctx = CudaContext.getBlasContext();

        CublasPointer xAPointer = new CublasPointer(slice1,ctx);
        CublasPointer xBPointer = new CublasPointer(slice2,ctx);


        BaseCudaDataBuffer buffer1 = (BaseCudaDataBuffer) xAPointer.getBuffer();
        BaseCudaDataBuffer buffer2 = (BaseCudaDataBuffer) xBPointer.getBuffer();

        // both buffers should NOT be freed, since second pointer to the same original array still exists
        xAPointer.close();
        assertFalse(buffer1.isFreed());
        assertFalse(buffer2.isFreed());


        xBPointer.close();

        // now, both buffers should be discarded
        assertTrue(buffer1.isFreed());
        assertTrue(buffer2.isFreed());

        // make sure buffer got 0 references left
        assertEquals(0, buffer1.getPointersToContexts().size());
        assertEquals(0, buffer2.getPointersToContexts().size());
    }

    /**
     * This test is suited for test of multiple consequent allocations over the same original buffer.
     *
     * Basic idea: We have large array, we get a slice (view), it receives a pointer to the original buffer + offset.
     * After view is released, original buffer should be released ONLY if it has NO more offset references left
     *
     * @throws Exception
     */
    @Test
    public void testPinnedMemoryNestedAllocation3() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        // reset to default MemoryStrategy, most probable is Pinned
        ContextHolder.getInstance().forceMemoryStrategyForThread(new PinnedMemoryStrategy());

        assertEquals("PinnedMemoryStrategy", ContextHolder.getInstance().getMemoryStrategy().getClass().getSimpleName());

        INDArray baseArray1 = Nd4j.rand(new int[]{1000, 200}, -1.0, 1.0, new DefaultRandom());

        INDArray slice1 = baseArray1.slice(1);
        INDArray slice2 = baseArray1.slice(2);

        // please note, slice(1) is not a typo here
        INDArray slice3 = baseArray1.slice(1);

        CudaContext ctx = CudaContext.getBlasContext();

        CublasPointer xAPointer = new CublasPointer(slice1,ctx);
        CublasPointer xBPointer = new CublasPointer(slice2,ctx);
        CublasPointer xCPointer = new CublasPointer(slice3,ctx);


        BaseCudaDataBuffer buffer1 = (BaseCudaDataBuffer) xAPointer.getBuffer();
        BaseCudaDataBuffer buffer2 = (BaseCudaDataBuffer) xBPointer.getBuffer();
        BaseCudaDataBuffer buffer3 = (BaseCudaDataBuffer) xCPointer.getBuffer();

        // all three buffers should NOT be freed, since second & third pointers to the same array still exist
        xAPointer.close();
        assertFalse(buffer1.isFreed());
        assertFalse(buffer2.isFreed());
        assertFalse(buffer3.isFreed());

        // the same, last pointer still alive and valid
        xBPointer.close();
        assertFalse(buffer1.isFreed());
        assertFalse(buffer2.isFreed());
        assertFalse(buffer3.isFreed());

        // now everything should be closed
        xCPointer.close();
        assertTrue(buffer1.isFreed());
        assertTrue(buffer2.isFreed());
        assertTrue(buffer3.isFreed());


        // make sure buffer got 0 references left
        assertEquals(0, buffer1.getPointersToContexts().size());
        assertEquals(0, buffer2.getPointersToContexts().size());
        assertEquals(0, buffer3.getPointersToContexts().size());
    }


    /**
     * This test is suited for test of multiple consequent allocations over the same original buffer.
     *
     * Basic idea: We have large array, we get a slice (view), it receives a pointer to the original buffer + offset.
     * After view is released, original buffer should be released ONLY if it has NO more offset references left
     *
     * @throws Exception
     */
    @Test
    public void testPageableMemoryNestedAllocation3() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        // reset to default MemoryStrategy, most probable is Pinned
        ContextHolder.getInstance().forceMemoryStrategyForThread(new PageableDirectBufferMemoryStrategy());

        assertEquals("PageableDirectBufferMemoryStrategy", ContextHolder.getInstance().getMemoryStrategy().getClass().getSimpleName());

        INDArray baseArray1 = Nd4j.rand(new int[]{1000, 200}, -1.0, 1.0, new DefaultRandom());

        INDArray slice1 = baseArray1.slice(1);
        INDArray slice2 = baseArray1.slice(2);

        // please note, slice(1) is not a typo here
        INDArray slice3 = baseArray1.slice(1);

        CudaContext ctx = CudaContext.getBlasContext();

        CublasPointer xAPointer = new CublasPointer(slice1,ctx);
        CublasPointer xBPointer = new CublasPointer(slice2,ctx);
        CublasPointer xCPointer = new CublasPointer(slice3,ctx);


        BaseCudaDataBuffer buffer1 = (BaseCudaDataBuffer) xAPointer.getBuffer();
        BaseCudaDataBuffer buffer2 = (BaseCudaDataBuffer) xBPointer.getBuffer();
        BaseCudaDataBuffer buffer3 = (BaseCudaDataBuffer) xCPointer.getBuffer();

        // all three buffers should NOT be freed, since second & third pointers to the same array still exist
        xAPointer.close();
        assertFalse(buffer1.isFreed());
        assertFalse(buffer2.isFreed());
        assertFalse(buffer3.isFreed());

        // the same, last pointer still alive and valid
        xBPointer.close();
        assertFalse(buffer1.isFreed());
        assertFalse(buffer2.isFreed());
        assertFalse(buffer3.isFreed());

        // now everything should be closed
        xCPointer.close();
        assertTrue(buffer1.isFreed());
        assertTrue(buffer2.isFreed());
        assertTrue(buffer3.isFreed());


        // make sure buffer got 0 references left
        assertEquals(0, buffer1.getPointersToContexts().size());
        assertEquals(0, buffer2.getPointersToContexts().size());
        assertEquals(0, buffer3.getPointersToContexts().size());
    }

    /**
     * This test addresses Pinned memory allocation using given buffer
     *
     * Basic idea: allocate, call axpy, copyback, free. After that original array2 should contain updated values, and buffer should be released.
     *
     * @throws Exception
     */
    @Test
    public void testPinnedBufferBasedAllocation1() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        // reset to default MemoryStrategy, most probable is Pinned
        ContextHolder.getInstance().forceMemoryStrategyForThread(new PinnedMemoryStrategy());

        assertEquals("PinnedMemoryStrategy", ContextHolder.getInstance().getMemoryStrategy().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});

        BaseCudaDataBuffer buffer2 = (BaseCudaDataBuffer) array2.data();

        CudaContext ctx = CudaContext.getBlasContext();

        CublasPointer xAPointer = new CublasPointer(array1, ctx);

        // please note, this pointer is targeting buffer
        CublasPointer xBPointer = new CublasPointer(buffer2, ctx);

        // now we assume we have pointer that can't have derived pointers
        // we'll call for blas, since we can't do cudaMemset on pinned memory

        JCublas2.cublasSaxpy(
                ctx.getHandle(),
                array1.length(),
                Pointer.to(new float[]{0.95f}),
                xAPointer.getDevicePointer().withByteOffset(array1.offset() * array1.data().getElementSize()),
                BlasBufferUtil.getBlasStride(array1),
                xBPointer.getDevicePointer().withByteOffset(array2.offset() * array2.data().getElementSize()),
                BlasBufferUtil.getBlasStride(array2));
        ctx.syncOldStream();

        // copyback our buffer
        xBPointer.copyToHost();

        ctx.finishBlasOperation();

        xAPointer.close();
        xBPointer.close();

        assertTrue(buffer2.isFreed());
        assertTrue(xBPointer.isClosed());
        assertTrue(xAPointer.isClosed());

        // so, we should have full array filled with zeroes now, not 1.01f's
        System.out.println("Value[0]: " + array2.getFloat(0));
        System.out.println("Value[1]: " + array2.getFloat(1));

        assertNotEquals(1.01f, array2.getFloat(0), 0.001f);
        assertNotEquals(1.01f, array2.getFloat(1), 0.001f);

        assertEquals(0, buffer2.getPointersToContexts().size());
    }


    /**
     * This test addresses Pageable memory allocation using given buffer
     *
     * Basic idea: allocate, memset, copyback, free.At the end of day host buffer should have updated value, and buffer should be released.
     *
     * @throws Exception
     */
    @Test
    public void testPageableBufferBasedAllocation1() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        // reset to default MemoryStrategy, most probable is Pinned
        ContextHolder.getInstance().forceMemoryStrategyForThread(new PageableDirectBufferMemoryStrategy());

        assertEquals("PageableDirectBufferMemoryStrategy", ContextHolder.getInstance().getMemoryStrategy().getClass().getSimpleName());

        INDArray array = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});

        BaseCudaDataBuffer buffer = (BaseCudaDataBuffer) array.data();

        //
        CudaContext ctx = CudaContext.getBlasContext();

        CublasPointer xBPointer = new CublasPointer(buffer, ctx);

        // now we assume we have pointer that can't have derived pointers
        // and now we'll just memset everything with zeroes, so actually whole array will become array of zeroes internally
        JCuda.cudaMemset(xBPointer.getDevicePointer(), 0, buffer.length() * buffer.getElementSize());

        // copyback our buffer
        xBPointer.copyToHost();

        ctx.finishBlasOperation();

        xBPointer.close();

        assertTrue(buffer.isFreed());
        assertTrue(xBPointer.isClosed());

        // so, we should have full array filled with zeroes now, not 1.01f's
        assertEquals(0.0d, array.getDouble(0), 0.001);
        assertEquals(0.0d, array.getDouble(1), 0.001);

        /*
            PLEASE NOTE: THERE SHOULD BE EXCEPTION.

            This part of test should throw exception with IllegalDevicePointer description.
            If everything is released properly, device pointer derived via CublasPointer() should be discarded and freed
         */
        try {
            JCuda.cudaMemset(xBPointer.getDevicePointer(), 0, buffer.length() * buffer.getElementSize());

            // if there's something wrong with buffer release, we should throw assertion exception
            assertTrue(false);
        } catch (jcuda.CudaException e) {
            // everything ok, device memory were released
            ;
        }

        // make sure buffer got 0 references left
        assertEquals(0, buffer.getPointersToContexts().size());
    }

    /**
     *  This test addresses memory allocation for arrays with stride > 1, since that's important feature for nd4j internal memory allocation.
     *
     *  Basic idea: allocate 2D array, get column view, and it will have stride > 1. After view is used and released, whole array should be released too, since it has no more usages left.
     *
     *
     */
    @Test
    public void testPinnedMemoryStridedSlice() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        // reset to default MemoryStrategy, most probable is Pinned
        ContextHolder.getInstance().forceMemoryStrategyForThread(new PinnedMemoryStrategy());

        assertEquals("PinnedMemoryStrategy", ContextHolder.getInstance().getMemoryStrategy().getClass().getSimpleName());

        // we create 2D array, that will be used for vertically sliced arrays
        INDArray baseArray = Nd4j.create(100, 200);

        // our slice has offset 10, and stride 200, length 100
        INDArray slice1 = baseArray.getColumn(10);

        CudaContext ctx = CudaContext.getBlasContext();

        CublasPointer xAPointer = new CublasPointer(slice1, ctx);

        BaseCudaDataBuffer buffer = (BaseCudaDataBuffer) xAPointer.getBuffer();

        // we have pointer allocated

        /*
            Now if we'll close this pointer, it won't be really released, since freeDevicePointer() call assumes stride size 0, and we have stride 200
         */

        xAPointer.close();

        // buffer must be free
        assertTrue(buffer.isFreed());

        // make sure buffer got 0 references left
        assertEquals(0, buffer.getPointersToContexts().size());
    }

    /**
     *  This test addresses memory allocation for arrays with stride > 1, since that's important value for nd4j internal memory allocation
     */
    @Test
    public void testPageableMemoryStridedSlice() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        // reset to default MemoryStrategy, most probable is Pinned
        ContextHolder.getInstance().forceMemoryStrategyForThread(new PageableDirectBufferMemoryStrategy());

        assertEquals("PageableDirectBufferMemoryStrategy", ContextHolder.getInstance().getMemoryStrategy().getClass().getSimpleName());

        // we create 2D array, that will be used for vertically sliced arrays
        INDArray baseArray = Nd4j.create(100, 200);

        // our slice has offset 10, and stride 200, length 100
        INDArray slice1 = baseArray.getColumn(10);

        CudaContext ctx = CudaContext.getBlasContext();

        CublasPointer xAPointer = new CublasPointer(slice1, ctx);

        BaseCudaDataBuffer buffer = (BaseCudaDataBuffer) xAPointer.getBuffer();

        // we have pointer allocated

        /*
            Now if we'll close this pointer, it won't be really released, since freeDevicePointer() call assumes stride size 0, and we have stride 200
         */

        xAPointer.close();

        // buffer must be free
        assertTrue(buffer.isFreed());

        // make sure buffer got 0 references left
        assertEquals(0, buffer.getPointersToContexts().size());
    }


    /**
     * This test is suited for primitive check for slices derived from N-dimensional arrays
     *
     * @throws Exception
     */
    @Test
    public void testPinned3DStride1() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        // reset to default MemoryStrategy, most probable is Pinned
        ContextHolder.getInstance().forceMemoryStrategyForThread(new PinnedMemoryStrategy());

        assertEquals("PinnedMemoryStrategy", ContextHolder.getInstance().getMemoryStrategy().getClass().getSimpleName());

        // we create 3D array, that will be used for vertically sliced arrays
        INDArray baseArray = Nd4j.create(100, 200, 300);

        INDArray slice = baseArray.slice(10,2);

        System.out.println("Slice length: ["+ slice.length()+"], offset: ["+ slice.offset()+"], columns: ["+ slice.columns()+"], rows: ["+ slice.rows()+"] ");


        CudaContext ctx = CudaContext.getBlasContext();

        CublasPointer xAPointer = new CublasPointer(slice, ctx);

        BaseCudaDataBuffer buffer = (BaseCudaDataBuffer) xAPointer.getBuffer();

        // we should have full buffer allocated
        assertEquals(6000000, buffer.length());

        /*
            Now if we'll close this pointer, it won't be really released, since freeDevicePointer() call assumes stride size 0, and we have stride 200
         */

        xAPointer.close();

        // buffer must be free
        assertTrue(buffer.isFreed());

        // make sure buffer got 0 references left
        assertEquals(0, buffer.getPointersToContexts().size());
    }
}

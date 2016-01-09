package jcuda.jcublas;

import jcuda.Pointer;
import jcuda.runtime.JCuda;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.blas.BlasBufferUtil;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.Assert;
import org.nd4j.linalg.jcublas.CublasPointer;
import org.nd4j.linalg.jcublas.JCublasBackend;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.*;

/**
 * This set of tests will check for
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
     *
     */
    @Test
    public void testPageableMemoryRelease() throws Exception {

    }

    @Test
    public void testPinnedMemoryRelease() throws Exception {

        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

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

        assertEquals(15, buffer.length());

        CublasPointer yCPointer = new CublasPointer(array2,ctx);

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

        // Please note: we do NOT test result pointer here,since we assume it's handled by JCuda
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

        INDArray baseArray1 = Nd4j.rand(new int[]{1000, 200}, -1.0, 1.0, new DefaultRandom());
        INDArray baseArray2 = Nd4j.rand(new int[]{1000, 200}, -1.0, 1.0, new DefaultRandom());

        INDArray slice1 = baseArray1.slice(1);
        INDArray slice2 = baseArray2.slice(1);

        CudaContext ctx = CudaContext.getBlasContext();

        CublasPointer xAPointer = new CublasPointer(slice1,ctx);

        BaseCudaDataBuffer buffer1 = (BaseCudaDataBuffer) xAPointer.getBuffer();

        assertEquals(DataBuffer.AllocationMode.DIRECT, buffer1.allocationMode());
        assertTrue(buffer1.copied(Thread.currentThread().getName()));
        assertFalse(buffer1.isPersist());

        // for sliced view we have whole original array allocated
        assertEquals(200000, buffer1.length());


        CublasPointer xBPointer = new CublasPointer(slice2,ctx);

        BaseCudaDataBuffer buffer2 = (BaseCudaDataBuffer) xBPointer.getBuffer();

        assertEquals(200000, buffer2.length());



        JCublas2.cublasDaxpy(
                ctx.getHandle(),
                slice1.length(),
                Pointer.to(new double[]{1.0}),
                xAPointer.getDevicePointer().withByteOffset(slice1.offset() * slice1.data().getElementSize()),
                BlasBufferUtil.getBlasStride(slice1),
                xBPointer.getDevicePointer().withByteOffset(slice2.offset() * slice2.data().getElementSize()),
                BlasBufferUtil.getBlasStride(slice2));
        ctx.syncOldStream();

        //now, since we have result array, we call for explicit copyback
        double valBefore = slice2.getDouble(0);

        xBPointer.copyToHost();

        // we don't care if the result is true here. All we want to know is: if memory was really updated after copyback
        double valAfter = slice2.getDouble(0);
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

    }

}

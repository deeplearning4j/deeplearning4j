package jcuda.jcublas;

import jcuda.Pointer;
import jcuda.runtime.JCuda;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.blas.BlasBufferUtil;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
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
    public void testPinnedMemoryReleaseDevicePointer() throws Exception {
        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});

        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

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
        // if not - freeHost() wasn't called for corresponding buffer
        assertTrue(buffer.isFreed());
    }

}

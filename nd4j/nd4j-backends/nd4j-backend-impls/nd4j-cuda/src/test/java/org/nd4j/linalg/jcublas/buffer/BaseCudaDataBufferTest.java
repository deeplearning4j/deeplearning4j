package org.nd4j.linalg.jcublas.buffer;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.util.PrintVariable;
import org.nd4j.linalg.factory.Nd4j;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.*;

@Slf4j
public class BaseCudaDataBufferTest extends BaseND4JTest {

    @Before
    public void setUp() {
        //
    }

    @Test
    public void testBasicAllocation_1() {
        val array = Nd4j.create(DataType.FLOAT, 5);

        // basic validation
        assertNotNull(array);
        assertNotNull(array.data());
        assertNotNull(((BaseCudaDataBuffer) array.data()).getOpaqueDataBuffer());

        // shape part
        assertArrayEquals(new long[]{1, 5, 1, 8192, 1, 99}, array.shapeInfoJava());
        assertArrayEquals(new long[]{1, 5, 1, 8192, 1, 99}, array.shapeInfoDataBuffer().asLong());

        // arrat as full of zeros at this point
        assertArrayEquals(new float[] {0.f, 0.f, 0.f, 0.f, 0.f}, array.data().asFloat(), 1e-5f);
    }

    @Test
    public void testBasicAllocation_2() {
        val array = Nd4j.createFromArray(1.f, 2.f, 3.f, 4.f, 5.f);

        // basic validation
        assertNotNull(array);
        assertNotNull(array.data());
        assertNotNull(((BaseCudaDataBuffer) array.data()).getOpaqueDataBuffer());

        // shape part
        assertArrayEquals(new long[]{1, 5, 1, 8192, 1, 99}, array.shapeInfoJava());
        assertArrayEquals(new long[]{1, 5, 1, 8192, 1, 99}, array.shapeInfoDataBuffer().asLong());

        // arrat as full of values at this point
        assertArrayEquals(new float[] {1.f, 2.f, 3.f, 4.f, 5.f}, array.data().asFloat(), 1e-5f);
    }

    @Test
    public void testBasicView_1() {
        val array = Nd4j.createFromArray(1.f, 2.f, 3.f, 4.f, 5.f, 6.f).reshape(3, 2);

        // basic validation
        assertNotNull(array);
        assertNotNull(array.data());
        assertNotNull(((BaseCudaDataBuffer) array.data()).getOpaqueDataBuffer());

        // checking TAD equality
        val row = array.getRow(1);
        assertArrayEquals(new float[]{3.0f, 4.0f}, row.data().dup().asFloat(), 1e-5f);
    }

    @Test
    public void testScalar_1() {
        val scalar = Nd4j.scalar(119.f);

        // basic validation
        assertNotNull(scalar);
        assertNotNull(scalar.data());
        assertNotNull(((BaseCudaDataBuffer) scalar.data()).getOpaqueDataBuffer());

        // shape part
        assertArrayEquals(new long[]{0, 8192, 1, 99}, scalar.shapeInfoJava());
        assertArrayEquals(new long[]{0, 8192, 1, 99}, scalar.shapeInfoDataBuffer().asLong());

        // pointers part
        val devPtr = AtomicAllocator.getInstance().getPointer(scalar.data());
        val hostPtr = AtomicAllocator.getInstance().getHostPointer(scalar.data());

        // dev pointer supposed to exist, and host pointer is not
        assertNotNull(devPtr);
        assertNull(hostPtr);

        assertEquals(119.f, scalar.getFloat(0), 1e-5f);
    }

    @Test
    public void testSerDe_1() throws Exception {
        val array = Nd4j.createFromArray(1.f, 2.f, 3.f, 4.f, 5.f, 6.f);
        val baos = new ByteArrayOutputStream();

        Nd4j.write(baos, array);
        INDArray restored = Nd4j.read(new ByteArrayInputStream(baos.toByteArray()));

        // basic validation
        assertNotNull(restored);
        assertNotNull(restored.data());
        assertNotNull(((BaseCudaDataBuffer) restored.data()).getOpaqueDataBuffer());

        // shape part
        assertArrayEquals(new long[]{1, 6, 1, 8192, 1, 99}, restored.shapeInfoJava());
        assertArrayEquals(new long[]{1, 6, 1, 8192, 1, 99}, restored.shapeInfoDataBuffer().asLong());

        // data equality
        assertArrayEquals(array.data().asFloat(), restored.data().asFloat(), 1e-5f);
    }

    @Test
    public void testBasicOpInvocation_1() {
        val array1 = Nd4j.createFromArray(1.f, 2.f, 3.f, 4.f, 5.f, 6.f);
        val array2 = Nd4j.createFromArray(1.f, 2.f, 3.f, 4.f, 5.f, 6.f);

        // shape pointers must be equal here
        val devPtr1 = AtomicAllocator.getInstance().getPointer(array1.shapeInfoDataBuffer());
        val devPtr2 = AtomicAllocator.getInstance().getPointer(array2.shapeInfoDataBuffer());

        val hostPtr1 = AtomicAllocator.getInstance().getHostPointer(array1.shapeInfoDataBuffer());
        val hostPtr2 = AtomicAllocator.getInstance().getHostPointer(array2.shapeInfoDataBuffer());

        // pointers must be equal on host and device, since we have shape cache
        assertEquals(devPtr1.address(), devPtr2.address());
        assertEquals(hostPtr1.address(), hostPtr2.address());

        assertEquals(array1, array2);
    }

    @Test
    public void testBasicOpInvocation_2() {
        val array1 = Nd4j.createFromArray(1.f, 200.f, 3.f, 4.f, 5.f, 6.f);
        val array2 = Nd4j.createFromArray(1.f, 2.f, 3.f, 4.f, 5.f, 6.f);

        assertNotEquals(array1, array2);
    }

    @Test
    public void testBasicOpInvocation_3() {
        val array = Nd4j.create(DataType.FLOAT, 6);
        val exp = Nd4j.createFromArray(1.f, 1.f, 1.f, 1.f, 1.f, 1.f);

        array.addi(1.0f);

        assertEquals(exp, array);
    }

    @Test
    public void testCustomOpInvocation_1() {
        val array = Nd4j.createFromArray(1.f, 2.f, 3.f, 4.f, 5.f, 6.f);

        Nd4j.exec(new PrintVariable(array, true));
        Nd4j.exec(new PrintVariable(array));
    }

    @Test
    public void testMultiDeviceMigration_1() throws Exception {
        if (Nd4j.getAffinityManager().getNumberOfDevices() < 2)
            return;

        // creating all arrays within main thread context
        val list = new ArrayList<INDArray>();
        for (int e = 0; e < Nd4j.getAffinityManager().getNumberOfDevices(); e++)
            list.add(Nd4j.create(DataType.FLOAT, 3, 5));

        val cnt = new AtomicInteger(0);

        // now we're creating threads
        for (int e = 0; e < Nd4j.getAffinityManager().getNumberOfDevices(); e++) {
            val f = e;
            val t = new Thread(new Runnable() {
                @Override
                public void run() {
                    // issuing one operation, just to see how migration works
                    list.get(f).addi(1.0f);

                    // synchronizing immediately
                    Nd4j.getExecutioner().commit();
                    cnt.incrementAndGet();
                }
            });

            t.start();
            t.join();
        }

        // there shoul dbe no exceptions during execution
        assertEquals(Nd4j.getAffinityManager().getNumberOfDevices(), cnt.get());
    }
}
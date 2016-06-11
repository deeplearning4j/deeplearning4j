package org.nd4j.jita.memory.impl;

import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.pointers.PointersPair;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
@Ignore
public class CudaDirectProviderTest {

    @Test
    public void mallocHost() throws Exception {
        CudaDirectProvider provider = new CudaDirectProvider();

        AllocationShape shape = new AllocationShape(100000, 4);
        AllocationPoint point = new AllocationPoint();
        point.setShape(shape);



        point.setPointers(provider.malloc(shape, point, AllocationStatus.HOST));

        System.out.println("Allocated...");
        Thread.sleep(1000);


        provider.free(point);

        System.out.println("Deallocated...");
        Thread.sleep(1000);
    }

    @Test
    public void mallocDevice() throws Exception {
        CudaDirectProvider provider = new CudaDirectProvider();

        AllocationShape shape = new AllocationShape(300000, 4);
        AllocationPoint point = new AllocationPoint();
        point.setShape(shape);


        point.setPointers(provider.malloc(shape, point, AllocationStatus.DEVICE));

        System.out.println("Allocated...");
        Thread.sleep(1000);


        point.setAllocationStatus(AllocationStatus.DEVICE);

        provider.free(point);

        System.out.println("Deallocated...");
        Thread.sleep(1000);
    }

    /**
     * This test should be run manually
     *
     * @throws Exception
     */
    @Test
    @Ignore
    public void testDelayedAllocation1() throws  Exception {
        CudaEnvironment.getInstance().getConfiguration()
                .setFirstMemory(AllocationStatus.DEVICE)
                .setMemoryModel(Configuration.MemoryModel.DELAYED);

        AtomicAllocator allocator = AtomicAllocator.getInstance();

        INDArray array = Nd4j.create(new float[]{1f, 2f, 3f, 4f, 5f});

        AllocationPoint pointer = allocator.getAllocationPoint(array);

        PointersPair pair = pointer.getPointers();

        // pointers should be equal, device memory wasn't allocated yet
        assertEquals(pair.getDevicePointer(), pair.getHostPointer());

        //////////////

        AllocationPoint shapePointer = allocator.getAllocationPoint(array.shapeInfoDataBuffer());

        // pointers should be equal, device memory wasn't allocated yet
        assertEquals(shapePointer.getPointers().getDevicePointer(), shapePointer.getPointers().getHostPointer());

        assertEquals(pointer.getAllocationStatus(), AllocationStatus.HOST);
        assertEquals(shapePointer.getAllocationStatus(), AllocationStatus.HOST);

        float sum = array.sumNumber().floatValue();

        assertEquals(15.0f, sum, 0.0001f);

        assertEquals(AllocationStatus.CONSTANT, shapePointer.getAllocationStatus());
        assertEquals(AllocationStatus.DEVICE, pointer.getAllocationStatus());

        // at this point all pointers show be different, since we've used OP (sumNumber)
        assertNotEquals(shapePointer.getPointers().getDevicePointer(), shapePointer.getPointers().getHostPointer());
    }
}
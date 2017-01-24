package org.nd4j.jita.memory.impl;

import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.linalg.api.buffer.DataBuffer;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
@Ignore
public class CudaDirectProviderTest {

    @Test
    public void mallocHost() throws Exception {
        CudaDirectProvider provider = new CudaDirectProvider();

        AllocationShape shape = new AllocationShape(100000, 4, DataBuffer.Type.FLOAT);
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

        AllocationShape shape = new AllocationShape(300000, 4, DataBuffer.Type.FLOAT);
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

}
package org.nd4j.jita.memory.impl;

import lombok.extern.slf4j.Slf4j;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class CudaFullCachingProviderTest {
    @Before
    public void setUp() throws Exception {

    }


    @Test
    public void testPurge1() throws Exception {
        for (int x = 0; x < 10000; x++) {
            INDArray array = Nd4j.create(10,10);
        }

        INDArray array = Nd4j.create(10,10);
        AllocationShape shape = AtomicAllocator.getInstance().getAllocationPoint(array).getShape();

        System.gc();
        System.gc();
        Thread.sleep(1000);

        CudaFullCachingProvider provider = (CudaFullCachingProvider) AtomicAllocator.getInstance().getMemoryHandler().getMemoryProvider();


        log.info("Shapes in cache: {}", provider.deviceCache.get(0).get(shape).size());
        assertTrue(provider.deviceCache.get(0).get(shape).size() > 0);

        provider.purgeCache();

        assertTrue(provider.deviceCache.get(0).get(shape).size() == 0);
    }
}
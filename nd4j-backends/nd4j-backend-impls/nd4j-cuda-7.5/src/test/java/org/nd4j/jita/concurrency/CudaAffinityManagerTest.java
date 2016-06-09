package org.nd4j.jita.concurrency;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
@Ignore
public class CudaAffinityManagerTest {
    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void getDeviceForCurrentThread() throws Exception {
        CudaAffinityManager manager = new CudaAffinityManager();

        Integer deviceId = manager.getDeviceForCurrentThread();

        assertEquals(0, deviceId.intValue());

        manager.attachThreadToDevice(Thread.currentThread().getId(), 1);

        assertEquals(1, manager.getDeviceForCurrentThread().intValue());

        manager.attachThreadToDevice(Thread.currentThread().getId(), 0);

        assertEquals(0, manager.getDeviceForCurrentThread().intValue());
    }

    @Test
    public void getDeviceForAnotherThread() throws Exception {
        CudaAffinityManager manager = new CudaAffinityManager();

        Integer deviceId = manager.getDeviceForCurrentThread();

        assertEquals(0, deviceId.intValue());

        manager.attachThreadToDevice(17L, 0);

        assertEquals(0, manager.getDeviceForThread(17L).intValue());
    }

    @Test
    public void getDeviceForAnotherThread2() throws Exception {
        CudaAffinityManager manager = new CudaAffinityManager();

        Integer deviceId = manager.getDeviceForCurrentThread();

        assertEquals(0, deviceId.intValue());

        manager.attachThreadToDevice(17L, 0);

        assertEquals(0, manager.getDeviceForThread(17L).intValue());
    }

}
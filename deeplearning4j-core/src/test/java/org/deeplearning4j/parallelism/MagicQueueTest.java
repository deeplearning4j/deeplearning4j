package org.deeplearning4j.parallelism;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class MagicQueueTest {
    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void addDataSet1() throws Exception {
        MagicQueue queue = new MagicQueue.Builder().build();

        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();

        DataSet dataSet_1 = new DataSet(Nd4j.create(new float[]{1f,2f,3f}), Nd4j.create(new float[]{1f,2f,3f}));
        DataSet dataSet_2 = new DataSet(Nd4j.create(new float[]{1f,2f,3f}), Nd4j.create(new float[]{1f,2f,3f}));
        DataSet dataSet_3 = new DataSet(Nd4j.create(new float[]{1f,2f,3f}), Nd4j.create(new float[]{1f,2f,3f}));
        DataSet dataSet_4 = new DataSet(Nd4j.create(new float[]{1f,2f,3f}), Nd4j.create(new float[]{1f,2f,3f}));
        DataSet dataSet_5 = new DataSet(Nd4j.create(new float[]{1f,2f,3f}), Nd4j.create(new float[]{1f,2f,3f}));
        DataSet dataSet_6 = new DataSet(Nd4j.create(new float[]{1f,2f,3f}), Nd4j.create(new float[]{1f,2f,3f}));
        DataSet dataSet_7 = new DataSet(Nd4j.create(new float[]{1f,2f,3f}), Nd4j.create(new float[]{1f,2f,3f}));
        DataSet dataSet_8 = new DataSet(Nd4j.create(new float[]{1f,2f,3f}), Nd4j.create(new float[]{1f,2f,3f}));

        queue.add(dataSet_1);
        queue.add(dataSet_2);
        queue.add(dataSet_3);
        queue.add(dataSet_4);
        queue.add(dataSet_5);
        queue.add(dataSet_6);
        queue.add(dataSet_7);
        queue.add(dataSet_8);

        Thread.sleep(500);

        assertEquals(8 / numDevices, queue.size());
    }

    /**
     * THIS TEST REQUIRES CUDA BACKEND AND MULTI-GPU ENVIRONMENT
     * TO USE THIS TEST - ENABLE ND4J-CUDA BACKEND FOR THIS MODULE
     *
     * In this test we check actual data relocation within MagicQueue
     *
     * @throws Exception
     */
    @Test
    public void addDataSet2() throws Exception {
        CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true).allowCrossDeviceAccess(false);

        MagicQueue queue = new MagicQueue.Builder().build();

        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();

        DataSet dataSet_1 = new DataSet(Nd4j.create(new float[]{1f,2f,3f}), Nd4j.create(new float[]{1f,2f,3f}));
        DataSet dataSet_2 = new DataSet(Nd4j.create(new float[]{1f,2f,3f}), Nd4j.create(new float[]{1f,2f,3f}));
        DataSet dataSet_3 = new DataSet(Nd4j.create(new float[]{1f,2f,3f}), Nd4j.create(new float[]{1f,2f,3f}));
        DataSet dataSet_4 = new DataSet(Nd4j.create(new float[]{1f,2f,3f}), Nd4j.create(new float[]{1f,2f,3f}));
        DataSet dataSet_5 = new DataSet(Nd4j.create(new float[]{1f,2f,3f}), Nd4j.create(new float[]{1f,2f,3f}));
        DataSet dataSet_6 = new DataSet(Nd4j.create(new float[]{1f,2f,3f}), Nd4j.create(new float[]{1f,2f,3f}));
        DataSet dataSet_7 = new DataSet(Nd4j.create(new float[]{1f,2f,3f}), Nd4j.create(new float[]{1f,2f,3f}));
        DataSet dataSet_8 = new DataSet(Nd4j.create(new float[]{1f,2f,3f}), Nd4j.create(new float[]{1f,2f,3f}));


        AllocationPoint point_1 = AtomicAllocator.getInstance().getAllocationPoint(dataSet_1.getFeatures());
        AllocationPoint point_2 = AtomicAllocator.getInstance().getAllocationPoint(dataSet_2.getFeatures());
        AllocationPoint point_3 = AtomicAllocator.getInstance().getAllocationPoint(dataSet_3.getFeatures());
        AllocationPoint point_4 = AtomicAllocator.getInstance().getAllocationPoint(dataSet_4.getFeatures());

        // All arrays are located on same initial device
        assertEquals(0, point_1.getDeviceId().intValue());
        assertEquals(0, point_2.getDeviceId().intValue());
        assertEquals(0, point_3.getDeviceId().intValue());
        assertEquals(0, point_4.getDeviceId().intValue());

        queue.add(dataSet_1);
        queue.add(dataSet_2);
        queue.add(dataSet_3);
        queue.add(dataSet_4);
        queue.add(dataSet_5);
        queue.add(dataSet_6);
        queue.add(dataSet_7);
        queue.add(dataSet_8);

        Thread.sleep(500);

        assertEquals(8 / numDevices, queue.size());

        // All arrays are spread over all available devices
        assertEquals(0, point_1.getDeviceId().intValue());
        assertEquals(1, point_2.getDeviceId().intValue());
        assertEquals(0, point_3.getDeviceId().intValue());
        assertEquals(1, point_4.getDeviceId().intValue());
    }

}

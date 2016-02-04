package org.nd4j.jita.allocator.impl;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.enums.SyncState;
import org.nd4j.jita.allocator.utils.AllocationUtils;
import org.nd4j.jita.balance.impl.FirstInBalancer;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.jita.conf.DeviceInformation;
import org.nd4j.jita.mover.DummyMover;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.*;

/**
 * This test set is part of tests for JITA, please assume test-driven development for ANY changes applied to JITA.
 *
 *  This particular set addresses basic mechanics,  working to atomic level and not touching any real allocations or data copies.
 *
 *  We assume that abstract Long id value represents buffer id, and shape describes buffer shape.
 *
 * @author raver119@gmail.com
 */
public class BasicAllocatorTest {

    private CudaEnvironment singleDevice4GBcc52;
    private CudaEnvironment doubleDevices4GBcc52;
    private CudaEnvironment fourDevices4GBcc52;

    private static Logger log = LoggerFactory.getLogger(BasicAllocatorTest.class);

    @Before
    public void setUp() throws Exception {
        singleDevice4GBcc52 = new CudaEnvironment();
        doubleDevices4GBcc52 = new CudaEnvironment();
        fourDevices4GBcc52 = new CudaEnvironment();

        DeviceInformation device1 = new DeviceInformation();
        device1.setDeviceId(1);
        device1.setCcMajor(5);
        device1.setCcMinor(2);
        device1.setTotalMemory(4 * 1024 * 1024 * 1024L);
        device1.setAvailableMemory(4 * 1024 * 1024 * 1024L);

        singleDevice4GBcc52.addDevice(device1);


        DeviceInformation device21 = new DeviceInformation();
        device21.setDeviceId(1);
        device21.setCcMajor(5);
        device21.setCcMinor(2);
        device21.setTotalMemory(4 * 1024 * 1024 * 1024L);
        device21.setAvailableMemory(4 * 1024 * 1024 * 1024L);

        DeviceInformation device22 = new DeviceInformation();
        device22.setDeviceId(1);
        device22.setCcMajor(5);
        device22.setCcMinor(2);
        device22.setTotalMemory(4 * 1024 * 1024 * 1024L);
        device22.setAvailableMemory(4 * 1024 * 1024 * 1024L);

        doubleDevices4GBcc52.addDevice(device21);
        doubleDevices4GBcc52.addDevice(device22);


        DeviceInformation device41 = new DeviceInformation();
        device41.setDeviceId(1);
        device41.setCcMajor(5);
        device41.setCcMinor(2);
        device41.setTotalMemory(4 * 1024 * 1024 * 1024L);
        device41.setAvailableMemory(4 * 1024 * 1024 * 1024L);

        DeviceInformation device42 = new DeviceInformation();
        device42.setDeviceId(1);
        device42.setCcMajor(5);
        device42.setCcMinor(2);
        device42.setTotalMemory(4 * 1024 * 1024 * 1024L);
        device42.setAvailableMemory(4 * 1024 * 1024 * 1024L);

        DeviceInformation device43 = new DeviceInformation();
        device43.setDeviceId(1);
        device43.setCcMajor(5);
        device43.setCcMinor(2);
        device43.setTotalMemory(4 * 1024 * 1024 * 1024L);
        device43.setAvailableMemory(4 * 1024 * 1024 * 1024L);

        DeviceInformation device44 = new DeviceInformation();
        device44.setDeviceId(1);
        device44.setCcMajor(5);
        device44.setCcMinor(2);
        device44.setTotalMemory(4 * 1024 * 1024 * 1024L);
        device44.setAvailableMemory(4 * 1024 * 1024 * 1024L);

        fourDevices4GBcc52.addDevice(device41);
        fourDevices4GBcc52.addDevice(device42);
        fourDevices4GBcc52.addDevice(device43);
        fourDevices4GBcc52.addDevice(device44);
    }

    @After
    public void tearDown() throws Exception {

    }

    /**
     * We want to be sure that instances are not sharing,
     * That's important for testing environment.
     *
     * @throws Exception
     */
    @Test
    public void testGetInstance() throws Exception {
        BasicAllocator instance = BasicAllocator.getInstance();

        BasicAllocator instance2 = new BasicAllocator();

        assertNotEquals(null, instance);
        assertNotEquals(null, instance2);
        assertNotEquals(instance, instance2);
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////                SINGLE ALLOCATION ON 1D ARRAY TESTS
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * We inject some abstract object into allocator, and check how it get's moved from host to device
     *
     */
    @Test
    public void testHostToDeviceMovement() throws Exception {
        BasicAllocator allocator = new BasicAllocator();
        allocator.setEnvironment(singleDevice4GBcc52);
        allocator.setMover(new DummyMover());
        /*
            Lets assume we have some INDArray backed by some DataBuffer, internally identified by Long id
        */
        Long objectId = 17L;

        AllocationShape shape = new AllocationShape();
        shape.setDataType(DataBuffer.Type.FLOAT);
        shape.setLength(100);
        shape.setOffset(0);
        shape.setStride(1);

        allocator.registerSpan(objectId, shape);

        /*
            We check that allocation point was created for our object
        */
        AllocationPoint point = allocator.getAllocationPoint(objectId);
        assertNotEquals(null, point);
        assertNotEquals(0, point.getAccessHost());
        assertEquals(0, point.getAccessDevice());
        assertEquals(AllocationStatus.UNDEFINED, point.getAllocationStatus());

        point.setAccessHost(0);
        point.setAccessDevice(0);

        // we emulate update call on host side, and subsequent update on device side
        allocator.tickHost(objectId);
        assertEquals(AllocationStatus.UNDEFINED, point.getAllocationStatus());

        // now we emulate use of this object on the device
        allocator.getDevicePointer(objectId);

        // tickDevice should be private and iternal call
        //allocator.tickDevice(objectId, 1);

        assertEquals(1, point.getDeviceTicks());
        assertTrue(point.getAccessDevice() > point.getAccessHost());
        assertEquals(AllocationStatus.ZERO, point.getAllocationStatus());

        // At this point we assume that memory chunk is accessed via zero-copy pinned memory strategy
        // and now we'll emulate data move from zero-copy memory to device memory
        allocator.relocateMemory(objectId, AllocationStatus.DEVICE);

        assertEquals(AllocationStatus.DEVICE, point.getAllocationStatus());

        assertEquals(1, point.getNumberOfDescendants());
        // we will not deallocate object in this test
    }

    /**
     * Checking how data gets moved from device to host
     *
     * @throws Exception
     */
    @Test
    public void testDeviceToHostMovement() throws Exception {
        BasicAllocator allocator = new BasicAllocator();
        allocator.setMover(new DummyMover());

        Long objectId = 19L;

        AllocationShape shape = new AllocationShape();
        shape.setDataType(DataBuffer.Type.FLOAT);
        shape.setLength(100);
        shape.setOffset(0);
        shape.setStride(1);

        allocator.registerSpan(objectId, shape);

        /*
            We check that allocation point was created for our object
        */
        AllocationPoint point = allocator.getAllocationPoint(objectId);
        assertNotEquals(null, point);
        assertNotEquals(0, point.getAccessHost());
        assertEquals(0, point.getAccessDevice());
        assertEquals(AllocationStatus.UNDEFINED, point.getAllocationStatus());

        // we emulate that memory is allocated on device and was used there
        allocator.getDevicePointer(objectId);

        allocator.synchronizeHostData(objectId);
        assertNotEquals(0, point.getAccessHost());
        assertEquals(point.getAccessDevice(), point.getAccessHost());
        assertEquals(SyncState.SYNC, point.getHostMemoryState());

        assertEquals(1, point.getNumberOfDescendants());
    }

    @Test
    public void testSingleDeallocation() throws Exception {
        BasicAllocator allocator = new BasicAllocator();
        allocator.setMover(new DummyMover());

        Long objectId = 19L;

        AllocationShape shape = new AllocationShape();
        shape.setDataType(DataBuffer.Type.FLOAT);
        shape.setLength(100);
        shape.setOffset(0);
        shape.setStride(1);

        allocator.registerSpan(objectId, shape);

        /*
            We check that allocation point was created for our object
        */
        AllocationPoint point = allocator.getAllocationPoint(objectId);
        assertNotEquals(null, point);
        assertNotEquals(0, point.getAccessHost());
        assertEquals(0, point.getAccessDevice());
        assertEquals(AllocationStatus.UNDEFINED, point.getAllocationStatus());

        // we emulate that memory is allocated on device and was used there
        allocator.getDevicePointer(objectId);

        assertEquals(AllocationStatus.ZERO, point.getAllocationStatus());

        // now we call for release
        allocator.releaseMemory(objectId, point.getShape());

        assertEquals(AllocationStatus.DEALLOCATED, point.getAllocationStatus());
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////                SINGLE ALLOCATION ON VIEWS TESTS
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * This test addresses allocation of the part of existing memory chunk
     * @throws Exception
     */
    @Test
    public void testSuballocationSecondaryFirst() throws Exception {
        BasicAllocator allocator = new BasicAllocator();
        allocator.setMover(new DummyMover());

        Long objectId = 19L;

        AllocationShape shape = new AllocationShape();
        shape.setDataType(DataBuffer.Type.FLOAT);
        shape.setLength(100);
        shape.setOffset(0);
        shape.setStride(1);

        allocator.registerSpan(objectId, shape);


        AllocationShape shape2 = new AllocationShape();
        shape2.setDataType(DataBuffer.Type.FLOAT);
        shape2.setLength(50);
        shape2.setOffset(10);
        shape2.setStride(1);

        allocator.registerSpan(objectId, shape2);

        assertEquals(1, allocator.tableSize());

        AllocationPoint point = allocator.getAllocationPoint(objectId);

        assertEquals(shape, point.getShape());
        assertNotEquals(shape2, point.getShape());

        /*
         now we should check that both shapes were registered within allocation point
        */
        assertEquals(2, point.getNumberOfDescendants());

        assertEquals(0, point.getDescendantTicks(shape2));
        assertEquals(0, point.getDescendantTicks(shape));

        allocator.getDevicePointer(objectId, shape2);

        assertEquals(1, point.getDescendantTicks(shape2));
        assertEquals(0, point.getDescendantTicks(shape));

        allocator.getDevicePointer(objectId, shape2);

        assertEquals(2, point.getDescendantTicks(shape2));
        assertEquals(0, point.getDescendantTicks(shape));
    }

    /**
     * This test addresses allocation of subspace, but in this case memory is already allocated for top-level chunk
     * @throws Exception
     */
    @Test
    public void tesSuballocationPrimaryFirst() throws Exception {
        BasicAllocator allocator = new BasicAllocator();
        allocator.setMover(new DummyMover());

        Long objectId = 19L;

        AllocationShape shape = new AllocationShape();
        shape.setDataType(DataBuffer.Type.FLOAT);
        shape.setLength(100);
        shape.setOffset(0);
        shape.setStride(1);

        allocator.registerSpan(objectId, shape);

        AllocationPoint point = allocator.getAllocationPoint(objectId);

        // we request pointer for original shape
        allocator.getDevicePointer(objectId);

        assertEquals(1, point.getDescendantTicks(shape));

        AllocationShape shape2 = new AllocationShape();
        shape2.setDataType(DataBuffer.Type.FLOAT);
        shape2.setLength(50);
        shape2.setOffset(10);
        shape2.setStride(1);

        allocator.registerSpan(objectId, shape2);

        // we request pointer for subarray
        allocator.getDevicePointer(objectId, shape2);

        assertEquals(2, point.getNumberOfDescendants());

        assertEquals(1, point.getDescendantTicks(shape));
        assertEquals(1, point.getDescendantTicks(shape2));
    }

    @Test
    public void testNestedDeallocationPrimaryFirst1() throws Exception {
        BasicAllocator allocator = new BasicAllocator();
        allocator.setMover(new DummyMover());

        Long objectId = 19L;

        AllocationShape shape = new AllocationShape();
        shape.setDataType(DataBuffer.Type.FLOAT);
        shape.setLength(100);
        shape.setOffset(0);
        shape.setStride(1);

        allocator.registerSpan(objectId, shape);

        AllocationPoint point = allocator.getAllocationPoint(objectId);

        // we request pointer for original shape
        allocator.getDevicePointer(objectId);

        assertEquals(1, point.getDescendantTicks(shape));

        AllocationShape shape2 = new AllocationShape();
        shape2.setDataType(DataBuffer.Type.FLOAT);
        shape2.setLength(50);
        shape2.setOffset(10);
        shape2.setStride(1);

        allocator.registerSpan(objectId, shape2);

        // we request pointer for subarray
        allocator.getDevicePointer(objectId, shape2);

        assertEquals(2, point.getNumberOfDescendants());

        assertEquals(1, point.getDescendantTicks(shape));
        assertEquals(1, point.getDescendantTicks(shape2));

        /*
         now we call dealloc for primary array.

         it should be skipped, because we still have nested allocation,
        */

        allocator.releaseMemory(objectId, shape);


        assertEquals(AllocationStatus.ZERO, point.getAllocationStatus());

    }

    @Test
    public void testNestedDeallocationSecondaryFirst1() throws Exception {
        BasicAllocator allocator = new BasicAllocator();
        allocator.setMover(new DummyMover());

        Long objectId = 19L;

        AllocationShape shape = new AllocationShape();
        shape.setDataType(DataBuffer.Type.FLOAT);
        shape.setLength(100);
        shape.setOffset(0);
        shape.setStride(1);

        allocator.registerSpan(objectId, shape);

        AllocationPoint point = allocator.getAllocationPoint(objectId);

        // we request pointer for original shape
        allocator.getDevicePointer(objectId);

        assertEquals(1, point.getDescendantTicks(shape));

        AllocationShape shape2 = new AllocationShape();
        shape2.setDataType(DataBuffer.Type.FLOAT);
        shape2.setLength(50);
        shape2.setOffset(10);
        shape2.setStride(1);

        allocator.registerSpan(objectId, shape2);

        // we request pointer for subarray
        allocator.getDevicePointer(objectId, shape2);

        assertEquals(2, point.getNumberOfDescendants());

        assertEquals(1, point.getDescendantTicks(shape));
        assertEquals(1, point.getDescendantTicks(shape2));

        /*
         now we call dealloc for secondary shape.

         it should remove that shape from suballocations list
        */

        allocator.releaseMemory(objectId, shape2);

        assertEquals(AllocationStatus.ZERO, point.getAllocationStatus());

        assertEquals(1, point.getNumberOfDescendants());
        assertEquals(1, point.getDescendantTicks(shape));
        assertEquals(-1, point.getDescendantTicks(shape2));
    }

    @Test
    public void testNestedDeallocationSecondaryThenPrimary1() throws Exception {
        BasicAllocator allocator = new BasicAllocator();
        allocator.setMover(new DummyMover());

        Long objectId = 19L;

        AllocationShape shape = new AllocationShape();
        shape.setDataType(DataBuffer.Type.FLOAT);
        shape.setLength(100);
        shape.setOffset(0);
        shape.setStride(1);

        allocator.registerSpan(objectId, shape);

        AllocationPoint point = allocator.getAllocationPoint(objectId);

        // we request pointer for original shape
        allocator.getDevicePointer(objectId);

        assertEquals(1, point.getDescendantTicks(shape));

        AllocationShape shape2 = new AllocationShape();
        shape2.setDataType(DataBuffer.Type.FLOAT);
        shape2.setLength(50);
        shape2.setOffset(10);
        shape2.setStride(1);

        allocator.registerSpan(objectId, shape2);

        // we request pointer for subarray
        allocator.getDevicePointer(objectId, shape2);

        assertEquals(2, point.getNumberOfDescendants());

        assertEquals(1, point.getDescendantTicks(shape));
        assertEquals(1, point.getDescendantTicks(shape2));

        /*
         now we call dealloc for secondary shape.

         it should remove that shape from suballocations list
        */

        allocator.releaseMemory(objectId, shape2);

        /*
            Now, since we have no nested allocations - initial allocation can be released
         */
        allocator.releaseMemory(objectId, shape);

        assertEquals(AllocationStatus.DEALLOCATED, point.getAllocationStatus());
        assertEquals(null, allocator.getAllocationPoint(objectId));
    }

    /**
     * This test addresses nested memory relocation from ZeroCopy memory to device memory
     *
     * @throws Exception
     */
    @Test
    public void testNestedRelocationToDevice1() throws Exception {
        BasicAllocator allocator = new BasicAllocator();
        allocator.setEnvironment(singleDevice4GBcc52);
        allocator.setMover(new DummyMover());

        Long objectId = 19L;

        AllocationShape shape = new AllocationShape();
        shape.setDataType(DataBuffer.Type.FLOAT);
        shape.setLength(100);
        shape.setOffset(0);
        shape.setStride(1);

        allocator.registerSpan(objectId, shape);

        AllocationPoint point = allocator.getAllocationPoint(objectId);

        // we request pointer for original shape
        Object dPtr = allocator.getDevicePointer(objectId);

        assertEquals(1, point.getDescendantTicks(shape));

        AllocationShape shape2 = new AllocationShape();
        shape2.setDataType(DataBuffer.Type.FLOAT);
        shape2.setLength(50);
        shape2.setOffset(10);
        shape2.setStride(1);

        allocator.registerSpan(objectId, shape2);

        assertEquals(AllocationStatus.ZERO, point.getAllocationStatus());

        allocator.relocateMemory(objectId, AllocationStatus.DEVICE);

        assertNotEquals(dPtr, point.getDevicePointer());
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////                DECISION MAKING ON ALLOCATION TESTS, SINGLE DEVICE
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     *
     * This test should agree on relocation
     *
     * @throws Exception
     */
    @Test
    public void testSinglePromoteDecision1() throws Exception {
        BasicAllocator allocator = new BasicAllocator();
        allocator.setEnvironment(singleDevice4GBcc52);
        allocator.setBalancer(new FirstInBalancer());
        allocator.setMover(new DummyMover());

        Long objectId = 19L;

        AllocationShape shape = new AllocationShape();
        shape.setDataType(DataBuffer.Type.FLOAT);
        shape.setLength(100);
        shape.setOffset(0);
        shape.setStride(1);

        allocator.registerSpan(objectId, shape);

        AllocationPoint point = allocator.getAllocationPoint(objectId);

        // we request pointer for original shape
        Object dPtr = allocator.getDevicePointer(objectId);
        Object dPtr2 = allocator.getDevicePointer(objectId);

        // data wasn't changed inbetween calls, and it's still in ZERO space
        assertEquals(dPtr, dPtr2);
        assertEquals(AllocationStatus.ZERO, point.getAllocationStatus());

        // data could be moved to device
        assertEquals(AllocationStatus.DEVICE, allocator.makePromoteDecision(objectId));
    }

    /**
     *
     * This tests checks allocation declined if device memory is not enough
     *
     * @throws Exception
     */
    @Test
    public void testSinglePromoteDecision2() throws Exception {
        BasicAllocator allocator = new BasicAllocator();
        allocator.setEnvironment(singleDevice4GBcc52);
        allocator.setBalancer(new FirstInBalancer());
        allocator.setMover(new DummyMover());


        Long objectId = 22L;

        AllocationShape shape = new AllocationShape();
        shape.setDataType(DataBuffer.Type.FLOAT);
        shape.setLength(10 * 1024 * 1024 * 1024L);
        shape.setOffset(0);
        shape.setStride(1);

        allocator.registerSpan(objectId, shape);

        AllocationPoint point = allocator.getAllocationPoint(objectId);

        // we request pointer for original shape
        Object dPtr = allocator.getDevicePointer(objectId);
        Object dPtr2 = allocator.getDevicePointer(objectId);

        // data wasn't changed inbetween calls, and it's still in ZERO space
        assertEquals(dPtr, dPtr2);
        assertEquals(AllocationStatus.ZERO, point.getAllocationStatus());

        // data CAN'T be moved to device, since there's not enough device memory to hold 10GB
        assertEquals(AllocationStatus.ZERO, allocator.makePromoteDecision(objectId));
    }

    /**
     *
     * This test covers situation, when we have multiple promotion candidates
     * First will gets allocated, second will be declined
     *
     * @throws Exception
     */
    @Test
    public void testSinglePromoteDecision3() throws Exception {
        BasicAllocator allocator = new BasicAllocator();
        allocator.setEnvironment(singleDevice4GBcc52);
        allocator.setBalancer(new FirstInBalancer());
        allocator.setMover(new DummyMover());

        assertEquals(0, singleDevice4GBcc52.getAllocatedMemoryForDevice(1));

        Long objectId1 = 22L;

        AllocationShape shape1 = new AllocationShape();
        shape1.setDataType(DataBuffer.Type.FLOAT);
        shape1.setLength(63 * 1024 * 1024L);
        shape1.setOffset(0);
        shape1.setStride(1);

        allocator.registerSpan(objectId1, shape1);

        assertEquals(0, singleDevice4GBcc52.getAllocatedMemoryForDevice(1));

        Long objectId2 = 44L;

        AllocationShape shape2 = new AllocationShape();
        shape2.setDataType(DataBuffer.Type.FLOAT);
        shape2.setLength(10 * 1024 * 1024L);
        shape2.setOffset(0);
        shape2.setStride(1);

        allocator.registerSpan(objectId2, shape2);

        assertEquals(0, singleDevice4GBcc52.getAllocatedMemoryForDevice(1));

        /*
            At this point we have two memory chunks, that can be moved to device.
            So, we emulate few accesses to them, and checking for decision
        */

        allocator.getDevicePointer(objectId1);

        assertEquals(0, singleDevice4GBcc52.getAllocatedMemoryForDevice(1));

        allocator.getDevicePointer(objectId2);
        allocator.getDevicePointer(objectId2);
        allocator.getDevicePointer(objectId2);

        assertEquals(0, singleDevice4GBcc52.getAllocatedMemoryForDevice(1));

        AllocationStatus target = allocator.makePromoteDecision(objectId1);
        assertEquals(AllocationStatus.DEVICE, target);
        allocator.relocateMemory(objectId1, target);

        assertEquals(AllocationStatus.ZERO, allocator.makePromoteDecision(objectId2));

        long allocatedMemory = singleDevice4GBcc52.getAllocatedMemoryForDevice(1);
        assertEquals(AllocationUtils.getRequiredMemory(shape1), allocatedMemory);
    }

    @Test
    public void testSingleDemoteDecision1() throws Exception {
        BasicAllocator allocator = new BasicAllocator();
        allocator.setEnvironment(singleDevice4GBcc52);
        allocator.setBalancer(new FirstInBalancer());
        allocator.setMover(new DummyMover());

        assertEquals(0, singleDevice4GBcc52.getAllocatedMemoryForDevice(1));

        Long objectId1 = 22L;

        AllocationShape shape1 = new AllocationShape();
        shape1.setDataType(DataBuffer.Type.FLOAT);
        shape1.setLength(63 * 1024 * 1024L);
        shape1.setOffset(0);
        shape1.setStride(1);

        allocator.registerSpan(objectId1, shape1);

        AllocationPoint point = allocator.getAllocationPoint(objectId1);
        allocator.getDevicePointer(objectId1);
        allocator.relocateMemory(objectId1, AllocationStatus.DEVICE);

        assertEquals(AllocationStatus.DEVICE, point.getAllocationStatus());

        long allocatedMemory = singleDevice4GBcc52.getAllocatedMemoryForDevice(1);
        assertEquals(AllocationUtils.getRequiredMemory(shape1), allocatedMemory);

        AllocationStatus targetStatus = allocator.makeDemoteDecision(objectId1);
        assertEquals(AllocationStatus.ZERO, targetStatus);
    }

    @Test
    public void testSingleDemoteDecision2() throws Exception {
        BasicAllocator allocator = new BasicAllocator();
        allocator.setEnvironment(singleDevice4GBcc52);
        allocator.setBalancer(new FirstInBalancer());
        allocator.setMover(new DummyMover());

        assertEquals(0, singleDevice4GBcc52.getAllocatedMemoryForDevice(1));

        Long objectId1 = 22L;

        AllocationShape shape1 = new AllocationShape();
        shape1.setDataType(DataBuffer.Type.FLOAT);
        shape1.setLength(1 * 1024 * 1024L);
        shape1.setOffset(0);
        shape1.setStride(1);

        allocator.registerSpan(objectId1, shape1);

        AllocationPoint point = allocator.getAllocationPoint(objectId1);
        allocator.getDevicePointer(objectId1);
        allocator.relocateMemory(objectId1, AllocationStatus.DEVICE);

        assertEquals(AllocationStatus.DEVICE, point.getAllocationStatus());

        long allocatedMemory = singleDevice4GBcc52.getAllocatedMemoryForDevice(1);
        assertEquals(AllocationUtils.getRequiredMemory(shape1), allocatedMemory);

        AllocationStatus targetStatus = allocator.makeDemoteDecision(objectId1);
        assertEquals(AllocationStatus.DEVICE, targetStatus);
    }
}
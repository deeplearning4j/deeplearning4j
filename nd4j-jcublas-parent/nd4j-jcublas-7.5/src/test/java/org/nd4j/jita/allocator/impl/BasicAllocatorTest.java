package org.nd4j.jita.allocator.impl;

import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.jita.allocator.enums.Aggressiveness;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.enums.SyncState;
import org.nd4j.jita.allocator.time.impl.SimpleTimer;
import org.nd4j.jita.allocator.utils.AllocationUtils;
import org.nd4j.jita.balance.impl.FirstInBalancer;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.jita.conf.DeviceInformation;
import org.nd4j.jita.mover.DummyMover;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

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
        device21.setDeviceId(0);
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
        device41.setDeviceId(0);
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
        device43.setDeviceId(2);
        device43.setCcMajor(5);
        device43.setCcMinor(2);
        device43.setTotalMemory(4 * 1024 * 1024 * 1024L);
        device43.setAvailableMemory(4 * 1024 * 1024 * 1024L);

        DeviceInformation device44 = new DeviceInformation();
        device44.setDeviceId(3);
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
        //allocator.tickDevice(objectId, shape);
        allocator.tackDevice(objectId, shape);

        assertEquals(1, point.getDeviceTicks());
        assertTrue(point.getAccessDevice() > point.getAccessHost());
        assertEquals(AllocationStatus.ZERO, point.getAllocationStatus());

        // At this point we assume that memory chunk is accessed via zero-copy pinned memory strategy
        // and now we'll emulate data move from zero-copy memory to device memory
        allocator.relocateMemory(objectId, AllocationStatus.DEVICE);

        assertEquals(AllocationStatus.DEVICE, point.getAllocationStatus());

        assertEquals(0, point.getNumberOfDescendants());
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
        allocator.setEnvironment(singleDevice4GBcc52);
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
        allocator.tickDevice(objectId, shape);
        allocator.tackDevice(objectId, shape);

        allocator.synchronizeHostData(objectId);



        assertNotEquals(0, point.getAccessHost());
        assertEquals(point.getAccessDevice(), point.getAccessHost());
        assertEquals(SyncState.SYNC, point.getHostMemoryState());

        assertEquals(0, point.getNumberOfDescendants());
    }

    @Test
    public void testSingleDeallocation1() throws Exception {
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
        allocator.tackDevice(objectId, point.getShape());

        assertEquals(AllocationStatus.ZERO, point.getAllocationStatus());

        // now we call for release
        allocator.releaseMemory(objectId, point.getShape());

        assertEquals(AllocationStatus.HOST, point.getAllocationStatus());
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
        allocator.setEnvironment(singleDevice4GBcc52);
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
        assertEquals(1, point.getNumberOfDescendants());

        assertEquals(0, point.getDescendantTicks(shape2));

        allocator.getDevicePointer(objectId, shape2);

        assertEquals(1, point.getDescendantTicks(shape2));

        allocator.getDevicePointer(objectId, shape2);

        assertEquals(2, point.getDescendantTicks(shape2));
    }

    /**
     * This test addresses allocation of subspace, but in this case memory is already allocated for top-level chunk
     * @throws Exception
     */
    @Test
    public void tesSuballocationPrimaryFirst() throws Exception {
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
        allocator.getDevicePointer(objectId);

        assertEquals(-1, point.getDescendantTicks(shape));

        AllocationShape shape2 = new AllocationShape();
        shape2.setDataType(DataBuffer.Type.FLOAT);
        shape2.setLength(50);
        shape2.setOffset(10);
        shape2.setStride(1);

        allocator.registerSpan(objectId, shape2);

        // we request pointer for subarray
        allocator.getDevicePointer(objectId, shape2);

        assertEquals(1, point.getNumberOfDescendants());

        assertEquals(-1, point.getDescendantTicks(shape));
        assertEquals(1, point.getDescendantTicks(shape2));
    }

    @Test
    public void testNestedDeallocationPrimaryFirst1() throws Exception {
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
        allocator.getDevicePointer(objectId);
        allocator.tackDevice(objectId, point.getShape());

        assertEquals(-1, point.getDescendantTicks(shape));

        AllocationShape shape2 = new AllocationShape();
        shape2.setDataType(DataBuffer.Type.FLOAT);
        shape2.setLength(50);
        shape2.setOffset(10);
        shape2.setStride(1);

        allocator.registerSpan(objectId, shape2);

        // we request pointer for subarray
        allocator.getDevicePointer(objectId, shape2);
        allocator.tackDevice(objectId, shape2);
        assertEquals(1, point.getNumberOfDescendants());

        assertEquals(-1, point.getDescendantTicks(shape));
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
        allocator.getDevicePointer(objectId);
        allocator.tackDevice(objectId, point.getShape());

        assertEquals(-1, point.getDescendantTicks(shape));

        AllocationShape shape2 = new AllocationShape();
        shape2.setDataType(DataBuffer.Type.FLOAT);
        shape2.setLength(50);
        shape2.setOffset(10);
        shape2.setStride(1);

        allocator.registerSpan(objectId, shape2);

        // we request pointer for subarray
        allocator.getDevicePointer(objectId, shape2);


        assertEquals(1, point.getNumberOfDescendants());

        assertEquals(-1, point.getDescendantTicks(shape));
        assertEquals(1, point.getDescendantTicks(shape2));

        /*
         now we call dealloc for secondary shape.

         it should remove that shape from suballocations list
        */

        allocator.tackDevice(objectId, shape2);
        allocator.releaseMemory(objectId, shape2);

        assertEquals(AllocationStatus.ZERO, point.getAllocationStatus());

        assertEquals(0, point.getNumberOfDescendants());
        assertEquals(-1, point.getDescendantTicks(shape));
        assertEquals(-1, point.getDescendantTicks(shape2));
    }

    @Test
    public void testNestedDeallocationSecondaryThenPrimary1() throws Exception {
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
        allocator.getDevicePointer(objectId);
        allocator.tackDevice(objectId, point.getShape());

        assertEquals(-1, point.getDescendantTicks(shape));

        AllocationShape shape2 = new AllocationShape();
        shape2.setDataType(DataBuffer.Type.FLOAT);
        shape2.setLength(50);
        shape2.setOffset(10);
        shape2.setStride(1);

        allocator.registerSpan(objectId, shape2);

        // we request pointer for subarray
        allocator.getDevicePointer(objectId, shape2);
        allocator.tackDevice(objectId, shape2);

        assertEquals(1, point.getNumberOfDescendants());

        assertEquals(-1, point.getDescendantTicks(shape));
        assertEquals(1, point.getDescendantTicks(shape2));


        // we request pointer for subarray one more time
        allocator.getDevicePointer(objectId, shape2);
        allocator.tackDevice(objectId, shape2);
        assertEquals(1, point.getNumberOfDescendants());

        /*
         now we call dealloc for secondary shape.

         it should remove that shape from suballocations list
        */

        allocator.releaseMemory(objectId, shape2);

        /*
            Now, since we have no nested allocations - initial allocation can be released
         */
        allocator.releaseMemory(objectId, shape);

        assertEquals(AllocationStatus.HOST, point.getAllocationStatus());
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

        AllocationShape shape2 = new AllocationShape();
        shape2.setDataType(DataBuffer.Type.FLOAT);
        shape2.setLength(50);
        shape2.setOffset(10);
        shape2.setStride(1);

        allocator.registerSpan(objectId, shape2);

        assertEquals(AllocationStatus.ZERO, point.getAllocationStatus());

        allocator.relocateMemory(objectId, AllocationStatus.DEVICE);

        assertNotEquals(dPtr, point.getCudaPointer());

        assertEquals(AllocationStatus.DEVICE, point.getAllocationStatus());
        assertEquals(1, allocator.getDeviceAllocations().size());
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
        assertEquals(AllocationStatus.DEVICE, allocator.makePromoteDecision(objectId, shape));
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
        assertEquals(AllocationStatus.ZERO, allocator.makePromoteDecision(objectId, shape));
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

        AllocationStatus target = allocator.makePromoteDecision(objectId1, shape1);
        assertEquals(AllocationStatus.DEVICE, target);
        allocator.relocateMemory(objectId1, target);

        assertEquals(AllocationStatus.ZERO, allocator.makePromoteDecision(objectId2, shape2));

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
        allocator.tackDevice(objectId1, shape1);
        allocator.relocateMemory(objectId1, AllocationStatus.DEVICE);

        assertEquals(AllocationStatus.DEVICE, point.getAllocationStatus());

        long allocatedMemory = singleDevice4GBcc52.getAllocatedMemoryForDevice(1);
        assertEquals(AllocationUtils.getRequiredMemory(shape1), allocatedMemory);

        AllocationStatus targetStatus = allocator.makeDemoteDecision(objectId1, shape1);

        log.info("Ticks: " + point.getDescendantsTicks() + " Tacks: " + point.getDescendantsTacks());

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

        AllocationStatus targetStatus = allocator.makeDemoteDecision(objectId1, shape1);
        assertEquals(AllocationStatus.DEVICE, targetStatus);
    }

    /**
     * This test should decline demote, since:
     *  1.  Memory is in use
     *  2.  Nested memory can't be demoted
     * @throws Exception
     */
    @Test
    public void testNestedDemoteDecision1() throws Exception {
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

        AllocationShape shape2 = new AllocationShape();
        shape2.setDataType(DataBuffer.Type.FLOAT);
        shape2.setLength(512 * 1024L);
        shape2.setOffset(100);
        shape2.setStride(1);

        AllocationPoint point = allocator.getAllocationPoint(objectId1);
        Object pointer1 = allocator.getDevicePointer(objectId1);
        allocator.relocateMemory(objectId1, AllocationStatus.DEVICE);

        // We are checking, that old pointer does not equal to new one, since memory was relocated
        assertNotEquals(pointer1, allocator.getAllocationPoint(objectId1));

        pointer1 = allocator.getDevicePointer(objectId1);

        allocator.registerSpan(objectId1, shape2);

        Object pointer2 = allocator.getDevicePointer(objectId1, shape2);
        assertNotEquals(pointer1, pointer2);

        assertNotEquals(pointer2, point.getCudaPointer());
        assertEquals(pointer1, point.getCudaPointer());
        assertEquals(AllocationStatus.DEVICE, point.getAllocationStatus());

        AllocationStatus decision1 = allocator.makeDemoteDecision(objectId1, shape1);
        assertEquals(AllocationStatus.DEVICE, decision1);


        // this method SHOULD fail for now, since it calls on demote for NESTED memory chunk
        try {
            AllocationStatus decision2 = allocator.makeDemoteDecision(objectId1, shape2);
            assertTrue(false);
        } catch (Exception e) {
            assertTrue(true);
        }
    }

    /**
     * This test should decline demote, since:
     *  1.  Nested memory is in use
     * @throws Exception
     */
    @Test
    public void testNestedDemoteDecision2() throws Exception {
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

        AllocationShape shape2 = new AllocationShape();
        shape2.setDataType(DataBuffer.Type.FLOAT);
        shape2.setLength(512 * 1024L);
        shape2.setOffset(100);
        shape2.setStride(1);

        AllocationPoint point = allocator.getAllocationPoint(objectId1);
        Object pointer1 = allocator.getDevicePointer(objectId1);
        allocator.relocateMemory(objectId1, AllocationStatus.DEVICE);

        // We are checking, that old pointer does not equal to new one, since memory was relocated
        assertNotEquals(pointer1, allocator.getAllocationPoint(objectId1));

        pointer1 = allocator.getDevicePointer(objectId1);
        allocator.tackDevice(objectId1, shape1);

        allocator.registerSpan(objectId1, shape2);

        Object pointer2 = allocator.getDevicePointer(objectId1, shape2);
        assertNotEquals(pointer1, pointer2);

        assertNotEquals(pointer2, point.getCudaPointer());
        assertEquals(pointer1, point.getCudaPointer());
        assertEquals(AllocationStatus.DEVICE, point.getAllocationStatus());

        AllocationStatus decision1 = allocator.makeDemoteDecision(objectId1, shape1);
        assertEquals(AllocationStatus.DEVICE, decision1);
    }

    /**
     * This test should accept demote, since memory space isn't used at this moment, as well as no used descendants
     *
     * @throws Exception
     */
    @Test
    public void testNestedDemoteDecision3() throws Exception {
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

        AllocationShape shape2 = new AllocationShape();
        shape2.setDataType(DataBuffer.Type.FLOAT);
        shape2.setLength(512 * 1024L);
        shape2.setOffset(100);
        shape2.setStride(1);

        AllocationPoint point = allocator.getAllocationPoint(objectId1);
        Object pointer1 = allocator.getDevicePointer(objectId1);
        allocator.relocateMemory(objectId1, AllocationStatus.DEVICE);

        // We are checking, that old pointer does not equal to new one, since memory was relocated
        assertNotEquals(pointer1, allocator.getAllocationPoint(objectId1));

        pointer1 = allocator.getDevicePointer(objectId1);
        allocator.tackDevice(objectId1, shape1);

        allocator.registerSpan(objectId1, shape2);

        Object pointer2 = allocator.getDevicePointer(objectId1, shape2);
        allocator.tackDevice(objectId1, shape2);
        assertNotEquals(pointer1, pointer2);

        assertNotEquals(pointer2, point.getCudaPointer());
        assertEquals(pointer1, point.getCudaPointer());
        assertEquals(AllocationStatus.DEVICE, point.getAllocationStatus());

        AllocationStatus decision1 = allocator.makeDemoteDecision(objectId1, shape1);
        assertEquals(AllocationStatus.ZERO, decision1);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////                DECISION MAKING ON PARTIAL ALLOCATION (TOE) TESTS, SINGLE DEVICE
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /*
        This set of tests addresses situation, where whole underlying array/buffer can't be allocated on device.
        But parts of this array are accessed separately, though they could be partially allocated on device
     */


    /**
     * This test should decline partial promotion, since it's not implemented now
     *
     * @throws Exception
     */
    @Test
    public void testSingleDeviceToe1() throws Exception {
        BasicAllocator allocator = new BasicAllocator();
        allocator.setEnvironment(singleDevice4GBcc52);
        allocator.setBalancer(new FirstInBalancer());
        allocator.setMover(new DummyMover());

        assertEquals(0, singleDevice4GBcc52.getAllocatedMemoryForDevice(1));

        Long objectId1 = 22L;

        AllocationShape shape1 = new AllocationShape();
        shape1.setDataType(DataBuffer.Type.FLOAT);
        shape1.setLength(128 * 1024 * 1024L);
        shape1.setOffset(0);
        shape1.setStride(1);

        allocator.registerSpan(objectId1, shape1);


        AllocationShape shape2 = new AllocationShape();
        shape2.setDataType(DataBuffer.Type.FLOAT);
        shape2.setLength(1 * 1024 * 1024L);
        shape2.setOffset(1);
        shape2.setStride(1);


        AllocationStatus target = allocator.makePromoteDecision(objectId1, shape1);
        assertEquals(AllocationStatus.ZERO, target);

        allocator.registerSpan(objectId1, shape2);

        AllocationStatus target2 = allocator.makePromoteDecision(objectId1, shape2);
        assertEquals(AllocationStatus.ZERO, target2);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////                BLIND ACCESS TESTS
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    @Test
    public void testBlindAccessSingle1() throws Exception {
        BasicAllocator allocator = new BasicAllocator();
        allocator.setEnvironment(singleDevice4GBcc52);
        allocator.setBalancer(new FirstInBalancer());
        allocator.setMover(new DummyMover());

        assertEquals(0, singleDevice4GBcc52.getAllocatedMemoryForDevice(1));

        Long objectId1 = 22L;

        AllocationShape shape1 = new AllocationShape();
        shape1.setDataType(DataBuffer.Type.FLOAT);
        shape1.setLength(2 * 1024 * 1024L);
        shape1.setOffset(0);
        shape1.setStride(1);

        // we don't call registerSpan explicitly, registration is hidden within getDevicePointer()
        Object pointer = allocator.getDevicePointer(objectId1, shape1);

        assertNotEquals(null, pointer);

        AllocationPoint point = allocator.getAllocationPoint(objectId1);

        assertEquals(AllocationStatus.ZERO, point.getAllocationStatus());

        assertEquals(pointer, point.getCudaPointer());
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////                STALE DETECTION TESTS
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * In this test we assume that there's some allocated zero-copy pinned memory that wasn't accessed for a long time
     *
     * @throws Exception
     */
    @Test
    public void testStaleZeroDetection1() throws Exception {
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

        Long objectId2 = 25L;

        AllocationShape shape2 = new AllocationShape();
        shape2.setDataType(DataBuffer.Type.FLOAT);
        shape2.setLength(1 * 1024 * 1024L);
        shape2.setOffset(0);
        shape2.setStride(1);

        allocator.registerSpan(objectId2, shape2);

        Object pointer = allocator.getDevicePointer(objectId1, shape1);
        allocator.tackDevice(objectId1, shape1);

        Object pointer2 = allocator.getDevicePointer(objectId2, shape2);
        allocator.tackDevice(objectId2, shape2);

        AllocationPoint point = allocator.getAllocationPoint(objectId1);
        assertEquals(AllocationStatus.ZERO, point.getAllocationStatus());

        AllocationPoint point2 = allocator.getAllocationPoint(objectId2);
        assertEquals(AllocationStatus.ZERO, point2.getAllocationStatus());

        assertEquals(1, point.getTimerShort().getNumberOfEvents());
        assertEquals(1, point.getTimerLong().getNumberOfEvents());

        SimpleTimer timerShort = new SimpleTimer(10, TimeUnit.SECONDS);
        SimpleTimer timerLong = new SimpleTimer(60, TimeUnit.SECONDS);

        assertEquals(0, timerLong.getNumberOfEvents());
        assertEquals(0, timerShort.getNumberOfEvents());

        point.setTimerLong(timerLong);
        point.setTimerShort(timerShort);

        assertEquals(2, allocator.zeroTableSize());
        assertEquals(2, allocator.tableSize());
        assertEquals(0, allocator.deviceTableSize());

        allocator.deallocateUnused();

        assertEquals(1, allocator.zeroTableSize());
        assertEquals(2, allocator.tableSize());
        assertEquals(0, allocator.deviceTableSize());

        assertEquals(AllocationStatus.HOST, point.getAllocationStatus());
    }




    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////                STALE OBJECT TESTS
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * This test addresses impossible copyback situation on dead objects.
     * I.e. memory was allocated on device, or in pinned area, and was never taken back.
     *
     * @throws Exception
     */
    @Test
    public void testSingleStale1() throws Exception {
        // TODO: test REQUIRED here
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////                AUTO-PROMOTION TESTS
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    @Test
    public void testAutoPromoteSingle1() throws Exception {
        Configuration configuration = new Configuration();
        configuration.setAllocAggressiveness(Aggressiveness.IMMEDIATE);
        configuration.setZeroCopyFallbackAllowed(true);
        configuration.setMinimumRelocationThreshold(4);

        BasicAllocator allocator = new BasicAllocator();
        allocator.applyConfiguration(configuration);
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

        for (int x = 0; x < 6; x++) {
            allocator.getDevicePointer(objectId1);
            allocator.tackDevice(objectId1, shape1);
        }

        assertEquals(AllocationStatus.DEVICE, point.getAllocationStatus());

        assertEquals(1, allocator.getDeviceAllocations().size());
    }


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////                                                    MULTITHREADED TESTS
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    @Test
    public void testMultithreadedStraightAccess1() throws Exception {
        Configuration configuration = new Configuration();

        final BasicAllocator allocator = new BasicAllocator();
        allocator.applyConfiguration(configuration);
        allocator.setEnvironment(singleDevice4GBcc52);
        allocator.setBalancer(new FirstInBalancer());
        allocator.setMover(new DummyMover());

        // create some objects with specific shapes
        final List<Long> objects = new ArrayList<>();
        for (int x = 0; x < 1000; x++) {
            Long objectId1 = new Long(x);

            AllocationShape shape = new AllocationShape();
            shape.setDataType(DataBuffer.Type.FLOAT);
            shape.setLength(1 * 1024 * 1024L);
            shape.setOffset(0);
            shape.setStride(1);

            allocator.registerSpan(objectId1, shape);
            objects.add(objectId1);
        }

        ThreadPoolExecutor service = new ThreadPoolExecutor(50, 150, 10, TimeUnit.SECONDS, new LinkedBlockingQueue<Runnable>());

        Runnable runnable = new Runnable() {
            @Override
            public void run() {
                Random rand = new Random();

                for (int x = 0; x< 1000; x++) {
                    int rnd = rand.nextInt(objects.size());
                    Long cObject = objects.get(rnd);
                    AllocationPoint point = allocator.getAllocationPoint(cObject);
                    Object pointer = allocator.getDevicePointer(cObject);

                    allocator.tackDevice(cObject, point.getShape());
                }
            }
        };

        for (int x = 0; x< 1000; x++) {
            service.execute(runnable);
        }

        while (service.getActiveCount() != 0) {
            Thread.sleep(500);
        }

        // At this point we had a number of accesses being done
        long allocatedMemory = singleDevice4GBcc52.getAllocatedMemoryForDevice(1);
        log.info("Allocated memory: " + allocatedMemory + " Max allocation: " + configuration.getMaximumDeviceAllocation());
        assertNotEquals(0, allocatedMemory);

        // we should NOT have memory allocated beyond max allocation
        assertTrue(allocatedMemory <= configuration.getMaximumDeviceAllocation());

        // now we emulate situation with unused memory
        for (Long object: objects) {
            AllocationPoint point  = allocator.getAllocationPoint(object);

            SimpleTimer timerShort = new SimpleTimer(10, TimeUnit.SECONDS);
            SimpleTimer timerLong = new SimpleTimer(60, TimeUnit.SECONDS);

            point.setTimerLong(timerLong);
            point.setTimerShort(timerShort);
        }

        /*
            So, at this point we have no memory used, and we force deallocation
        */
        allocator.deallocateUnused();



        for (Long object: objects) {
            AllocationPoint point  = allocator.getAllocationPoint(object);

            assertEquals(AllocationStatus.HOST, point.getAllocationStatus());
        }

        allocatedMemory = singleDevice4GBcc52.getAllocatedMemoryForDevice(1);
        log.info("Allocated memory: " + allocatedMemory + " Max allocation: " + configuration.getMaximumDeviceAllocation());
        assertEquals(0, allocatedMemory);
    }

    /**
     * This test simulates following real-world use situation:
     *  1. We have few hot arrays
     *  2. We have few*2 warm arrays
     *  3. We have loads of cold arrays.
     *
     * Data gets accessed according their ap, and at the end of day we should have the following situation:
     * 1. Hot arrays located on device
     * 2. Warm arrays located on device and on zero
     * 3. Everything else is located on zero
     *
     * @throws Exception
     */
    @Test
    public void testMultithreadedCrossedPressure1() throws Exception {
        Configuration configuration = new Configuration();

        final BasicAllocator allocator = new BasicAllocator();
        allocator.applyConfiguration(configuration);
        allocator.setEnvironment(singleDevice4GBcc52);
        allocator.setBalancer(new FirstInBalancer());
        allocator.setMover(new DummyMover());

        // create HOT objects
        final List<Long> hotObjects = new ArrayList<>();
        for (int x = 0; x < 100; x++) {
            Long objectId1 = new Long(x);

            AllocationShape shape = new AllocationShape();
            shape.setDataType(DataBuffer.Type.FLOAT);
            shape.setLength(768 * 1024L);
            shape.setOffset(0);
            shape.setStride(1);

            allocator.registerSpan(objectId1, shape);
            hotObjects.add(objectId1);
        }

        // create some WARM objects with specific shapes
        final List<Long> warmObjects = new ArrayList<>();
        for (int x = 100; x < 300; x++) {
            Long objectId1 = new Long(x);

            AllocationShape shape = new AllocationShape();
            shape.setDataType(DataBuffer.Type.FLOAT);
            shape.setLength(8192L);
            shape.setOffset(0);
            shape.setStride(1);

            allocator.registerSpan(objectId1, shape);
            warmObjects.add(objectId1);
        }

        // create some COLD objects with specific shapes
        final List<Long> coldObjects = new ArrayList<>();
        for (int x = 1100; x < 300000; x++) {
            Long objectId1 = new Long(x);

            AllocationShape shape = new AllocationShape();
            shape.setDataType(DataBuffer.Type.FLOAT);
            shape.setLength(1024L);
            shape.setOffset(0);
            shape.setStride(1);

            allocator.registerSpan(objectId1, shape);
            coldObjects.add(objectId1);
        }

        ThreadPoolExecutor service = new ThreadPoolExecutor(50, 150, 10, TimeUnit.SECONDS, new LinkedBlockingQueue<Runnable>());

        // now, we emulate cold -> warm -> hot access using 1 x 1 x 3 pattern
        for (int x = 0; x < hotObjects.size() * 20; x++) {
            Runnable newRunnable = new Runnable() {
                @Override
                public void run() {
                    Random rnd = new Random();
                    Long hotObject = hotObjects.get(rnd.nextInt(hotObjects.size()));
                    AllocationPoint point = allocator.getAllocationPoint(hotObject);

                    for (int x = 0; x < 10; x++) {
                        allocator.getDevicePointer(hotObject, point.getShape());
                        allocator.tackDevice(hotObject, point.getShape());
                    }

                    // warm object access, do 3 times
                    for (int x = 0; x < 3; x++) {
                        Long warmObject = warmObjects.get(rnd.nextInt(warmObjects.size()));
                        AllocationPoint pointWarm = allocator.getAllocationPoint(warmObject);

                        allocator.getDevicePointer(warmObject, pointWarm.getShape());
                        allocator.tackDevice(warmObject, pointWarm.getShape());
                    }

                    // cold object access, do once
                    Long coldObject = coldObjects.get(rnd.nextInt(coldObjects.size()));
                    AllocationPoint pointWarm = allocator.getAllocationPoint(coldObject);

                    allocator.getDevicePointer(coldObject, pointWarm.getShape());
                    allocator.tackDevice(coldObject, pointWarm.getShape());
                }
            };

            service.execute(newRunnable);
        }

        while (service.getActiveCount() != 0) {
            Thread.sleep(500);
        }

        // all hot objects should reside in device memory in this case
        double averageRate = 0;
        int device = 0;
        for (Long hotObject: hotObjects) {
            AllocationPoint point = allocator.getAllocationPoint(hotObject);

            averageRate += point.getTimerLong().getFrequencyOfEvents();

            if (point.getAllocationStatus() == AllocationStatus.DEVICE)
                device++;
        }
        log.info("Hot objects in memory: [" + device + "], Average rate: ["+ (averageRate / hotObjects.size())+"]");
        // only 85 hot objects could fit in memory within current environment and configuration
        assertTrue(device > 50);

        // some of warm objects MIGHT be in device memory
        device = 0;
        averageRate = 0;
        for (Long warmObject: warmObjects) {
            AllocationPoint point = allocator.getAllocationPoint(warmObject);

            averageRate += point.getTimerLong().getFrequencyOfEvents();

            if (point.getAllocationStatus() == AllocationStatus.DEVICE)
                device++;
        }
        log.info("Warm objects in memory: [" + device + "], Average rate: ["+ (averageRate / warmObjects.size())+"]");
        assertNotEquals(0, device);

        // cold objects MIGHT be in device memory too, but their number should be REALLY low
        device = 0;
        averageRate = 0;
        for (Long coldObject: coldObjects) {
            AllocationPoint point = allocator.getAllocationPoint(coldObject);

            averageRate += point.getTimerLong().getFrequencyOfEvents();

            if (point.getAllocationStatus() == AllocationStatus.DEVICE)
                device++;
        }
        log.info("Cold objects in memory: [" + device + "], Average rate: ["+ (averageRate / coldObjects.size())+"]");
        assertTrue(device < 20);

        long allocatedMemory = singleDevice4GBcc52.getAllocatedMemoryForDevice(1);
        log.info("Allocated memory: " + allocatedMemory + " Max allocation: " + configuration.getMaximumDeviceAllocation());
        assertNotEquals(0, allocatedMemory);
    }

    /**
     * This test is addressing preemptive relocation. We have the following setup:
     *  1. We have few hot arrays
     *  2. We have few*2 warm arrays
     *  3. We have loads of cold arrays.
     *
     *  But in this scenario we have memory filled with another objects, that are NOT uses for quite some time.
     *
     *  At the end of day we should see the following state:
     *  1. All older objects are moved away from memory and removed from allocation tables
     *  2. As much as possible hot objects stored on gpu
     *  3. Some of warm objects are stored on gpu, the rest are in zero memory
     *  4. Cold objects shouldn't be on device memory.
     *
     *
     *
     * @throws Exception
     */
    @Test
    public void testMultithreadedPreemptiveRelocation1() throws Exception {
        Configuration configuration = new Configuration();

        final BasicAllocator allocator = new BasicAllocator();
        allocator.applyConfiguration(configuration);
        allocator.setEnvironment(singleDevice4GBcc52);
        allocator.setBalancer(new FirstInBalancer());
        allocator.setMover(new DummyMover());

        final Random rnd = new Random();

        // thats our initial objects, that are directly seeded into gpu memory
        final List<Long> initialObjects = new ArrayList<>();
        for (int x = 0; x< 50; x++) {
            Long objectId1 = new Long(rnd.nextInt(1000000) + 10000000L);

            AllocationShape shape = new AllocationShape();
            shape.setDataType(DataBuffer.Type.FLOAT);
            shape.setLength((rnd.nextInt(256) + 10) * 1024L);
            shape.setOffset(0);
            shape.setStride(1);

            log.info("Allocating ID: " + objectId1 + " Memory size: " + AllocationUtils.getRequiredMemory(shape));

            allocator.registerSpan(objectId1, shape);
            initialObjects.add(objectId1);

            allocator.getDevicePointer(objectId1, shape);
            allocator.tackDevice(objectId1, shape);

            // we force each initial object to be moved into device memory, but tag it as cold
            allocator.relocateMemory(objectId1, AllocationStatus.DEVICE);

            AllocationPoint point = allocator.getAllocationPoint(objectId1);
            assertEquals(AllocationStatus.DEVICE, point.getAllocationStatus());

            SimpleTimer timerLong = new SimpleTimer(60, TimeUnit.SECONDS);
            SimpleTimer timerShort = new SimpleTimer(10, TimeUnit.SECONDS);

            point.setTimerShort(timerShort);
            point.setTimerLong(timerLong);
        }

        long allocatedMemory = singleDevice4GBcc52.getAllocatedMemoryForDevice(1);
        log.info("Allocated memory: " + allocatedMemory + " Max allocation: " + configuration.getMaximumDeviceAllocation());
        assertNotEquals(0, allocatedMemory);


        // create HOT objects
        final List<Long> hotObjects = new ArrayList<>();
        for (int x = 100; x < 200; x++) {
            Long objectId1 = new Long(x);

            AllocationShape shape = new AllocationShape();
            shape.setDataType(DataBuffer.Type.FLOAT);
            shape.setLength(768 * 1024L);
            shape.setOffset(0);
            shape.setStride(1);

            allocator.registerSpan(objectId1, shape);
            hotObjects.add(objectId1);
        }

        // create some WARM objects with specific shapes
        final List<Long> warmObjects = new ArrayList<>();
        for (int x = 200; x < 500; x++) {
            Long objectId1 = new Long(x);

            AllocationShape shape = new AllocationShape();
            shape.setDataType(DataBuffer.Type.FLOAT);
            shape.setLength(8192L);
            shape.setOffset(0);
            shape.setStride(1);

            allocator.registerSpan(objectId1, shape);
            warmObjects.add(objectId1);
        }

        // create some COLD objects with specific shapes
        final List<Long> coldObjects = new ArrayList<>();
        for (int x = 500; x < 300000; x++) {
            Long objectId1 = new Long(x);

            AllocationShape shape = new AllocationShape();
            shape.setDataType(DataBuffer.Type.FLOAT);
            shape.setLength(1024L);
            shape.setOffset(0);
            shape.setStride(1);

            allocator.registerSpan(objectId1, shape);
            coldObjects.add(objectId1);
        }


        ThreadPoolExecutor service = new ThreadPoolExecutor(50, 150, 10, TimeUnit.SECONDS, new LinkedBlockingQueue<Runnable>());

        // now, we emulate cold -> warm -> hot access using 1 x 1 x 3 pattern
        for (int x = 0; x < hotObjects.size() * 20; x++) {
            Runnable newRunnable = new Runnable() {
                @Override
                public void run() {
                    Random rnd = new Random();
                    Long hotObject = hotObjects.get(rnd.nextInt(hotObjects.size()));
                    AllocationPoint point = allocator.getAllocationPoint(hotObject);

                    for (int x = 0; x < 10; x++) {
                        allocator.getDevicePointer(hotObject, point.getShape());
                        allocator.tackDevice(hotObject, point.getShape());
                    }

                    // warm object access, do 3 times
                    for (int x = 0; x < 3; x++) {
                        Long warmObject = warmObjects.get(rnd.nextInt(warmObjects.size()));
                        AllocationPoint pointWarm = allocator.getAllocationPoint(warmObject);

                        allocator.getDevicePointer(warmObject, pointWarm.getShape());
                        allocator.tackDevice(warmObject, pointWarm.getShape());
                    }

                    // cold object access, do once
                    Long coldObject = coldObjects.get(rnd.nextInt(coldObjects.size()));
                    AllocationPoint pointWarm = allocator.getAllocationPoint(coldObject);

                    allocator.getDevicePointer(coldObject, pointWarm.getShape());
                    allocator.tackDevice(coldObject, pointWarm.getShape());
                }
            };

            service.execute(newRunnable);
        }

        while (service.getActiveCount() != 0) {
            Thread.sleep(500);
        }

        int device = 0;
        double averageRate = 0;
        for (Long initial: initialObjects) {
            AllocationPoint point = allocator.getAllocationPoint(initial);

            averageRate += point.getTimerLong().getFrequencyOfEvents();

            if (point.getAllocationStatus() == AllocationStatus.DEVICE)
                device++;
        }
        log.info("Initial objects in memory: [" + device + "], Average rate: ["+ (averageRate / initialObjects.size())+"]");
        assertEquals(0, device);

        // all hot objects should reside in device memory in this case
        averageRate = 0;
        device = 0;
        for (Long hotObject: hotObjects) {
            AllocationPoint point = allocator.getAllocationPoint(hotObject);

            averageRate += point.getTimerLong().getFrequencyOfEvents();

            if (point.getAllocationStatus() == AllocationStatus.DEVICE)
                device++;
        }
        log.info("Hot objects in memory: [" + device + "], Average rate: ["+ (averageRate / hotObjects.size())+"]");
        // only 85 hot objects could fit in memory within current environment and configuration
        assertTrue(device > 50);

        // some of warm objects MIGHT be in device memory
        device = 0;
        averageRate = 0;
        for (Long warmObject: warmObjects) {
            AllocationPoint point = allocator.getAllocationPoint(warmObject);

            averageRate += point.getTimerLong().getFrequencyOfEvents();

            if (point.getAllocationStatus() == AllocationStatus.DEVICE)
                device++;
        }
        log.info("Warm objects in memory: [" + device + "], Average rate: ["+ (averageRate / warmObjects.size())+"]");
        assertNotEquals(0, device);

        // cold objects MIGHT be in device memory too, but their number should be REALLY low
        device = 0;
        averageRate = 0;
        for (Long coldObject: coldObjects) {
            AllocationPoint point = allocator.getAllocationPoint(coldObject);

            averageRate += point.getTimerLong().getFrequencyOfEvents();

            if (point.getAllocationStatus() == AllocationStatus.DEVICE)
                device++;
        }
        log.info("Cold objects in memory: [" + device + "], Average rate: ["+ (averageRate / coldObjects.size())+"]");
        assertTrue(device < 20);
    }

    /**
     * This test is addressing preemptive relocation. We have the following setup:
     *  1. We have few old hot arrays
     *  2. We have few NEW hot arrays.
     *  3. We have few*2 warm arrays
     *  4. We have loads of cold arrays.
     *
     *  But in this scenario we have memory filled with another initial objects, that are used for quite some time.
     *  After some cycles we change uses set of objects, WITHOUT hints to allocator.
     *
     *  At the end of day we should see the following state:
     *  1. All older objects are moved away from memory and removed from allocation tables
     *  2. As much as possible new hot objects stored on gpu
     *  3. Some of warm objects are stored on gpu, the rest are in zero memory
     *  4. Cold objects shouldn't be on device memory.
     *
     * The main difference between this test and testMultithreadedPreemptiveRelocation1 is the score-based relocation
     *
     *
     * @throws Exception
     */
    @Test
    public void testMultithreadedPreemptiveRelocation2() throws Exception {
        Configuration configuration = new Configuration();

        final BasicAllocator allocator = new BasicAllocator();
        allocator.applyConfiguration(configuration);
        allocator.setEnvironment(singleDevice4GBcc52);
        allocator.setBalancer(new FirstInBalancer());
        allocator.setMover(new DummyMover());

        final Random rnd = new Random();

        // thats our initial objects, that are directly seeded into gpu memory
        final List<Long> initialObjects = new ArrayList<>();
        for (int x = 0; x< 50; x++) {
            Long objectId1 = new Long(rnd.nextInt(1000000) + 10000000L);

            AllocationShape shape = new AllocationShape();
            shape.setDataType(DataBuffer.Type.FLOAT);
            shape.setLength((rnd.nextInt(256) + 10) * 1024L);
            shape.setOffset(0);
            shape.setStride(1);

            log.info("Allocating ID: " + objectId1 + " Memory size: " + AllocationUtils.getRequiredMemory(shape));

            allocator.registerSpan(objectId1, shape);
            initialObjects.add(objectId1);

            for (int y = 0; y < 20; y++) {
                allocator.getDevicePointer(objectId1, shape);
                allocator.tackDevice(objectId1, shape);
            }

            AllocationPoint point = allocator.getAllocationPoint(objectId1);
            assertEquals(AllocationStatus.DEVICE, point.getAllocationStatus());
        }

        long allocatedMemory = singleDevice4GBcc52.getAllocatedMemoryForDevice(1);
        log.info("Allocated memory: " + allocatedMemory + " Max allocation: " + configuration.getMaximumDeviceAllocation());
        assertNotEquals(0, allocatedMemory);


        // create HOT objects
        final List<Long> hotObjects = new ArrayList<>();
        for (int x = 100; x < 200; x++) {
            Long objectId1 = new Long(x);

            AllocationShape shape = new AllocationShape();
            shape.setDataType(DataBuffer.Type.FLOAT);
            shape.setLength(768 * 1024L);
            shape.setOffset(0);
            shape.setStride(1);

            allocator.registerSpan(objectId1, shape);
            hotObjects.add(objectId1);
        }

        // create some WARM objects with specific shapes
        final List<Long> warmObjects = new ArrayList<>();
        for (int x = 200; x < 500; x++) {
            Long objectId1 = new Long(x);

            AllocationShape shape = new AllocationShape();
            shape.setDataType(DataBuffer.Type.FLOAT);
            shape.setLength(8192L);
            shape.setOffset(0);
            shape.setStride(1);

            allocator.registerSpan(objectId1, shape);
            warmObjects.add(objectId1);
        }

        // create some COLD objects with specific shapes
        final List<Long> coldObjects = new ArrayList<>();
        for (int x = 500; x < 300000; x++) {
            Long objectId1 = new Long(x);

            AllocationShape shape = new AllocationShape();
            shape.setDataType(DataBuffer.Type.FLOAT);
            shape.setLength(1024L);
            shape.setOffset(0);
            shape.setStride(1);

            allocator.registerSpan(objectId1, shape);
            coldObjects.add(objectId1);
        }


        ThreadPoolExecutor service = new ThreadPoolExecutor(50, 150, 10, TimeUnit.SECONDS, new LinkedBlockingQueue<Runnable>());

        // now, we emulate cold -> warm -> hot access using 1 x 1 x 3 pattern
        for (int x = 0; x < hotObjects.size() * 20; x++) {
            Runnable newRunnable = new Runnable() {
                @Override
                public void run() {
                    Random rnd = new Random();
                    Long hotObject = hotObjects.get(rnd.nextInt(hotObjects.size()));
                    AllocationPoint point = allocator.getAllocationPoint(hotObject);

                    for (int x = 0; x < 10; x++) {
                        allocator.getDevicePointer(hotObject, point.getShape());
                        allocator.tackDevice(hotObject, point.getShape());
                    }

                    // warm object access, do 3 times
                    for (int x = 0; x < 3; x++) {
                        Long warmObject = warmObjects.get(rnd.nextInt(warmObjects.size()));
                        AllocationPoint pointWarm = allocator.getAllocationPoint(warmObject);

                        allocator.getDevicePointer(warmObject, pointWarm.getShape());
                        allocator.tackDevice(warmObject, pointWarm.getShape());
                    }

                    // cold object access, do once
                    Long coldObject = coldObjects.get(rnd.nextInt(coldObjects.size()));
                    AllocationPoint pointWarm = allocator.getAllocationPoint(coldObject);

                    allocator.getDevicePointer(coldObject, pointWarm.getShape());
                    allocator.tackDevice(coldObject, pointWarm.getShape());
                }
            };

            service.execute(newRunnable);
        }

        while (service.getActiveCount() != 0) {
            Thread.sleep(500);
        }

        int device = 0;
        double averageRate = 0;
        for (Long initial: initialObjects) {
            AllocationPoint point = allocator.getAllocationPoint(initial);

            averageRate += point.getTimerLong().getFrequencyOfEvents();

            if (point.getAllocationStatus() == AllocationStatus.DEVICE)
                device++;
        }
        log.info("Initial objects in memory: [" + device + "], Average rate: ["+ (averageRate / initialObjects.size())+"]");
        assertEquals(0, device);

        // all hot objects should reside in device memory in this case
        averageRate = 0;
        device = 0;
        for (Long hotObject: hotObjects) {
            AllocationPoint point = allocator.getAllocationPoint(hotObject);

            averageRate += point.getTimerLong().getFrequencyOfEvents();

            if (point.getAllocationStatus() == AllocationStatus.DEVICE)
                device++;
        }
        log.info("Hot objects in memory: [" + device + "], Average rate: ["+ (averageRate / hotObjects.size())+"]");
        // only 85 hot objects could fit in memory within current environment and configuration
        assertTrue(device > 50);

        // some of warm objects MIGHT be in device memory
        device = 0;
        averageRate = 0;
        for (Long warmObject: warmObjects) {
            AllocationPoint point = allocator.getAllocationPoint(warmObject);

            averageRate += point.getTimerLong().getFrequencyOfEvents();

            if (point.getAllocationStatus() == AllocationStatus.DEVICE)
                device++;
        }
        log.info("Warm objects in memory: [" + device + "], Average rate: ["+ (averageRate / warmObjects.size())+"]");
        assertNotEquals(0, device);

        // cold objects MIGHT be in device memory too, but their number should be REALLY low
        device = 0;
        averageRate = 0;
        for (Long coldObject: coldObjects) {
            AllocationPoint point = allocator.getAllocationPoint(coldObject);

            averageRate += point.getTimerLong().getFrequencyOfEvents();

            if (point.getAllocationStatus() == AllocationStatus.DEVICE)
                device++;
        }
        log.info("Cold objects in memory: [" + device + "], Average rate: ["+ (averageRate / coldObjects.size())+"]");
        assertTrue(device < 20);
    }


    /**
     *
     * This test addresses device access from multiple threads.
     *
     * At the end of test objects registered within different threads, should have different devices used.
     *
     * @throws Exception
     */
    @Test
    public void testMultipleDevicesAllocation1() throws Exception {
        Configuration configuration = new Configuration();

        final BasicAllocator allocator = new BasicAllocator();
        allocator.applyConfiguration(configuration);
        allocator.setEnvironment(fourDevices4GBcc52);
        allocator.setBalancer(new FirstInBalancer());
        allocator.setMover(new DummyMover());


        ThreadPoolExecutor service = new ThreadPoolExecutor(50, 150, 10, TimeUnit.SECONDS, new LinkedBlockingQueue<Runnable>());

        final Map<Integer, List<Long>> objects = new ConcurrentHashMap<>();

        // we emulate 16 executor nodes that launching some calculations
        for (int y = 0; y < 16; y++) {
            objects.put(y, new CopyOnWriteArrayList<Long>());

            ManagedRunnable1 runnable = new ManagedRunnable1(y, allocator, objects);

            service.execute(runnable);
        }


        while (service.getActiveCount() != 0) {
            Thread.sleep(500);
        }

        assertEquals(16, objects.size());
        assertEquals(100, objects.get(0).size());

        Map<Integer, AtomicInteger> devicesUsed = new ConcurrentHashMap<>();

        for (int y = 0; y < 16; y++) {
            // prefetching valid deviceId that will be used for future checks
            Integer validDevice = allocator.getAllocationPoint(objects.get(y).get(0)).getDeviceId();

            for (Long object: objects.get(y)) {
                AllocationPoint point = allocator.getAllocationPoint(object);

                Integer deviceId = point.getDeviceId();
                assertNotEquals(null, deviceId);

                // every allocation within this bucket should use the same device
                assertEquals(validDevice, deviceId);

                if (!devicesUsed.containsKey(deviceId))
                    devicesUsed.put(deviceId, new AtomicInteger(0));

                devicesUsed.get(deviceId).incrementAndGet();
            }
        }

        // we should have 4 devices used now.
        assertEquals(4, devicesUsed.size());
    }

    /**
     * This simple test addresses upper allocation boundaries for both device and zero-copy memory.
     *
     * We'll throw in memory, as much as we can.
     *
     * Allocation should never step beyond maximum per-device limits, as well as maximum zero-copy limits.
     * In real-world use zero-copy limit will be derived from -Xmx jvm value, but right now we'll imitate it using Configuration
     *
     * @throws Exception
     */
    @Test
    public void testSingleDeviceBoundedAllocation1() throws Exception {
        final Configuration configuration = new Configuration();
        configuration.setMaximumZeroAllocation(200 * 4 *  1024L * 1024L);

        final BasicAllocator allocator = new BasicAllocator();
        allocator.applyConfiguration(configuration);
        allocator.setEnvironment(singleDevice4GBcc52);
        allocator.setBalancer(new FirstInBalancer());
        allocator.setMover(new DummyMover());


        // create HOT objects
        final List<Long> hotObjects = new ArrayList<>();
        for (int x = 0; x < 200; x++) {
            Long objectId1 = new Long(x);

            AllocationShape shape = new AllocationShape();
            shape.setDataType(DataBuffer.Type.FLOAT);
            shape.setLength(2 * 1024 * 1024L);
            shape.setOffset(0);
            shape.setStride(1);

            allocator.registerSpan(objectId1, shape);
            hotObjects.add(objectId1);
        }

        ThreadPoolExecutor service = new ThreadPoolExecutor(50, 150, 10, TimeUnit.SECONDS, new LinkedBlockingQueue<Runnable>());

        /*
            now we just imitate load on hot objects, and generate new cold objects on the fly. something like training NN and passing in data in iterations.
        */

        final List<Long> coldObjects = new CopyOnWriteArrayList<>();

        for (int x = 0; x < 20000; x++) {
            Runnable runnable = new Runnable() {
                Random rnd = new Random();
                @Override
                public void run() {
                    Long hotObject = hotObjects.get(rnd.nextInt(hotObjects.size()));

                    for (int y = 0; y < 50; y++) {
                        Long coldObject = new Long(rnd.nextLong() + 200);

                        AllocationShape shape = new AllocationShape();
                        shape.setDataType(DataBuffer.Type.FLOAT);
                        shape.setLength(2 * 1024 * 1024L);
                        shape.setOffset(0);
                        shape.setStride(1);

                        allocator.registerSpan(coldObject, shape);

                        allocator.getDevicePointer(hotObject);
                        allocator.tackDevice(coldObject, shape);

                        allocator.getDevicePointer(coldObject);
                        allocator.tackDevice(coldObject, shape);

                        coldObjects.add(coldObject);

                        assertNotEquals(0, allocator.getHostAllocatedMemory());
                        assertTrue(allocator.getHostAllocatedMemory() < configuration.getMaximumZeroAllocation());
                    }

                }
            };

            service.execute(runnable);
        }

        while (service.getActiveCount() != 0) {
            Thread.sleep(500);
        }

        assertNotEquals(0, allocator.getHostAllocatedMemory());
        assertTrue(allocator.getHostAllocatedMemory() < configuration.getMaximumZeroAllocation());
    }

    /**
     * private utility class, used in testMultipleDevicesAllocation1()
     */
    private static class ManagedRunnable1 implements Runnable {
        private int Y;
        private BasicAllocator allocator;
        private Map<Integer, List<Long>> objects;

        public ManagedRunnable1(int Y, BasicAllocator basicAllocator, Map<Integer, List<Long>> objects) {
            this.Y = Y;
            this.allocator = basicAllocator;
            this.objects = objects;
        }

        @Override
        public void run() {
            Random random = new Random();

            for (int x = 1; x <= 100; x++) {
                Long objectId = new Long((Y *x) + x + random.nextInt());

                AllocationShape shape = new AllocationShape();
                shape.setDataType(DataBuffer.Type.FLOAT);
                shape.setLength(8192L);
                shape.setOffset(0);
                shape.setStride(1);

                allocator.registerSpan(objectId, shape);

                objects.get(Y).add(objectId);
            }
        }
    }
}
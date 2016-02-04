package org.nd4j.jita.allocator.impl;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.enums.SyncState;
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

    private static Logger log = LoggerFactory.getLogger(BasicAllocatorTest.class);

    @Before
    public void setUp() throws Exception {

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

        allocator.validateHostData(objectId);
        assertNotEquals(0, point.getAccessHost());
        assertEquals(point.getAccessDevice(), point.getAccessHost());
        assertEquals(SyncState.SYNC, point.getHostMemoryState());

        assertEquals(1, point.getNumberOfDescendants());
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

        allocator.getDevicePointer(objectId);

        assertEquals(1, point.getDescendantTicks(shape));

        AllocationShape shape2 = new AllocationShape();
        shape2.setDataType(DataBuffer.Type.FLOAT);
        shape2.setLength(50);
        shape2.setOffset(10);
        shape2.setStride(1);

        allocator.registerSpan(objectId, shape2);

        allocator.getDevicePointer(objectId, shape2);

        assertEquals(2, point.getNumberOfDescendants());

        assertEquals(1, point.getDescendantTicks(shape));
        assertEquals(1, point.getDescendantTicks(shape2));
    }
}
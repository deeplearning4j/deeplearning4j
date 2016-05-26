package org.nd4j.jita.flow.impl;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;

import static org.junit.Assert.*;

/**
 * This set of tests validates async flow controller behavior on atomic level
 *
 * @author raver119@gmail.com
 */
public class AsynchronousFlowControllerTest {
    private AtomicAllocator allocator;
    private AsynchronousFlowController controller;

    @Before
    public void setUp() throws Exception {
        CudaEnvironment.getInstance().getConfiguration()
                .setFirstMemory(AllocationStatus.DEVICE)
                .setExecutionModel(Configuration.ExecutionModel.ASYNCHRONOUS)
                .setAllocationModel(Configuration.AllocationModel.CACHE_ALL)
                .setMaximumSingleDeviceAllocation(1024 * 1024 * 1024L)
                .setMaximumBlockSize(128)
                .allowPreallocation(true)
                .setPreallocationCalls(20)
                .setMaximumGridSize(256)
                .enableDebug(false)
                .setVerbose(false);

        if (allocator == null)
            allocator = AtomicAllocator.getInstance();

        if (controller == null)
            controller = (AsynchronousFlowController) allocator.getFlowController();
    }

    @Test
    public void testDependencies1() throws Exception {


        INDArray array = Nd4j.create(new float[]{1f, 2f, 3f});

        // we use synchronization to make sure it completes activeWrite caused by array creation
        String arrayContents = array.toString();

        AllocationPoint point = allocator.getAllocationPoint(array);

        assertPointHasNoDependencies(point);
    }

    @Test
    public void testDependencies2() throws Exception {


        INDArray arrayWrite = Nd4j.create(new float[]{1f, 2f, 3f});
        INDArray array = Nd4j.create(new float[]{1f, 2f, 3f});

        // we use synchronization to make sure it completes activeWrite caused by array creation
        String arrayContents = array.toString();

        AllocationPoint point = allocator.getAllocationPoint(array);

        assertPointHasNoDependencies(point);

        CudaContext context = controller.prepareAction(arrayWrite, array);
        controller.registerAction(context, arrayWrite, array);

        assertTrue(controller.hasActiveReads(point));
        assertEquals(-1, controller.hasActiveWrite(point));
    }

    @Test
    public void testDependencies3() throws Exception {
        INDArray arrayWrite = Nd4j.create(new float[]{1f, 2f, 3f});
        INDArray array = Nd4j.create(new float[]{1f, 2f, 3f});

        // we use synchronization to make sure it completes activeWrite caused by array creation
        String arrayContents = array.toString();

        AllocationPoint point = allocator.getAllocationPoint(array);
        AllocationPoint pointWrite = allocator.getAllocationPoint(arrayWrite);

        assertPointHasNoDependencies(point);

        CudaContext context = controller.prepareAction(arrayWrite, array);
        controller.registerAction(context, arrayWrite, array);

        assertTrue(controller.hasActiveReads(point));
        assertFalse(controller.hasActiveReads(pointWrite));
        assertNotEquals(-1, controller.hasActiveWrite(pointWrite));

        controller.synchronizeReadLanes(point);

        assertPointHasNoDependencies(point);

        assertEquals(-1, controller.hasActiveWrite(pointWrite));
    }

    @Test
    public void testDependencies4() throws Exception {
        INDArray arrayWrite = Nd4j.create(new float[]{1f, 2f, 3f});
        INDArray array = Nd4j.create(new float[]{1f, 2f, 3f});

        // we use synchronization to make sure it completes activeWrite caused by array creation
        String arrayContents = array.toString();

        AllocationPoint point = allocator.getAllocationPoint(array);
        AllocationPoint pointWrite = allocator.getAllocationPoint(arrayWrite);

        assertPointHasNoDependencies(point);
        controller.cutTail();

        CudaContext context = controller.prepareAction(arrayWrite, array);
        controller.registerAction(context, arrayWrite, array);

        assertTrue(controller.hasActiveReads(point));
        assertFalse(controller.hasActiveReads(pointWrite));
        assertNotEquals(-1, controller.hasActiveWrite(pointWrite));

        Configuration configuration = CudaEnvironment.getInstance().getConfiguration();

        controller.sweepTail();

        assertTrue(controller.hasActiveReads(point));
        assertFalse(controller.hasActiveReads(pointWrite));
        assertNotEquals(-1, controller.hasActiveWrite(pointWrite));

        controller.sweepTail();

        assertTrue(controller.hasActiveReads(point));
        assertFalse(controller.hasActiveReads(pointWrite));
        assertNotEquals(-1, controller.hasActiveWrite(pointWrite));

        for (int i = 0; i < configuration.getCommandQueueLength(); i++)
            controller.sweepTail();

        assertPointHasNoDependencies(point);
        assertPointHasNoDependencies(pointWrite);
    }


    protected void assertPointHasNoDependencies(AllocationPoint point) {
        assertFalse(controller.hasActiveReads(point));
        assertEquals(-1, controller.hasActiveWrite(point));
    }
}
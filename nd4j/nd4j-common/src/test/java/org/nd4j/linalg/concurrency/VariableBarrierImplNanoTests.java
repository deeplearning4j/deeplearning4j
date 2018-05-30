package org.nd4j.linalg.concurrency;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;

import static org.junit.Assert.*;

@Slf4j
public class VariableBarrierImplNanoTests {

    @Test
    public void testPlanMapping_1() {
        val barrier = new VariableBarrierImpl(true);

        int[] array = {3, 3, 4};
        barrier.updatePlan(array);

        assertEquals(3, barrier.getConsumersForIteration(0));
        assertEquals(3, barrier.getConsumersForIteration(1));
        assertEquals(3, barrier.getConsumersForIteration(2));
        assertEquals(1, barrier.getConsumersForIteration(3));
        assertEquals(0, barrier.getConsumersForIteration(4));
        assertEquals(0, barrier.getConsumersForIteration(5));
        assertEquals(0, barrier.getConsumersForIteration(6));
    }


    @Test
    public void testPlanMapping_2() {
        val barrier = new VariableBarrierImpl(true);

        int[] array = {1, 2, 3, 4};
        barrier.updatePlan(array);

        assertEquals(4, barrier.getConsumersForIteration(0));
        assertEquals(3, barrier.getConsumersForIteration(1));
        assertEquals(2, barrier.getConsumersForIteration(2));
        assertEquals(1, barrier.getConsumersForIteration(3));
        assertEquals(0, barrier.getConsumersForIteration(4));
    }

    @Test
    public void testPlanMapping_3() {
        val barrier = new VariableBarrierImpl(true);

        int[] array = {2, 17};
        barrier.updatePlan(array);

        assertEquals(2, barrier.getConsumersForIteration(0));
        assertEquals(2, barrier.getConsumersForIteration(1));
        assertEquals(1, barrier.getConsumersForIteration(2));
        assertEquals(1, barrier.getConsumersForIteration(3));
        assertEquals(1, barrier.getConsumersForIteration(4));
        assertEquals(1, barrier.getConsumersForIteration(5));
        assertEquals(0, barrier.getConsumersForIteration(17));
        assertEquals(0, barrier.getConsumersForIteration(18));
    }


    @Test (timeout = 2000L)
    public void test_1() {
        val barrier = new VariableBarrierImpl(false);

        // we're officially on sync phase now
        barrier.setPhase(1);

        barrier.setConsumers(3);

        barrier.synchronizedBlock();

        assertEquals(1, barrier.getPhase());

        barrier.synchronizedBlock();

        assertEquals(1, barrier.getPhase());

        barrier.synchronizedBlock();

        // we have 3 customers, so this one was last one, and we're on 2nd phase now
        assertEquals(2, barrier.getPhase());
    }

    @Test (timeout = 2000L)
    public void test_2() {
        val barrier = new VariableBarrierImpl(false);

        // we're officially on desync phase now
        barrier.setPhase(2);

        barrier.setConsumers(3);

        barrier.desynchronizedBlock();

        assertEquals(2, barrier.getPhase());

        barrier.desynchronizedBlock();

        assertEquals(2, barrier.getPhase());

        barrier.desynchronizedBlock();

        // we have 3 customers, so this one was last one, and we're back to 0 phase now
        assertEquals(0, barrier.getPhase());
    }

    @Test (timeout = 2000L)
    public void test_Planned_1() {
        val barrier = new VariableBarrierImpl(true);

        barrier.registerConsumers(new int[]{3, 1});

        assertEquals(1, barrier.getPhase());

        assertEquals(1, barrier.getIteration());

        // at this iteration we should have 2 customers
        assertEquals(2, barrier.getConsumers());

        // first iteration for 2 workers
        barrier.synchronizedBlock();
        barrier.synchronizedBlock();

        // end of first iteration for 2 workers
        barrier.desynchronizedBlock();
        barrier.desynchronizedBlock();

        assertEquals(2, barrier.getIteration());

        // just 1 consumer left
        assertEquals(1, barrier.getConsumers());

        barrier.synchronizedBlock();
        barrier.desynchronizedBlock();

        assertEquals(3, barrier.getIteration());
        // still 1 consumer left
        assertEquals(1, barrier.getConsumers());

        barrier.synchronizedBlock();
        barrier.desynchronizedBlock();

        // now we must be on 0 phase
        assertEquals(0, barrier.getPhase());
    }
}
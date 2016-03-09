package org.nd4j.jita.allocator.concurrency;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.jita.allocator.enums.AccessState;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class AtomicStateTest {

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testRequestTick1() throws Exception {
        AtomicState ticker = new AtomicState();

        assertEquals(AccessState.TACK, ticker.getCurrentState());
    }


    @Test
    public void testRequestTick2() throws Exception {
        AtomicState ticker = new AtomicState();

        ticker.requestTick();

        assertEquals(AccessState.TICK, ticker.getCurrentState());

        ticker.requestTack();

        assertEquals(AccessState.TACK, ticker.getCurrentState());

        ticker.requestToe();
        assertEquals(AccessState.TOE, ticker.getCurrentState());

        ticker.releaseToe();
        assertEquals(AccessState.TACK, ticker.getCurrentState());
    }

    @Test
    public void testRequestTick3() throws Exception {
        AtomicState ticker = new AtomicState();

        ticker.requestTick();
        ticker.requestTick();
        assertEquals(AccessState.TICK, ticker.getCurrentState());

        ticker.requestTack();

        assertEquals(AccessState.TICK, ticker.getCurrentState());
        assertEquals(2, ticker.getTickRequests());
        assertEquals(1, ticker.getTackRequests());

        ticker.requestTack();
        assertEquals(AccessState.TACK, ticker.getCurrentState());

        assertEquals(0, ticker.getTickRequests());
        assertEquals(0, ticker.getTackRequests());
    }

    /**
     * This test addresses reentrance for Toe state
     *
     * @throws Exception
     */
    @Test
    public void testRequestTick4() throws Exception {
        AtomicState ticker = new AtomicState();

        ticker.requestTick();

        assertEquals(AccessState.TICK, ticker.getCurrentState());

        ticker.requestTack();

        assertEquals(AccessState.TACK, ticker.getCurrentState());

        ticker.requestToe();
        assertEquals(AccessState.TOE, ticker.getCurrentState());

        ticker.requestToe();
        assertEquals(AccessState.TOE, ticker.getCurrentState());

        ticker.releaseToe();
        assertEquals(AccessState.TOE, ticker.getCurrentState());

        ticker.releaseToe();
        assertEquals(AccessState.TACK, ticker.getCurrentState());
    }
}
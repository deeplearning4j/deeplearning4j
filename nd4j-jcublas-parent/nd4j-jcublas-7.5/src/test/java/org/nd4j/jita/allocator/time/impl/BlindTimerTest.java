package org.nd4j.jita.allocator.time.impl;

import org.junit.Before;
import org.junit.Test;

import java.util.concurrent.TimeUnit;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class BlindTimerTest {

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testIsAlive1() throws Exception {
        BlindTimer timer = new BlindTimer(2, TimeUnit.SECONDS);
        timer.triggerEvent();

        assertTrue(timer.isAlive());
    }

    @Test
    public void testIsAlive2() throws Exception {
        BlindTimer timer = new BlindTimer(2, TimeUnit.SECONDS);
        timer.triggerEvent();

        Thread.sleep(3000);

        assertFalse(timer.isAlive());
    }
}
package org.nd4j.jita.allocator.impl;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * This is test set for JITA, please assume test-driven development for ANY changes applied to JITA.
 *
 * @author raver119@gmail.com
 */
public class BasicAllocatorTest {


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

    /**
     * We inject some abstract object into allocator, and check if it gets released few seconds later as unused
     *
     */
    @Test
    public void testTickDecay1() {

    }
}
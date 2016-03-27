package org.nd4j.jita.allocator.utils;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.linalg.api.buffer.DataBuffer;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
@Ignore
public class AllocationUtilsTest {

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testGetRequiredMemory1() throws Exception {
        AllocationShape shape = new AllocationShape();
        shape.setOffset(0);
        shape.setLength(10);
        shape.setStride(1);
        shape.setDataType(DataBuffer.Type.DOUBLE);

        assertEquals(80, AllocationUtils.getRequiredMemory(shape));
    }

    @Test
    public void testGetRequiredMemory2() throws Exception {
        AllocationShape shape = new AllocationShape();
        shape.setOffset(0);
        shape.setLength(10);
        shape.setStride(1);
        shape.setDataType(DataBuffer.Type.FLOAT);

        assertEquals(40, AllocationUtils.getRequiredMemory(shape));
    }

    @Test
    public void testGetRequiredMemory3() throws Exception {
        AllocationShape shape = new AllocationShape();
        shape.setOffset(0);
        shape.setLength(10);
        shape.setStride(2);
        shape.setDataType(DataBuffer.Type.FLOAT);

        assertEquals(80, AllocationUtils.getRequiredMemory(shape));
    }
}
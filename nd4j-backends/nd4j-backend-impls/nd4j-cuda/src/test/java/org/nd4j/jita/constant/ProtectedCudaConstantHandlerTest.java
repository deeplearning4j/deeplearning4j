package org.nd4j.jita.constant;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class ProtectedCudaConstantHandlerTest {
    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testPurge1() throws Exception {
        DataBuffer buffer = Nd4j.getConstantHandler().getConstantBuffer(new float[]{1, 2, 3, 4, 5});

        ProtectedCudaConstantHandler handler = (ProtectedCudaConstantHandler) ((CudaConstantHandler)Nd4j.getConstantHandler()).wrappedHandler;

        assertEquals(1, handler.amountOfEntries(0));

        handler.purgeConstants();

        assertEquals(0, handler.amountOfEntries(0));
    }

}
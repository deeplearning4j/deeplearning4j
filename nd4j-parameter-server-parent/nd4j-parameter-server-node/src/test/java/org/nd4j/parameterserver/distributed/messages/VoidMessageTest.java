package org.nd4j.parameterserver.distributed.messages;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class VoidMessageTest {
    @Before
    public void setUp() throws Exception {

    }

    @After
    public void tearDown() throws Exception {

    }

    @Test
    public void testSerDe1() throws Exception {
        NegativeBatchMessage message = new NegativeBatchMessage();
        message.setW1(new int[]{10, 20, 30, 40});
        message.setW2(new int[]{15, 25, 35, 45});

        byte[] bytes = message.asBytes();

        NegativeBatchMessage restored = (NegativeBatchMessage) VoidMessage.fromBytes(bytes);

        assertNotEquals(null, restored);

        assertEquals(message, restored);
        assertArrayEquals(message.getW1(), restored.getW1());
        assertArrayEquals(message.getW2(), restored.getW2());
    }

}
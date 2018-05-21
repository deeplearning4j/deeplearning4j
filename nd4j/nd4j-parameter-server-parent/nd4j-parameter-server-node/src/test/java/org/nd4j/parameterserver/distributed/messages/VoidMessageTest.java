package org.nd4j.parameterserver.distributed.messages;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.parameterserver.distributed.messages.requests.SkipGramRequestMessage;

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
        SkipGramRequestMessage message = new SkipGramRequestMessage(10, 12, new int[] {10, 20, 30, 40},
                        new byte[] {(byte) 0, (byte) 0, (byte) 1, (byte) 0}, (short) 0, 0.0, 117L);

        byte[] bytes = message.asBytes();

        SkipGramRequestMessage restored = (SkipGramRequestMessage) VoidMessage.fromBytes(bytes);

        assertNotEquals(null, restored);

        assertEquals(message, restored);
        assertArrayEquals(message.getPoints(), restored.getPoints());
        assertArrayEquals(message.getCodes(), restored.getCodes());
    }

}

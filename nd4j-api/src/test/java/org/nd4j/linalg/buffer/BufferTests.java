package org.nd4j.linalg.buffer;

import static org.junit.Assert.*;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DoubleBuffer;
import org.nd4j.linalg.api.buffer.test.DataBufferTest;

/**
 * Created by agibsonccc on 10/11/14.
 */
public class BufferTests  {

    @Test
    public void testDoubleBuffer() {
        DataBuffer d = new DoubleBuffer(1);
        d.put(0,1);
        assertEquals(1,d.getDouble(0),1e-1);
    }

}

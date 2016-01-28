package org.nd4j.linalg.cpu.javacpp;

import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.cpu.CBLAS;
import org.nd4j.linalg.factory.Nd4j;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.concurrent.TimeUnit;
import static org.junit.Assert.*;
/**
 * @author Adam Gibson
 */
public class LoopTest {
    @Test
    public void testLoop() {
        ByteBuffer buff = ByteBuffer.allocateDirect(48);
        FloatBuffer buff2 = buff.asFloatBuffer();
        for(int i = 0;i < 4; i++)
            buff2.put(i,i);
        for(int i = 0; i < 4; i++)
            System.out.println(buff2.get(i));
        System.out.println(System.getProperty("java.library.path"));
        float sum = CBLAS.sasum(4,buff2,1);
        assertEquals(10,sum,1e-1);

    }


}

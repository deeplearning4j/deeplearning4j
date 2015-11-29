package org.nd4j.linalg.util;

import org.junit.Test;
import org.nd4j.org.nd4j.linalg.util.NioUtil;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;

import static org.junit.Assert.*;

/**
 * Created by agibsonccc on 11/29/15.
 */
public class NioUtilTests {
    public final static int ELEMENT_SIZE = 8;
    @Test
    public void testNioCopy() {
        int smallerBufferLength = 5;
        int biggerBufferLength = 10;
        ByteBuffer elevens = ByteBuffer.allocate(smallerBufferLength * ELEMENT_SIZE);
        DoubleBuffer buf2 = elevens.asDoubleBuffer();
        for(int i = 0; i < smallerBufferLength; i++)
            buf2.put(11.0);
        for(int i = 0; i < smallerBufferLength; i++) {
            assertEquals(11.0,buf2.get(i),1e-1);
        }
        ByteBuffer biggerBuffer = ByteBuffer.allocate(biggerBufferLength * ELEMENT_SIZE);
        DoubleBuffer buf3 = biggerBuffer.asDoubleBuffer();
        for(int i = 0; i < biggerBufferLength; i++) {
            buf3.put(i);
        }

        for(int i = 0; i < biggerBufferLength; i++) {
            assertEquals(i,buf3.get(i),1e-1);
        }

        double[] expected = new double[biggerBufferLength];
        for(int i = 0; i < expected.length; i++) {
            if(i % 2 == 0)
                expected[i] = 11.0;
            else
                expected[i] = i;
        }

        NioUtil.copyAtStride(5, NioUtil.BufferType.DOUBLE, elevens,0,1,biggerBuffer,0,2);
        for(int i = 0; i < expected.length; i++) {
            assertEquals(expected[i],buf3.get(i),1e-1);
        }

    }

}

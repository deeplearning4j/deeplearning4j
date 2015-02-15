package org.nd4j.linalg.api.buffer.test;


import static org.junit.Assert.*;

import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by agibsonccc on 2/14/15.
 */
public abstract class DataBufferTest {

    @Test
    public void testGetSet() {
        double[] d1 = new double[]{1,2,3,4};
        DataBuffer d = Nd4j.createBuffer(d1);
        double[] d2 = d.asDouble();
        assertArrayEquals(d1,d2,1e-1);
        d.destroy();

    }


}

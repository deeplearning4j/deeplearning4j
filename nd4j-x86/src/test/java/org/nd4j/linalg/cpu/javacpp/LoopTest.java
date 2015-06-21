package org.nd4j.linalg.cpu.javacpp;

import org.junit.Test;

/**
 * @author Adam Gibson
 */
public class LoopTest {
    @Test
    public void testLoop() {
        Loop loop = new Loop();
        float[] add1 = new float[3];
        float[] add2 = new float[1];
        loop.execFloatTransform(add1,3,0,1,"exp",add2);
    }


}

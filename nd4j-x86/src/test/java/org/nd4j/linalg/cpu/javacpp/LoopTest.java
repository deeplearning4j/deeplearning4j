package org.nd4j.linalg.cpu.javacpp;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.concurrent.TimeUnit;

/**
 * @author Adam Gibson
 */
public class LoopTest {
    @Test
    public void testLoop() {
        Loop loop = new Loop();
        float[] add1 = new float[1000000];
        float[] add2 = new float[1];
        long start = System.nanoTime();
        loop.execFloatTransform(add1,3,0,0,1,1,"exp",add2,add1);
        long end = System.nanoTime();
        System.out.println((TimeUnit.MILLISECONDS.convert(Math.abs(end - start),TimeUnit.NANOSECONDS)));
        loop.execScalarFloat(add1,add1,add1.length,0,0,1,1,"div_scalar",new float[]{1});

    }

    @Test
    public void testCreate() {
        INDArray arr = Nd4j.create(new double[10]);
    }

    @Test
    public void testShape() {
        INDArray arr = Nd4j.create(new int[]{10});
        System.out.println(Arrays.toString(arr.shape()));
    }


    @Test
    public void testDup() {
        INDArray arr = Nd4j.ones(3);
        System.out.println(arr.dup());
    }

}

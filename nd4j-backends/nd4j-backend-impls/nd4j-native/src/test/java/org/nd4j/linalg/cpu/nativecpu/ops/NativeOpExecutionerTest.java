package org.nd4j.linalg.cpu.nativecpu.ops;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Exp;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
@Ignore
public class NativeOpExecutionerTest {
    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void execBroadcastOp() throws Exception {
        INDArray array = Nd4j.ones(1024, 1024);
        INDArray arrayRow = Nd4j.linspace(1, 1024, 1024);

        float sum = (float) array.sumNumber().doubleValue();

        array.addiRowVector(arrayRow);

        long time1 = System.nanoTime();
        for (int x = 0; x < 1000; x++) {
            array.addiRowVector(arrayRow);
        }
        long time2 = System.nanoTime();

        System.out.println("Execution time: " + ((time2 - time1) / 1000));

        assertEquals(1002, array.getFloat(0), 0.1f);
        assertEquals(2003, array.getFloat(1), 0.1f);
    }

    @Test
    public void execReduceOp1() throws Exception {
        INDArray array = Nd4j.ones(1024, 1024);
        INDArray arrayRow1 = Nd4j.linspace(1, 1024, 1024);
        INDArray arrayRow2 = Nd4j.linspace(0, 1023, 1024);

        float sum = (float) array.sumNumber().doubleValue();
        long time1 = System.nanoTime();
        sum = array.sumNumber().floatValue();
        long time2 = System.nanoTime();

        System.out.println("Execution time: " + (time2 - time1));
        System.out.println("Result: " + sum);
    }

    @Test
    public void execReduceOp2() throws Exception {
        INDArray array = Nd4j.ones(3, 1024);
        INDArray arrayRow = Nd4j.linspace(1, 1024, 1024);

        float sum = array.sumNumber().floatValue();
        long time1 = System.nanoTime();
        sum = array.sumNumber().floatValue();
        long time2 = System.nanoTime();

        System.out.println("Execution time: " + (time2 - time1));
    }

    @Test
    public void execTransformOp1() throws Exception {
        INDArray array1 = Nd4j.linspace(1, 2048, 2048);
        INDArray array2 = Nd4j.linspace(1, 2048, 2048);

        Nd4j.getExecutioner().exec(new Exp(array1, array2));

        long time1 = System.nanoTime();
        for (int x = 0; x < 10000; x++) {
            Nd4j.getExecutioner().exec(new Exp(array1, array2));
        }
        long time2 = System.nanoTime();

        System.out.println("Execution time: " + ((time2 - time1) / 10000));

       // System.out.println("Array1: " + array1);
       // System.out.println("Array2: " + array2);

        assertEquals(2.71f, array2.getFloat(0), 0.01);
    }


    @Test
    public void testScalarOp1() throws Exception {
        // simple way to stop test if we're not on CUDA backend here

        INDArray array1 = Nd4j.linspace(1, 20480, 20480);
        INDArray array2 = Nd4j.linspace(1, 20480, 20480);
        array2.addi(0.5f);

        long time1 = System.nanoTime();
        for (int x = 0; x < 10000; x++) {
            array2.addi(0.5f);
        }
        long time2 = System.nanoTime();

        System.out.println("Execution time: " + ((time2 - time1) / 10000));

        System.out.println("Divi result: " + array2.getFloat(0));
        assertEquals(5001.5, array2.getFloat(0), 0.01f);

    }
}
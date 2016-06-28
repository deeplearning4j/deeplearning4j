package org.nd4j.jita.memory.impl;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * This set of tests targets special concurrent environments, like Spark.
 * In this kind of environments, data pointers might be travelling across different threads
 *
 * PLEASE NOTE: This set of tests worth running on multi-gpu systems only. Running them on single-gpu system, will just show "PASSED" for everything.
 *
 * @author raver119@gmail.com
 */
@Ignore
public class WeirdSparkTests {

    @Before
    public void setUp() {
        CudaEnvironment.getInstance().getConfiguration().setAllocationModel(Configuration.AllocationModel.CACHE_ALL);
    }

    @Test
    public void aestMultithreadedTup1() throws Exception {
        final INDArray array1 = Nd4j.create(new float[]{1f, 2f, 3f, 4f, 5f});

        float sum = array1.sumNumber().floatValue();
        assertEquals(15f, sum, 0.001f);

        sum = array1.sumNumber().floatValue();
        assertEquals(15f, sum, 0.001f);

        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                System.out.println("--------------------------------------------");
                System.out.println("           External thread started");
                array1.putScalar(0, 0f);
                float sum = array1.sumNumber().floatValue();
                assertEquals(14f, sum, 0.001f);
            }
        });

        Nd4j.getAffinityManager().attachThreadToDevice(thread, 1);
        thread.start();
        thread.join();

        System.out.println("--------------------------------------------");
        System.out.println("            Back to main thread");

        sum = array1.sumNumber().floatValue();
        assertEquals(14f, sum, 0.001f);
    }

    @Test
    public void testMultithreadedDup1() throws Exception {
        final INDArray array1 = Nd4j.create(new float[]{1f, 2f, 3f, 4f, 5f});

     //   float sum = array1.sumNumber().floatValue();
    //    assertEquals(15f, sum, 0.001f);

        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                System.out.println("--------------------------------------------");
                System.out.println("           External thread started");
                INDArray array = array1.dup();

                float sum = array.sumNumber().floatValue();
                assertEquals(15f, sum, 0.001f);
            }
        });

        Nd4j.getAffinityManager().attachThreadToDevice(thread, 1);
        thread.start();
        thread.join();

        float sum = array1.sumNumber().floatValue();
        assertEquals(15f, sum, 0.001f);
    }
}


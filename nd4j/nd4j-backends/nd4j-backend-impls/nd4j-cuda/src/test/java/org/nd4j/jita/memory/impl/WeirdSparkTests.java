package org.nd4j.jita.memory.impl;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

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
        CudaEnvironment.getInstance().getConfiguration()
                .enableDebug(false)
                .setVerbose(false)
                .allowPreallocation(false)
                .setAllocationModel(Configuration.AllocationModel.CACHE_ALL)
                .setMemoryModel(Configuration.MemoryModel.IMMEDIATE);
    }

    @Test
    public void testMultithreaded1() throws Exception {
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

       float sum = array1.sumNumber().floatValue();
        assertEquals(15f, sum, 0.001f);

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

        sum = array1.sumNumber().floatValue();
        assertEquals(15f, sum, 0.001f);
    }

    @Test
    public void testMultithreadedDup2() throws Exception {
        final INDArray array1 = Nd4j.create(new float[]{1f, 2f, 3f, 4f, 5f});

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

    @Test
    @Ignore
    public void testMultithreadedFree1() throws Exception {
        final DataBuffer buffer = Nd4j.createBuffer(500000000,0);

        Thread.sleep(5000);

        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                System.out.println("Current device: " + AtomicAllocator.getInstance().getDeviceId());
                AllocationPoint point = AtomicAllocator.getInstance().getAllocationPoint(buffer);
                AtomicAllocator.getInstance().getMemoryHandler().getMemoryProvider().free(point);

                System.out.println("Pointer released");
                try {
                    Thread.sleep(100000);
                } catch (Exception e) {

                }
            }
        });

        thread.start();
        thread.join();
    }


    @Test
    public void testMultithreadedViews1() throws Exception {
        final INDArray array = Nd4j.ones(10,10);
        final INDArray view = array.getRow(1);

        assertEquals(1.0f, view.getFloat(0), 0.01f);

        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                assertEquals(1.0f, view.getFloat(0), 0.01f);

                view.subi(1.0f);

                try {
                    Thread.sleep(100);
                } catch (Exception e) {
                    //
                }

                System.out.println(view);
            }
        });

        Nd4j.getAffinityManager().attachThreadToDevice(thread, 1);
        thread.start();
        thread.join();

        //System.out.println(view);
        assertEquals(0.0f, view.getFloat(0), 0.01f);
    }


    @Test
    public void testMultithreadedRandom1() throws Exception{
    //    for (int i = 0; i < 5; i++) {
     //       System.out.println("Starting iteration " + i);
            final List<INDArray> holder = new ArrayList<>();
            final AtomicLong failures = new AtomicLong(0);

            Thread thread = new Thread(new Runnable() {
                @Override
                public void run() {
                    holder.add(Nd4j.ones(10));
                }
            });


            Nd4j.getAffinityManager().attachThreadToDevice(thread, 1);
            thread.start();
            thread.join();

            Thread[] threads = new Thread[100];
            for (int x = 0; x < threads.length; x++) {
                threads[x] = new Thread(new Runnable() {
                    @Override
                    public void run() {
                        try {
                            INDArray array = holder.get(0).dup();

                          //  ((CudaGridExecutioner) Nd4j.getExecutioner()).flushQueueBlocking();
                        } catch (Exception e) {
                            failures.incrementAndGet();
                            throw new RuntimeException(e);
                        }
                    }
                });

                threads[x].start();
            }

            for (int x = 0; x < threads.length; x++) {
                threads[x].join();
            }

            assertEquals(0, failures.get());
   //     }
    }
}


package jcuda.jcublas.ops;

import org.apache.commons.math3.util.Pair;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

/**
 * @author raver119@gmail.com
 */
public class AveragingTests {
    private final int THREADS = 16;
    private final int LENGTH = 512000 * 4;


    @Before
    public void setUp() {
        CudaEnvironment.getInstance().getConfiguration()
                .allowMultiGPU(true)
                .allowCrossDeviceAccess(true)
                .enableDebug(true)
                .setMaximumGridSize(512)
                .setMaximumBlockSize(256)
                .setVerbose(true);
    }


    @Test
    public void testSingleDeviceAveraging() throws Exception {
        INDArray array1 = Nd4j.valueArrayOf(500, 1.0);
        INDArray array2 = Nd4j.valueArrayOf(500, 2.0);
        INDArray array3 = Nd4j.valueArrayOf(500, 3.0);

        INDArray arrayMean = Nd4j.averageAndPropagate(array1, array2, array3);


        assertNotEquals(null, arrayMean);

        assertEquals(2.0f, arrayMean.getFloat(12), 0.1f);
        assertEquals(2.0f, arrayMean.getFloat(150), 0.1f);
        assertEquals(2.0f, arrayMean.getFloat(475), 0.1f);


        assertEquals(2.0f, array1.getFloat(475), 0.1f);
        assertEquals(2.0f, array2.getFloat(475), 0.1f);
        assertEquals(2.0f, array3.getFloat(475), 0.1f);
    }


    /**
     * This test should be run on multi-gpu system only. On single-gpu system this test will fail
     * @throws Exception
     */
    @Test
    public void testMultiDeviceAveraging() throws Exception {
        final List<Pair<INDArray, INDArray>> pairs = new ArrayList<>();

        int numDevices = CudaEnvironment.getInstance().getConfiguration().getAvailableDevices().size();
        AtomicAllocator allocator = AtomicAllocator.getInstance();


        for (int i = 0; i < THREADS; i++) {
            final int order = i;
            Thread thread = new Thread(new Runnable() {
                @Override
                public void run() {
                    pairs.add(new Pair<INDArray, INDArray>(Nd4j.valueArrayOf(LENGTH, (double) order), null));

                    try {
                        Thread.sleep(100);
                    } catch (Exception e) {
                        //
                    }
                }
            });

            thread.start();
            thread.join();
        }

        assertEquals(THREADS, pairs.size());
        final List<INDArray> arrays = new ArrayList<>();

        AtomicBoolean hasNonZero = new AtomicBoolean(false);

        for (int i = 0; i < THREADS; i++) {
            INDArray array = pairs.get(i).getKey();
            AllocationPoint point = allocator.getAllocationPoint(array.data());

            if (point.getDeviceId().intValue() != 0 )
                hasNonZero.set(true);

            arrays.add(array);
        }

        assertEquals(true, hasNonZero.get());

/*
        // old way of averaging, without further propagation
        INDArray z = Nd4j.create(LENGTH);
        long time1 = System.currentTimeMillis();
        for (int i = 0; i < THREADS; i++) {
            z.addi(arrays.get(i));
        }
        z.divi((float) THREADS);
        CudaContext context = (CudaContext) allocator.getDeviceContext().getContext();
        context.syncOldStream();
        long time2 = System.currentTimeMillis();
        System.out.println("Execution time: " + (time2 - time1));

*/

        long time1 = System.currentTimeMillis();
        INDArray z = Nd4j.averageAndPropagate(arrays);
        long time2 = System.currentTimeMillis();
        System.out.println("Execution time: " + (time2 - time1));


        assertEquals(7.5f, z.getFloat(0), 0.01f);
        assertEquals(7.5f, z.getFloat(10), 0.01f);

        for (int i = 0; i < THREADS; i++) {

            for (int x = 0; x < LENGTH; x++) {
                assertEquals("Failed on array [" +i+ "], element [" +x+ "]",z.getFloat(0), arrays.get(i).getFloat(x), 0.01f);
            }
        }



    }

}

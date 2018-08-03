/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package jcuda.jcublas.ops;

import org.nd4j.linalg.primitives.Pair;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT);
        CudaEnvironment.getInstance().getConfiguration()
                .allowMultiGPU(true)
                .allowCrossDeviceAccess(true)
                .enableDebug(true)
                .setMaximumGridSize(512)
                .setMaximumBlockSize(256)
                .setVerbose(true);
    }

    @Test
    @Ignore
    public void testReshape() {
        INDArray a = Nd4j.linspace(0,1000 , 100000000).reshape(1000, 1000, 100).permutei(0, 2, 1);

        long startDup = System.nanoTime();
        INDArray a2 = a.dup().reshape(500, 2000, 100);
        System.out.println(String.format("Dup time: %.3f s", (System.nanoTime() - startDup) / 1000000000.));

        int[] newShape = new int[]{500, 2000, 100};
        long startTime = System.nanoTime();
        INDArray b = Nd4j.createUninitialized(newShape, 'c').assign(a);
        System.out.println(String.format("Assign: %.3f s", (System.nanoTime() - startTime) / 1000000000.));

        assertEquals(a2, b);
    }


    @Test
    public void testSingleDeviceAveraging() throws Exception {
        INDArray array1 = Nd4j.valueArrayOf(LENGTH, 1.0);
        INDArray array2 = Nd4j.valueArrayOf(LENGTH, 2.0);
        INDArray array3 = Nd4j.valueArrayOf(LENGTH, 3.0);
        INDArray array4 = Nd4j.valueArrayOf(LENGTH, 4.0);
        INDArray array5 = Nd4j.valueArrayOf(LENGTH, 5.0);
        INDArray array6 = Nd4j.valueArrayOf(LENGTH, 6.0);
        INDArray array7 = Nd4j.valueArrayOf(LENGTH, 7.0);
        INDArray array8 = Nd4j.valueArrayOf(LENGTH, 8.0);
        INDArray array9 = Nd4j.valueArrayOf(LENGTH, 9.0);
        INDArray array10 = Nd4j.valueArrayOf(LENGTH, 10.0);
        INDArray array11 = Nd4j.valueArrayOf(LENGTH, 11.0);
        INDArray array12 = Nd4j.valueArrayOf(LENGTH, 12.0);
        INDArray array13 = Nd4j.valueArrayOf(LENGTH, 13.0);
        INDArray array14 = Nd4j.valueArrayOf(LENGTH, 14.0);
        INDArray array15 = Nd4j.valueArrayOf(LENGTH, 15.0);
        INDArray array16 = Nd4j.valueArrayOf(LENGTH, 16.0);


        long time1 = System.currentTimeMillis();
        INDArray arrayMean = Nd4j.averageAndPropagate(new INDArray[] {array1, array2, array3, array4, array5, array6, array7, array8, array9, array10, array11, array12, array13, array14, array15, array16});
        long time2 = System.currentTimeMillis();
        System.out.println("Execution time: " + (time2 - time1));

        assertNotEquals(null, arrayMean);

        assertEquals(8.5f, arrayMean.getFloat(12), 0.1f);
        assertEquals(8.5f, arrayMean.getFloat(150), 0.1f);
        assertEquals(8.5f, arrayMean.getFloat(475), 0.1f);


        assertEquals(8.5f, array1.getFloat(475), 0.1f);
        assertEquals(8.5f, array2.getFloat(475), 0.1f);
        assertEquals(8.5f, array3.getFloat(475), 0.1f);
        assertEquals(8.5f, array5.getFloat(475), 0.1f);
        assertEquals(8.5f, array16.getFloat(475), 0.1f);
    }


    /**
     * This test should be run on multi-gpu system only. On single-gpu system this test will fail
     * @throws Exception
     */
    @Test
    public void testMultiDeviceAveraging() throws Exception {
        final List<Pair<INDArray, INDArray>> pairs = new ArrayList<>();

        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
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

            if (point.getDeviceId() != 0 )
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

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

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
@Ignore
public class CudaBroadcastTests {

    @Before
    public void setUp() {
        CudaEnvironment.getInstance().getConfiguration()
                .setExecutionModel(Configuration.ExecutionModel.SEQUENTIAL)
                .setFirstMemory(AllocationStatus.DEVICE)
                .setMaximumBlockSize(64)
                .setMaximumGridSize(128)
                .enableDebug(true);

        System.out.println("Init called");
    }

    @Test
    public void testPinnedAddiRowVector() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        for (int iter = 0; iter < 100; iter++) {

            INDArray array1 = Nd4j.zeros(15, 15);

            for (int y = 0; y < 15; y++) {
                for (int x = 0; x < 15; x++) {
                    assertEquals("Failed on iteration: ["+iter+"], y.x: ["+y+"."+x+"]", 0.0f, array1.getRow(y).getFloat(x), 0.01);
                }
            }
            INDArray array2 = Nd4j.create(new float[]{2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f});

            for (int i = 0; i < 30; i++) {
                array1.addiRowVector(array2);
            }

            //System.out.println("Array1: " + array1);
            //System.out.println("Array2: " + array2);

            for (int y = 0; y < 15; y++) {
                for (int x = 0; x < 15; x++) {
                    assertEquals("Failed on iteration: ["+iter+"], y.x: ["+y+"."+x+"]", 60.0f, array1.getRow(y).getFloat(x), 0.01);
                }
            }
        }
    }

    @Test
    public void testPinnedSubiRowVector() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        INDArray array1 = Nd4j.zeros(1500,150);
        INDArray array2 = Nd4j.linspace(1,150,150);

        AtomicAllocator.getInstance().getPointer(array1, (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext());
        AtomicAllocator.getInstance().getPointer(array2, (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext());

        long time1 = System.currentTimeMillis();
        array1.subiRowVector(array2);
        long time2 = System.currentTimeMillis();

        System.out.println("Execution time: " + (time2 - time1));

     //   System.out.println("Array1: " + array1);
//        System.out.println("Array2: " + array2);

        assertEquals(-1.0f, array1.getRow(0).getFloat(0), 0.01);
        assertEquals(-3.0f, array1.getRow(0).getFloat(2), 0.01);
        assertEquals(-10.0f, array1.getRow(0).getFloat(9), 0.01);
    }

    @Test
    public void testPinnedSubiColumnVector2() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        INDArray array1 = Nd4j.zeros(1500,150);
        INDArray array2 = Nd4j.linspace(1,1500,1500).reshape(1500,1);

        array1.subiColumnVector(array2);

//        System.out.println("Array1: " + array1);
//        System.out.println("Array2: " + array2);

        assertEquals(-1.0f, array1.getRow(0).getFloat(0), 0.01);
        assertEquals(-1.0f, array1.getRow(0).getFloat(0), 0.01);
        assertEquals(-301.0f, array1.getRow(300).getFloat(0), 0.01);
        assertEquals(-1500.0f, array1.getRow(1499).getFloat(0), 0.01);
    }

    @Test
    public void testPinnedSubiRowVector2() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        INDArray array1 = Nd4j.zeros(1500,150);
        INDArray array2 = Nd4j.linspace(1,1500,1500).reshape(1500,1);

        array1.subiRowVector(array2);

        System.out.println("Array1: " + array1.shapeInfoDataBuffer());
        System.out.println("Array2: " + array2.shapeInfoDataBuffer());

        assertEquals(-1.0f, array1.getRow(0).getFloat(0), 0.01);
        assertEquals(-1.0f, array1.getRow(0).getFloat(0), 0.01);
        assertEquals(-301.0f, array1.getRow(300).getFloat(0), 0.01);
        assertEquals(-1500.0f, array1.getRow(1499).getFloat(0), 0.01);
    }

    @Test
    public void testPinnedRSubiRowVector() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(15,15);
        INDArray array2 = Nd4j.create(new float[]{2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f});

        array1.rsubiRowVector(array2);

        System.out.println("Array1: " + array1);
        //System.out.println("Array2: " + array2);

        assertEquals(2.0f, array1.getRow(0).getFloat(0), 0.01);
        assertEquals(2.0f, array1.getRow(1).getFloat(0), 0.01);
        assertEquals(2.0f, array1.getRow(3).getFloat(3), 0.01);
    }

    @Test
    public void testPinnedSubiColumnVector() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        INDArray array1 = Nd4j.zeros(150,3);
        INDArray array2 = Nd4j.linspace(1, 150, 150).reshape(150,1);

        array1.subiColumnVector(array2);

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(-1.0f, array1.getRow(0).getFloat(0), 0.01);
        assertEquals(-2.0f, array1.getRow(1).getFloat(0), 0.01);
        assertEquals(-3.0f, array1.getRow(2).getFloat(0), 0.01);
    }

    @Test
    public void testPinnedMulRowVector() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.zeros(15,15);
        array1.putRow(0, Nd4j.create(new float[]{2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f}));
        array1.putRow(1, Nd4j.create(new float[]{2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f}));
        INDArray array2 = Nd4j.create(new float[]{2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f});

        array1.muliRowVector(array2);

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(4.0f, array1.getRow(0).getFloat(0), 0.01);
    }

    @Test
    public void testPinnedDivRowVector() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.zeros(15,15);
        array1.putRow(0, Nd4j.create(new float[]{2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f}));
        array1.putRow(1, Nd4j.create(new float[]{2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f}));
        INDArray array2 = Nd4j.create(new float[]{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

        array1.diviRowVector(array2);

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(2.0f, array1.getRow(0).getFloat(0), 0.01);
    }

    @Test
    public void testPinnedRDivRowVector() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.zeros(15,15);
        array1.putRow(0, Nd4j.create(new float[]{2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f}));
        array1.putRow(1, Nd4j.create(new float[]{2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f}));
        INDArray array2 = Nd4j.create(new float[]{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

        array1.rdiviRowVector(array2);

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(0.5f, array1.getRow(0).getFloat(0), 0.01);
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
    public void execBroadcastOpTimed2() throws Exception {
        Nd4j.create(1);
        System.out.println("A ----------------");
        INDArray array = Nd4j.zeros(2048, 1024);
        System.out.println("0 ----------------");
        INDArray arrayRow = Nd4j.ones(1024);

        System.out.println("1 ----------------");

        float sum = (float) array.sumNumber().doubleValue();
        float sum2 = (float) arrayRow.sumNumber().doubleValue();

        System.out.println("2 ----------------");

        long time1 = System.nanoTime();
        for (int x = 0; x < 1000; x++) {
            array.addiRowVector(arrayRow);
        }
        long time2 = System.nanoTime();

        System.out.println("Execution time: " + ((time2 - time1) / 1000));

        for (int x = 0; x < array.rows(); x++) {
            INDArray row = array.getRow(x);
            for (int y = 0; y < array.columns(); y++) {
                assertEquals("Failed on x.y: ["+x+"."+y+"]",1000f, row.getFloat(y), 0.01);
            }
        }
    }
}


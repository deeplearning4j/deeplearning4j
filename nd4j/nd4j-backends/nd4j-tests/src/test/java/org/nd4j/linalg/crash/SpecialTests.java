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

package org.nd4j.linalg.crash;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.lang3.RandomUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Broadcast;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;

import static org.junit.Assert.*;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

/**
 * @author raver119@gmail.com
 */
@Slf4j
@RunWith(Parameterized.class)
public class SpecialTests extends BaseNd4jTest {
    public SpecialTests(Nd4jBackend backend) {
        super(backend);
    }


    @Test
    public void testDimensionalThings1() {
        INDArray x = Nd4j.rand(new int[] {20, 30, 50});
        INDArray y = Nd4j.rand(x.shape());

        INDArray result = transform(x, y);
    }

    @Test
    public void testDimensionalThings2() {
        INDArray x = Nd4j.rand(new int[] {20, 30, 50});
        INDArray y = Nd4j.rand(x.shape());


        for (int i = 0; i < 1; i++) {
            int number = 5;
            int start = RandomUtils.nextInt(0, (int) x.shape()[2] - number);

            transform(getView(x, start, 5), getView(y, start, 5));
        }
    }

    protected static INDArray getView(INDArray x, int from, int number) {
        return x.get(all(), all(), interval(from, from + number));
    }

    protected static INDArray transform(INDArray a, INDArray b) {
        int nShape[] = new int[] {1, 2};
        INDArray a_reduced = a.sum(nShape);
        INDArray b_reduced = b.sum(nShape);

        //log.info("reduced shape: {}", Arrays.toString(a_reduced.shapeInfoDataBuffer().asInt()));

        return Transforms.abs(a_reduced.sub(b_reduced)).div(a_reduced);
    }


    @Test(expected = ND4JIllegalStateException.class)
    public void testScalarShuffle1() throws Exception {
        List<DataSet> listData = new ArrayList<>();
        for (int i = 0; i < 3; i++) {
            INDArray features = Nd4j.ones(25, 25);
            INDArray label = Nd4j.create(new float[] {1}, new int[] {1});
            DataSet dataset = new DataSet(features, label);
            listData.add(dataset);
        }
        DataSet data = DataSet.merge(listData);
        data.shuffle();
    }


    @Test
    public void testScalarShuffle2() throws Exception {
        List<DataSet> listData = new ArrayList<>();
        for (int i = 0; i < 3; i++) {
            INDArray features = Nd4j.ones(14, 25);
            INDArray label = Nd4j.create(14, 50);
            DataSet dataset = new DataSet(features, label);
            listData.add(dataset);
        }
        DataSet data = DataSet.merge(listData);
        data.shuffle();
    }

    @Test
    public void testVstack2() throws Exception {
        INDArray matrix = Nd4j.create(10000, 100);

        List<INDArray> views = new ArrayList<>();
        views.add(matrix.getRow(1));
        views.add(matrix.getRow(4));
        views.add(matrix.getRow(7));

        INDArray result = Nd4j.vstack(views);
    }

    @Test
    public void testVstack1() throws Exception {
        INDArray matrix = Nd4j.create(10000, 100);

        List<INDArray> views = new ArrayList<>();
        for (int i = 0; i < matrix.rows() / 2; i++) {
            views.add(matrix.getRow(RandomUtils.nextInt(0, (int) matrix.rows())));
            //views.add(Nd4j.create(1, 10));
        }

        log.info("Starting...");

        //while (true) {
        for (int i = 0; i < 1; i++) {
            INDArray result = Nd4j.vstack(views);

            System.gc();
        }
    }

    @Test
    public void testConcatMulti() throws Exception {
        val shapeA = new int[] {50, 20};
        val shapeB = new int[] {50, 497};

        //Nd4j.create(1);

        val executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(2);

        for (int e = 0; e < 1; e++) {
            executor.submit(new Runnable() {
                @Override
                public void run() {
                    val arrayA = Nd4j.createUninitialized(shapeA);
                }
            });
        }

        Thread.sleep(1000);
    }

    @Test
    public void testConcatMulti2() throws Exception {
        Nd4j.create(1);
        val executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(2);
        executor.submit(new Runnable() {
            @Override
            public void run() {
                System.out.println("A");
            }
        });
    }

    @Test
    public void testMigrationMultiGpu_1() throws Exception {
        if (Nd4j.getAffinityManager().getNumberOfDevices() < 2)
            return;

        val list = new CopyOnWriteArrayList<INDArray>();
        val threads = new ArrayList<Thread>();
        for (int e = 0; e< Nd4j.getAffinityManager().getNumberOfDevices(); e++) {
            val f = e;
            val t = new Thread(new Runnable() {
                @Override
                public void run() {
                    for (int i = 0; i < 10; i++) {
                        list.add(Nd4j.create(100, 100).assign(1.0f));
                        Nd4j.getExecutioner().commit();
                    }
                }
            });

            t.start();
            threads.add(t);
        }

        for (val t:threads)
            t.join();

        for (val a:list)
            assertEquals(1.0f, a.meanNumber().floatValue(), 1e-5);
    }

    @Test
    public void testMigrationMultiGpu_2() throws Exception {
        if (Nd4j.getAffinityManager().getNumberOfDevices() < 2)
            return;

        val wsConf = WorkspaceConfiguration.builder()
                .policyReset(ResetPolicy.ENDOFBUFFER_REACHED)
                .policyAllocation(AllocationPolicy.STRICT)
                .initialSize(50 * 1024L * 1024L)
                .build();

        for (int x = 0; x < 10; x++) {

            val list = new CopyOnWriteArrayList<INDArray>();
            val threads = new ArrayList<Thread>();
            for (int e = 0; e < Nd4j.getAffinityManager().getNumberOfDevices(); e++) {
                val f = e;
                val t = new Thread(new Runnable() {
                    @Override
                    public void run() {
                        for (int i = 0; i < 100; i++) {
                            try (val ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(wsConf, "id")) {
                                list.add(Nd4j.create(3, 3).assign(1.0f));
                                Nd4j.getExecutioner().commit();
                            }
                        }
                    }
                });

                t.start();
                threads.add(t);
            }

            for (val t : threads)
                t.join();

            for (val a : list) {
                assertTrue(a.isAttached());
                assertEquals(1.0f, a.meanNumber().floatValue(), 1e-5);
            }

            System.gc();
        }
    }

    @Test
    public void testBroadcastLt(){
        for( int i=0; i<10; i++) {

            INDArray x = Nd4j.create(DataType.DOUBLE, 1, 3, 2, 4, 4);
            INDArray y = Nd4j.create(DataType.DOUBLE, 1, 2, 4, 4);
            INDArray z = Nd4j.create(DataType.BOOL, 1, 3, 2, 4, 4);
            Broadcast.lt(x, y, z, 0, 2, 3, 4);

        }
    }

    @Test
    public void testBroadcastLt2(){
        for( int i=0; i<10; i++) {
            INDArray orig = Nd4j.create(DataType.DOUBLE, 1, 7, 4, 4);
            INDArray y = orig.get(all(), interval(0,2), all(), all());

            INDArray x = Nd4j.create(DataType.DOUBLE, 1, 3, 2, 4, 4);
            INDArray z = Nd4j.create(DataType.BOOL, 1, 3, 2, 4, 4);
            Broadcast.lt(x, y, z, 0, 2, 3, 4);

        }
    }

    @Test
    public void reproduceWorkspaceCrash(){
        val conf = WorkspaceConfiguration.builder().build();

        val ws = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(conf, "WS");

        INDArray arr = Nd4j.create(new double[]{1, 0, 0, 0, 1, 0, 0, 0, 0, 0}, new long[]{1, 10});

        //assertNotEquals(Nd4j.defaultFloatingPointType(), arr.dataType());
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);

        for( int i=0; i<100; i++ ) {
            try(val ws2 = ws.notifyScopeEntered()) {
                System.out.println("Iteration: " + i);
                INDArray ok = arr.eq(0.0);
                ok.dup();

                assertEquals(arr.dataType(), Nd4j.defaultFloatingPointType());
                assertEquals(DataType.DOUBLE, Nd4j.defaultFloatingPointType());
                INDArray crash = arr.eq(0.0).castTo(Nd4j.defaultFloatingPointType());
                crash.dup();        //Crashes here on i=1 iteration
            }
        }
    }

    @Test
    public void reproduceWorkspaceCrash_2(){
        val dtypes = new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF, DataType.LONG, DataType.INT, DataType.SHORT, DataType.BYTE, DataType.UBYTE, DataType.BOOL};
        for (val dX : dtypes) {
            for (val dZ: dtypes) {
                val array = Nd4j.create(dX, 2, 5).assign(1);

                log.info("Trying to cast {} to {}", dX, dZ);
                val casted = array.castTo(dZ);

                val exp = Nd4j.create(dZ, 2, 5).assign(1);
                assertEquals(exp, casted);
            }
        }
    }

    @Test
    public void reproduceWorkspaceCrash_3(){
        val conf = WorkspaceConfiguration.builder().build();

        val ws = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(conf, "WS");
        val dtypes = new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF, DataType.LONG, DataType.INT, DataType.SHORT, DataType.BYTE, DataType.UBYTE, DataType.BOOL};
        for (val dX : dtypes) {
            for (val dZ: dtypes) {
                try(val ws2 = ws.notifyScopeEntered()) {
                    val array = Nd4j.create(dX, 2, 5).assign(1);
                    log.info("Trying to cast {} to {}", dX, dZ);
                    val casted = array.castTo(dZ);
                    val exp = Nd4j.create(dZ, 2, 5).assign(1);
                    assertEquals(exp, casted);

                    Nd4j.getExecutioner().commit();
                }
            }
        }
    }

    @Test
    public void testCastLong_1() {
        val array = Nd4j.create(DataType.LONG, 100, 100).assign(1);
        val second = Nd4j.create(DataType.LONG, 100, 100).assign(1);
        log.info("----------------");
        val castedA = array.castTo(DataType.BYTE).assign(3);
        val castedB = array.castTo(DataType.BYTE).assign(3);
        Nd4j.getExecutioner().commit();
        assertEquals(castedA, castedB);

        assertEquals(array, second);
    }

    @Test
    public void testCastHalf_1() throws Exception {
        val array = Nd4j.create(DataType.HALF, 2, 5).assign(1);
        assertEquals(10.f, array.sumNumber().floatValue(), 1e-3);
    }

    @Test
    public void testCastHalf_2() throws Exception {
        val array = Nd4j.create(DataType.HALF, 2, 5).assign(1);
        assertEquals(10.f, array.sumNumber().floatValue(), 1e-3);
    }

    @Test
    public void testCastHalf_3() throws Exception {
        val arrayY = Nd4j.create(DataType.FLOAT, 2, 5).assign(2);
        val arrayX = Nd4j.create(DataType.HALF, 2, 5).assign(arrayY);
        assertEquals(20.f, arrayX.sumNumber().floatValue(), 1e-3);
    }

    @Test
    public void testReduce_Small_1() {
        val array = Nd4j.create(DataType.SHORT, 100, 30).assign(1);
        assertEquals(3000, array.sumNumber().intValue());
    }

    @Test
    public void testReduce_Small_2() {
        val array = Nd4j.create(DataType.BYTE, 100, 100).assign(0);
        assertEquals(0, array.sumNumber().intValue());
    }

    @Test
    public void testReduce3_Small_1() {
        val arrayA = Nd4j.create(DataType.SHORT, 100, 100).assign(1);
        val arrayB = Nd4j.create(DataType.SHORT, 100, 100).assign(1);
        assertEquals(arrayA, arrayB);
    }

    @Test
    public void testReduce3_Small_2() {
        val arrayA = Nd4j.create(DataType.BYTE, 100, 100).assign(1);
        val arrayB = Nd4j.create(DataType.BYTE, 100, 100).assign(1);
        assertEquals(arrayA, arrayB);
    }

    @Test
    public void reproduceWorkspaceCrash_4(){
        val conf = WorkspaceConfiguration.builder().build();

        val ws = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(conf, "WS");
        val dtypes = new DataType[]{DataType.LONG, DataType.DOUBLE, DataType.FLOAT, DataType.HALF, DataType.INT, DataType.SHORT, DataType.BYTE, DataType.UBYTE, DataType.BOOL};
        for (val dX : dtypes) {
            for (val dZ: dtypes) {
                try(val ws2 = Nd4j.getWorkspaceManager().getAndActivateWorkspace("WS")) {
                    val array = Nd4j.create(dX, 100, 100).assign(1);

                    log.info("Trying to cast {} to {}", dX, dZ);
                    val casted = array.castTo(dZ);

                    val exp = Nd4j.create(dZ, 100, 100).assign(1);
                    assertEquals(exp, casted);
                }
            }
        }
    }

    @Test
    public void reproduceWorkspaceCrash_5(){
        val conf = WorkspaceConfiguration.builder().build();

        val ws = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(conf, "WS");

        INDArray arr = Nd4j.create(new double[]{1, 0, 0, 0, 1, 0, 0, 0, 0, 0}, new long[]{1, 10});

        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
        assertEquals(DataType.DOUBLE, arr.dataType());

        for( int i=0; i<100; i++ ) {
            try(val ws2 = ws.notifyScopeEntered()) {
                INDArray crash = arr.castTo(DataType.BOOL).castTo(DataType.DOUBLE);
                crash.dup();
            }
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}

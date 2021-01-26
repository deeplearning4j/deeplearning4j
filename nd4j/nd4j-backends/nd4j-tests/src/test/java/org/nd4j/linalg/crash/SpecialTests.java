/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.crash;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import lombok.var;
import org.apache.commons.lang3.RandomUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.reduce.longer.MatchCondition;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Broadcast;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.nd4j.linalg.indexing.NDArrayIndex.*;

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
    public void testScalarShuffle1() {
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
    public void testScalarShuffle2() {
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
    public void testVstack2() {
        INDArray matrix = Nd4j.create(10000, 100);

        List<INDArray> views = new ArrayList<>();
        views.add(matrix.getRow(1));
        views.add(matrix.getRow(4));
        views.add(matrix.getRow(7));

        INDArray result = Nd4j.vstack(views);
    }

    @Test
    public void testVstack1() {
        INDArray matrix = Nd4j.create(10000, 100);

        List<INDArray> views = new ArrayList<>();
        for (int i = 0; i < matrix.rows() / 2; i++) {
            views.add(matrix.getRow(RandomUtils.nextInt(0, matrix.rows())));
            //views.add(Nd4j.create(1, 10));
        }

//        log.info("Starting...");

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
    public void testConcatMulti2() {
        Nd4j.create(1);
        val executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(2);
        executor.submit(new Runnable() {
            @Override
            public void run() {
//                System.out.println("A");
            }
        });
    }

    @Test
    public void testMigrationMultiGpu_1() throws Exception {
        if (Nd4j.getAffinityManager().getNumberOfDevices() < 2)
            return;

        val list = new CopyOnWriteArrayList<INDArray>();
        val threads = new ArrayList<Thread>();
        val devices = Nd4j.getAffinityManager().getNumberOfDevices();
        for (int e = 0; e < devices; e++) {
            val f = e;
            val t = new Thread(new Runnable() {
                @Override
                public void run() {
                    val deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();
                    log.info("Current device: {}", deviceId);
                    for (int i = 0; i < 10; i++) {
                        val ar = Nd4j.create(100, 100).assign(1.0f);

                        assertEquals(deviceId, Nd4j.getAffinityManager().getDeviceForArray(ar));
                        list.add(ar);
                        Nd4j.getExecutioner().commit();
                    }
                }
            });

            t.start();
            t.join();
            threads.add(t);

//            log.info("------------------------");
        }

        for (val t:threads)
            t.join();

        for (val a:list) {
            val device = Nd4j.getAffinityManager().getDeviceForArray(a);
            try {
                assertEquals(1.0f, a.meanNumber().floatValue(), 1e-5);
            } catch (Exception e) {
                log.error("Failed for array from device [{}]", device);
                throw e;
            }
        }
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
//                System.out.println("Iteration: " + i);
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

//                log.info("Trying to cast {} to {}", dX, dZ);
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
//                    log.info("Trying to cast {} to {}", dX, dZ);
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
//        log.info("----------------");
        val castedA = array.castTo(DataType.BYTE).assign(3);
        val castedB = array.castTo(DataType.BYTE).assign(3);
        Nd4j.getExecutioner().commit();
        assertEquals(castedA, castedB);

        assertEquals(array, second);
    }

    @Test
    public void testCastHalf_1() {
        val array = Nd4j.create(DataType.HALF, 2, 5).assign(1);
        assertEquals(10.f, array.sumNumber().floatValue(), 1e-3);
    }

    @Test
    public void testCastHalf_2() {
        val array = Nd4j.create(DataType.HALF, 2, 5).assign(1);
        assertEquals(10.f, array.sumNumber().floatValue(), 1e-3);
    }

    @Test
    public void testCastHalf_3() {
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

//                    log.info("Trying to cast {} to {}", dX, dZ);
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

    @Test
    public void testConcatAgain(){
        INDArray[] toConcat = new INDArray[3];
        for( int i=0; i<toConcat.length; i++ ) {
            toConcat[i] = Nd4j.valueArrayOf(new long[]{10, 1}, i).castTo(DataType.FLOAT);
        }

        INDArray out = Nd4j.concat(1, toConcat);
//        System.out.println(out);
    }

    @Test
    public void testConcat2(){
        //Nd4j.getExecutioner().enableDebugMode(true);
        //Nd4j.getExecutioner().enableVerboseMode(true);
        int n = 784;  //OK for 10, 100, 500
        //Fails for 784, 783, 750, 720, 701, 700

        INDArray[] arrs = new INDArray[n];
        for( int i=0; i<n; i++ ){
            INDArray a = Nd4j.create(DataType.DOUBLE, 10,1).assign(i);      //Also fails for FLOAT
            arrs[i] = a;
        }

        Nd4j.getExecutioner().commit();
        INDArray out = null;
        for (int e = 0; e < 5; e++) {
            if (e % 10 == 0)
//                log.info("Iteration: [{}]", e);

            out = Nd4j.concat(1, arrs);
        }
        Nd4j.getExecutioner().commit();
//        System.out.println(out);
    }

    @Test
    public void testYoloStyle(){
        WorkspaceConfiguration WS_ALL_LAYERS_ACT_CONFIG = WorkspaceConfiguration.builder()
                .initialSize(0)
                .overallocationLimit(0.05)
                .policyLearning(LearningPolicy.FIRST_LOOP)
                .policyReset(ResetPolicy.BLOCK_LEFT)
                .policySpill(SpillPolicy.REALLOCATE)
                .policyAllocation(AllocationPolicy.OVERALLOCATE)
                .build();



        for( int i=0; i<10; i++ ){
            try(val ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(WS_ALL_LAYERS_ACT_CONFIG, "ws")){
//                System.out.println("STARTING: " + i);

                INDArray objectPresentMask = Nd4j.create(DataType.BOOL, 1,4,4);

                long[] shape = {1,3,2,4,4};
                INDArray noIntMask1 = Nd4j.createUninitialized(DataType.BOOL, shape, 'c');
                INDArray noIntMask2 = Nd4j.createUninitialized(DataType.BOOL, shape, 'c');

                noIntMask1 = Transforms.or(noIntMask1.get(all(), all(), point(0), all(), all()), noIntMask1.get(all(), all(), point(1), all(), all()) );    //Shape: [mb, b, H, W]. Values 1 if no intersection
                noIntMask2 = Transforms.or(noIntMask2.get(all(), all(), point(0), all(), all()), noIntMask2.get(all(), all(), point(1), all(), all()) );
                INDArray noIntMask = Transforms.or(noIntMask1, noIntMask2 );

                Nd4j.getExecutioner().commit();

                INDArray intMask = Transforms.not(noIntMask); //Values 0 if no intersection
                Nd4j.getExecutioner().commit();

                Broadcast.mul(intMask, objectPresentMask, intMask, 0, 2, 3);
                Nd4j.getExecutioner().commit();
//                System.out.println("DONE: " + i);
            }
        }
    }


    @Test
    public void testSpaceToBatch() {
        Nd4j.getRandom().setSeed(7331);

        int miniBatch = 4;
        int[] inputShape = new int[]{1, 2, 2, 1};

        int M = 2;

        INDArray input = Nd4j.randn(inputShape).castTo(DataType.DOUBLE);
        INDArray blocks = Nd4j.createFromArray(2, 2);
        INDArray padding = Nd4j.createFromArray(0, 0, 0, 0).reshape(2,2);

        INDArray expOut = Nd4j.create(DataType.DOUBLE, miniBatch, 1, 1, 1);
        val op = DynamicCustomOp.builder("space_to_batch_nd")
                .addInputs(input, blocks, padding)
                .addOutputs(expOut).build();
        Nd4j.getExecutioner().execAndReturn(op);
    }

    @Test
    public void testBatchToSpace() {
        Nd4j.getRandom().setSeed(1337);

        int miniBatch = 4;
        int[] inputShape = new int[]{miniBatch, 1, 1, 1};

        int M = 2;

        INDArray input = Nd4j.randn(inputShape).castTo(DataType.DOUBLE);
        INDArray blocks = Nd4j.createFromArray(2, 2);
        INDArray crops = Nd4j.createFromArray(0, 0, 0, 0).reshape(2,2);

        INDArray expOut = Nd4j.create(DataType.DOUBLE, 1, 2, 2, 1);
        DynamicCustomOp op = DynamicCustomOp.builder("batch_to_space_nd")
                .addInputs(input, blocks, crops)
                .addOutputs(expOut).build();
        Nd4j.getExecutioner().execAndReturn(op);
    }

    @Test
    public void testYoloS(){
        //Nd4j.getExecutioner().enableDebugMode(true);
        //Nd4j.getExecutioner().enableVerboseMode(true);
        //Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);

        WorkspaceConfiguration WS_ALL_LAYERS_ACT_CONFIG = WorkspaceConfiguration.builder()
                .initialSize(10 * 1024 * 1024)
                .overallocationLimit(0.05)
                .policyLearning(LearningPolicy.FIRST_LOOP)
                .policyReset(ResetPolicy.BLOCK_LEFT)
                .policySpill(SpillPolicy.REALLOCATE)
                .policyAllocation(AllocationPolicy.OVERALLOCATE)
                .build();


        INDArray labels = Nd4j.create(DataType.DOUBLE, 1,7,5,7);

        for( int i=0; i<10; i++ ){
            try(val ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(WS_ALL_LAYERS_ACT_CONFIG, "ws")){
//                System.out.println("STARTING: " + i);

                val nhw = new long[]{1, 5, 7};

                val size1 = labels.size(1);
                INDArray classLabels = labels.get(all(), interval(4,size1), all(), all());   //Shape: [minibatch, nClasses, H, W]
                INDArray maskObjectPresent = classLabels.sum(Nd4j.createUninitialized(DataType.DOUBLE, nhw, 'c'), 1).castTo(DataType.BOOL); //Shape: [minibatch, H, W]

                INDArray labelTLXY = labels.get(all(), interval(0,2), all(), all());
                INDArray labelBRXY = labels.get(all(), interval(2,4), all(), all());

                Nd4j.getExecutioner().commit();

                INDArray labelCenterXY = labelTLXY.add(labelBRXY);
                val m = labelCenterXY.muli(0.5);  //In terms of grid units
                INDArray labelsCenterXYInGridBox = labelCenterXY.dup(labelCenterXY.ordering());         //[mb, 2, H, W]
                Nd4j.getExecutioner().commit();
//                System.out.println("DONE: " + i);
            }
        }
    }

    @Test
    public void testMatchCondition(){
        INDArray x = Nd4j.valueArrayOf(new long[]{10,10}, 2.0, DataType.DOUBLE);
        val op = new MatchCondition(x, Conditions.equals(2));
        INDArray z = Nd4j.getExecutioner().exec(op);
        int count = z.getInt(0);
        assertEquals(100, count);
    }

    @Test
    public void testBroadcastMul_bool() {
        val mask = Nd4j.create(DataType.BOOL, 1, 3, 4, 4);
        val object = Nd4j.create(DataType.BOOL, 1, 4, 4);

        Broadcast.mul(mask, object, mask, 0, 2, 3);
        Nd4j.getExecutioner().commit();
    }

    @Test
    public void testReshape(){
        INDArray c = Nd4j.linspace(1,6,6, DataType.DOUBLE).reshape('c', 2,3);
        INDArray f = c.dup('f');
        val fr = f.reshape('f', 3, 2).dup('f');

//        log.info("FO: {}", f.data().asFloat());
//        log.info("FR: {}", fr.data().asFloat());

        INDArray outC = Nd4j.create(DataType.DOUBLE, 3,2);
        INDArray outF = Nd4j.create(DataType.DOUBLE, 3,2);

        var op = DynamicCustomOp.builder("reshape")
                .addInputs(c)
                .addOutputs(outC)
                .addIntegerArguments(3,2)
                .build();

        Nd4j.getExecutioner().exec(op);

        op = DynamicCustomOp.builder("reshape")
                .addInputs(f)
                .addOutputs(outF)
                .addIntegerArguments(-99, 3,2)
                .build();

        Nd4j.getExecutioner().exec(op);

        assertEquals(outC, outF);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}

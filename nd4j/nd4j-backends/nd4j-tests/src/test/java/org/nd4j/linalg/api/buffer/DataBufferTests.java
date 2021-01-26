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

package org.nd4j.linalg.api.buffer;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.nativeblas.NativeOpsHolder;


import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.Assert.*;

@Slf4j
@RunWith(Parameterized.class)
public class DataBufferTests extends BaseNd4jTest {

    public DataBufferTests(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    @Ignore("AB 2019/06/03 - CI issue: \"CUDA stream synchronization failed\" - see issue 7657")
    public void testNoArgCreateBufferFromArray() {

        //Tests here:
        //1. Create from JVM array
        //2. Create from JVM array with offset -> does this even make sense?
        //3. Create detached buffer

        WorkspaceConfiguration initialConfig = WorkspaceConfiguration.builder().initialSize(10 * 1024L * 1024L)
                .policyAllocation(AllocationPolicy.STRICT).policyLearning(LearningPolicy.NONE).build();
        MemoryWorkspace workspace = Nd4j.getWorkspaceManager().createNewWorkspace(initialConfig, "WorkspaceId");

        for (boolean useWs : new boolean[]{false, true}) {

            try (MemoryWorkspace ws = (useWs ? workspace.notifyScopeEntered() : null)) {

                //Float
                DataBuffer f = Nd4j.createBuffer(new float[]{1, 2, 3});
                checkTypes(DataType.FLOAT, f, 3);
                assertEquals(useWs, f.isAttached());
                testDBOps(f);

                f = Nd4j.createBuffer(new float[]{1, 2, 3}, 0);
                checkTypes(DataType.FLOAT, f, 3);
                assertEquals(useWs, f.isAttached());
                testDBOps(f);

                f = Nd4j.createBufferDetached(new float[]{1, 2, 3});
                checkTypes(DataType.FLOAT, f, 3);
                assertFalse(f.isAttached());
                testDBOps(f);

                //Double
                DataBuffer d = Nd4j.createBuffer(new double[]{1, 2, 3});
                checkTypes(DataType.DOUBLE, d, 3);
                assertEquals(useWs, d.isAttached());
                testDBOps(d);

                d = Nd4j.createBuffer(new double[]{1, 2, 3}, 0);
                checkTypes(DataType.DOUBLE, d, 3);
                assertEquals(useWs, d.isAttached());
                testDBOps(d);

                d = Nd4j.createBufferDetached(new double[]{1, 2, 3});
                checkTypes(DataType.DOUBLE, d, 3);
                assertFalse(d.isAttached());
                testDBOps(d);

                //Int
                DataBuffer i = Nd4j.createBuffer(new int[]{1, 2, 3});
                checkTypes(DataType.INT, i, 3);
                assertEquals(useWs, i.isAttached());
                testDBOps(i);

                i = Nd4j.createBuffer(new int[]{1, 2, 3});
                checkTypes(DataType.INT, i, 3);
                assertEquals(useWs, i.isAttached());
                testDBOps(i);

                i = Nd4j.createBufferDetached(new int[]{1, 2, 3});
                checkTypes(DataType.INT, i, 3);
                assertFalse(i.isAttached());
                testDBOps(i);

                //Long
                DataBuffer l = Nd4j.createBuffer(new long[]{1, 2, 3});
                checkTypes(DataType.LONG, l, 3);
                assertEquals(useWs, l.isAttached());
                testDBOps(l);

                l = Nd4j.createBuffer(new long[]{1, 2, 3});
                checkTypes(DataType.LONG, l, 3);
                assertEquals(useWs, l.isAttached());
                testDBOps(l);

                l = Nd4j.createBufferDetached(new long[]{1, 2, 3});
                checkTypes(DataType.LONG, l, 3);
                assertFalse(l.isAttached());
                testDBOps(l);

                //byte
//                DataBuffer b = Nd4j.createBuffer(new byte[]{1, 2, 3});
//                checkTypes(DataType.BYTE, b, 3);
//                testDBOps(b);
//
//                b = Nd4j.createBuffer(new byte[]{1, 2, 3}, 0);
//                checkTypes(DataType.BYTE, b, 3);
//                testDBOps(b);
//
//                b = Nd4j.createBufferDetached(new byte[]{1,2,3});
//                checkTypes(DataType.BYTE, b, 3);
//                testDBOps(b);

                //short
                //TODO
            }
        }
    }

    protected static void checkTypes(DataType dataType, DataBuffer db, long expLength) {
        assertEquals(dataType, db.dataType());
        assertEquals(expLength, db.length());
        switch (dataType) {
            case DOUBLE:
                assertTrue(db.pointer() instanceof DoublePointer);
                assertTrue(db.indexer() instanceof DoubleIndexer);
                break;
            case FLOAT:
                assertTrue(db.pointer() instanceof FloatPointer);
                assertTrue(db.indexer() instanceof FloatIndexer);
                break;
            case HALF:
                assertTrue(db.pointer() instanceof ShortPointer);
                assertTrue(db.indexer() instanceof HalfIndexer);
                break;
            case LONG:
                assertTrue(db.pointer() instanceof LongPointer);
                assertTrue(db.indexer() instanceof LongIndexer);
                break;
            case INT:
                assertTrue(db.pointer() instanceof IntPointer);
                assertTrue(db.indexer() instanceof IntIndexer);
                break;
            case SHORT:
                assertTrue(db.pointer() instanceof ShortPointer);
                assertTrue(db.indexer() instanceof ShortIndexer);
                break;
            case UBYTE:
                assertTrue(db.pointer() instanceof BytePointer);
                assertTrue(db.indexer() instanceof UByteIndexer);
                break;
            case BYTE:
                assertTrue(db.pointer() instanceof BytePointer);
                assertTrue(db.indexer() instanceof ByteIndexer);
                break;
            case BOOL:
                //Bool type uses byte pointers
                assertTrue(db.pointer() instanceof BooleanPointer);
                assertTrue(db.indexer() instanceof BooleanIndexer);
                break;
        }
    }

    protected static void testDBOps(DataBuffer db) {
        for (int i = 0; i < 3; i++) {
            if (db.dataType() != DataType.BOOL)
                testGet(db, i, i + 1);
            else
                testGet(db, i, 1);
        }
        testGetRange(db);
        testAsArray(db);

        if (db.dataType() != DataType.BOOL)
            testAssign(db);
    }

    protected static void testGet(DataBuffer from, int idx, Number exp) {
        assertEquals(exp.doubleValue(), from.getDouble(idx), 0.0);
        assertEquals(exp.floatValue(), from.getFloat(idx), 0.0f);
        assertEquals(exp.intValue(), from.getInt(idx));
        assertEquals(exp.longValue(), from.getLong(idx), 0.0f);
    }

    protected static void testGetRange(DataBuffer from) {
        if (from.dataType() != DataType.BOOL) {
            assertArrayEquals(new double[]{1, 2, 3}, from.getDoublesAt(0, 3), 0.0);
            assertArrayEquals(new double[]{1, 3}, from.getDoublesAt(0, 2, 2), 0.0);
            assertArrayEquals(new double[]{2, 3}, from.getDoublesAt(1, 1, 2), 0.0);
            assertArrayEquals(new float[]{1, 2, 3}, from.getFloatsAt(0, 3), 0.0f);
            assertArrayEquals(new float[]{1, 3}, from.getFloatsAt(0, 2, 2), 0.0f);
            assertArrayEquals(new float[]{2, 3}, from.getFloatsAt(1, 1, 3), 0.0f);
            assertArrayEquals(new int[]{1, 2, 3}, from.getIntsAt(0, 3));
            assertArrayEquals(new int[]{1, 3}, from.getIntsAt(0, 2, 2));
            assertArrayEquals(new int[]{2, 3}, from.getIntsAt(1, 1, 3));
        } else {
            assertArrayEquals(new double[]{1, 1, 1}, from.getDoublesAt(0, 3), 0.0);
            assertArrayEquals(new double[]{1, 1}, from.getDoublesAt(0, 2, 2), 0.0);
            assertArrayEquals(new double[]{1, 1}, from.getDoublesAt(1, 1, 2), 0.0);
            assertArrayEquals(new float[]{1, 1, 1}, from.getFloatsAt(0, 3), 0.0f);
            assertArrayEquals(new float[]{1, 1}, from.getFloatsAt(0, 2, 2), 0.0f);
            assertArrayEquals(new float[]{1, 1}, from.getFloatsAt(1, 1, 3), 0.0f);
            assertArrayEquals(new int[]{1, 1, 1}, from.getIntsAt(0, 3));
            assertArrayEquals(new int[]{1, 1}, from.getIntsAt(0, 2, 2));
            assertArrayEquals(new int[]{1, 1}, from.getIntsAt(1, 1, 3));
        }
    }

    protected static void testAsArray(DataBuffer db) {
        if (db.dataType() != DataType.BOOL) {
            assertArrayEquals(new double[]{1, 2, 3}, db.asDouble(), 0.0);
            assertArrayEquals(new float[]{1, 2, 3}, db.asFloat(), 0.0f);
            assertArrayEquals(new int[]{1, 2, 3}, db.asInt());
            assertArrayEquals(new long[]{1, 2, 3}, db.asLong());
        } else {
            assertArrayEquals(new double[]{1, 1, 1}, db.asDouble(), 0.0);
            assertArrayEquals(new float[]{1, 1, 1}, db.asFloat(), 0.0f);
            assertArrayEquals(new int[]{1, 1, 1}, db.asInt());
            assertArrayEquals(new long[]{1, 1, 1}, db.asLong());
        }
    }

    protected static void testAssign(DataBuffer db) {
        db.assign(5.0);
        testGet(db, 0, 5.0);
        testGet(db, 2, 5.0);

        if (db.dataType().isSigned()) {
            db.assign(-3.0f);
            testGet(db, 0, -3.0);
            testGet(db, 2, -3.0);
        }

        db.assign(new long[]{0, 1, 2}, new float[]{10, 9, 8}, true);
        testGet(db, 0, 10);
        testGet(db, 1, 9);
        testGet(db, 2, 8);

        db.assign(new long[]{0, 2}, new float[]{7, 6}, false);
        testGet(db, 0, 7);
        testGet(db, 1, 9);
        testGet(db, 2, 6);
    }



    @Test
    public void testCreateTypedBuffer() {

        WorkspaceConfiguration initialConfig = WorkspaceConfiguration.builder().initialSize(10 * 1024L * 1024L)
                .policyAllocation(AllocationPolicy.STRICT).policyLearning(LearningPolicy.NONE).build();
        MemoryWorkspace workspace = Nd4j.getWorkspaceManager().createNewWorkspace(initialConfig, "WorkspaceId");

        for (String sourceType : new String[]{"int", "long", "float", "double", "short", "byte", "boolean"}) {
            for (DataType dt : DataType.values()) {
                if (dt == DataType.UTF8 || dt == DataType.COMPRESSED || dt == DataType.UNKNOWN) {
                    continue;
                }

//                log.info("Testing source [{}]; target: [{}]", sourceType, dt);

                for (boolean useWs : new boolean[]{false, true}) {

                    try (MemoryWorkspace ws = (useWs ? workspace.notifyScopeEntered() : null)) {

                        DataBuffer db1;
                        DataBuffer db2;
                        switch (sourceType) {
                            case "int":
                                db1 = Nd4j.createTypedBuffer(new int[]{1, 2, 3}, dt);
                                db2 = Nd4j.createTypedBufferDetached(new int[]{1, 2, 3}, dt);
                                break;
                            case "long":
                                db1 = Nd4j.createTypedBuffer(new long[]{1, 2, 3}, dt);
                                db2 = Nd4j.createTypedBufferDetached(new long[]{1, 2, 3}, dt);
                                break;
                            case "float":
                                db1 = Nd4j.createTypedBuffer(new float[]{1, 2, 3}, dt);
                                db2 = Nd4j.createTypedBufferDetached(new float[]{1, 2, 3}, dt);
                                break;
                            case "double":
                                db1 = Nd4j.createTypedBuffer(new double[]{1, 2, 3}, dt);
                                db2 = Nd4j.createTypedBufferDetached(new double[]{1, 2, 3}, dt);
                                break;
                            case "short":
                                db1 = Nd4j.createTypedBuffer(new short[]{1, 2, 3}, dt);
                                db2 = Nd4j.createTypedBufferDetached(new short[]{1, 2, 3}, dt);
                                break;
                            case "byte":
                                db1 = Nd4j.createTypedBuffer(new byte[]{1, 2, 3}, dt);
                                db2 = Nd4j.createTypedBufferDetached(new byte[]{1, 2, 3}, dt);
                                break;
                            case "boolean":
                                db1 = Nd4j.createTypedBuffer(new boolean[]{true, false, true}, dt);
                                db2 = Nd4j.createTypedBufferDetached(new boolean[]{true, false, true}, dt);
                                break;
                            default:
                                throw new RuntimeException();
                        }

                        checkTypes(dt, db1, 3);
                        checkTypes(dt, db2, 3);

                        assertEquals(useWs, db1.isAttached());
                        assertFalse(db2.isAttached());

                        if(!sourceType.equals("boolean")){
                            testDBOps(db1);
                            testDBOps(db2);
                        }
                    }
                }
            }
        }
    }

    @Test
    public void testAsBytes() {
        INDArray orig = Nd4j.linspace(DataType.INT, 0, 10, 1);

        for (DataType dt : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF, DataType.BFLOAT16,
                DataType.LONG, DataType.INT, DataType.SHORT, DataType.BYTE, DataType.BOOL,
                DataType.UINT64, DataType.UINT32, DataType.UINT16, DataType.UBYTE}) {
            INDArray arr = orig.castTo(dt);

            byte[] b = arr.data().asBytes();        //NOTE: BIG ENDIAN

            if(ByteOrder.nativeOrder().equals(ByteOrder.LITTLE_ENDIAN)) {
                //Switch from big endian (as defined by asBytes which uses big endian) to little endian
                int w = dt.width();
                if (w > 1) {
                    int len = b.length / w;
                    for (int i = 0; i < len; i++) {
                        for (int j = 0; j < w / 2; j++) {
                            byte temp = b[(i + 1) * w - j - 1];
                            b[(i + 1) * w - j - 1] = b[i * w + j];
                            b[i * w + j] = temp;
                        }
                    }
                }
            }

            INDArray arr2 = Nd4j.create(dt, arr.shape());
            ByteBuffer bb = arr2.data().pointer().asByteBuffer();
            bb.position(0);
            bb.put(b);

            Nd4j.getAffinityManager().tagLocation(arr2, AffinityManager.Location.HOST);

            assertEquals(arr.toString(), arr2.toString());
            assertEquals(arr, arr2);

            //Sanity check on data buffer getters:
            DataBuffer db = arr.data();
            DataBuffer db2 = arr2.data();
            for( int i=0; i<10; i++ ){
                assertEquals(db.getDouble(i), db2.getDouble(i), 0);
                assertEquals(db.getFloat(i), db2.getFloat(i), 0);
                assertEquals(db.getInt(i), db2.getInt(i), 0);
                assertEquals(db.getLong(i), db2.getLong(i), 0);
                assertEquals(db.getNumber(i), db2.getNumber(i));
            }

            assertArrayEquals(db.getDoublesAt(0, 10), db2.getDoublesAt(0, 10), 0);
            assertArrayEquals(db.getFloatsAt(0, 10), db2.getFloatsAt(0, 10), 0);
            assertArrayEquals(db.getIntsAt(0, 10), db2.getIntsAt(0, 10));
            assertArrayEquals(db.getLongsAt(0, 10), db2.getLongsAt(0, 10));
        }
    }


    @Test
    public void testEnsureLocation(){
        //https://github.com/eclipse/deeplearning4j/issues/8783
        Nd4j.create(1);

        BytePointer bp = new BytePointer(5);


        Pointer ptr = NativeOpsHolder.getInstance().getDeviceNativeOps().pointerForAddress(bp.address());
        DataBuffer buff = Nd4j.createBuffer(ptr, 5, DataType.INT8);


        INDArray arr2 = Nd4j.create(buff, new long[]{5}, new long[]{1}, 0, 'c', DataType.INT8);
        long before = arr2.data().pointer().address();
        Nd4j.getAffinityManager().ensureLocation(arr2, AffinityManager.Location.HOST);
        long after = arr2.data().pointer().address();

        assertEquals(before, after);
    }


    @Override
    public char ordering() {
        return 'c';
    }

}

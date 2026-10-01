/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.nd4j.linalg.api.buffer;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.util.ArrayTypeConverters;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.custom.BitCast;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.nativeblas.NativeOpsHolder;


import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.*;

@Slf4j
@NativeTag
public class DataBufferTests extends BaseNd4jTestWithBackends {


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNoArgCreateBufferFromArray(Nd4jBackend backend) {

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
        assertEquals(exp.doubleValue(), from.getDouble(idx), 0.0,"Whole data buffer: " + from + " expected: " + exp);
        assertEquals(exp.floatValue(), from.getFloat(idx), 0.0f,"Whole data buffer: " + from + " expected: " + exp);
        assertEquals(exp.intValue(), from.getInt(idx),"Whole data buffer: " + from + " expected: " + exp);
        assertEquals(exp.longValue(), from.getLong(idx), 0.0f,"Whole data buffer: " + from + " expected: " + exp);
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



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCreateTypedBuffer(Nd4jBackend backend) {

        WorkspaceConfiguration initialConfig = WorkspaceConfiguration.builder().initialSize(10 * 1024L * 1024L)
                .policyAllocation(AllocationPolicy.STRICT).policyLearning(LearningPolicy.NONE).build();
        MemoryWorkspace workspace = Nd4j.getWorkspaceManager().createNewWorkspace(initialConfig, "WorkspaceId");

        for (String sourceType : new String[]{"int", "long", "float", "double", "short", "byte", "boolean"}) {
            for (DataType dt : DataType.values()) {
                if (dt == DataType.UTF8 || dt == DataType.UTF16 || dt == DataType.UTF32 || dt == DataType.COMPRESSED || dt == DataType.UNKNOWN || dt == DataType.FLOAT8 || dt == DataType.FLOAT8_E5M2) {
                    continue;
                }

                /*
                TODO: look in to data inconsistency issues. Possible sources:
                1. deallocation race conditions
                2. data sync between cpu/gpu
                3. buffer reuse causing 1 to be a problem?
                4. could be workspace related leading to something like 3? (not confirmed and very unlikely from other testing)
                5.  Data seems to be very random.
                6. Short of this also run comparisons on cpu based tests isolating certain passing cpu but failing gpu tests to run to see
                if data loading is failing there.
                7. Likely things to look in to would be the allocation point of each array/databuffer.
                 */

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

                        assertEquals(useWs, db1.isAttached(),"useWs: " + useWs + " db1 data type " + db1.dataType() + " sourceType: " + sourceType);
                        assertFalse(db2.isAttached());

                        //this test has issues with the correct bit conversion from short to half/bfloat16. We exclude this case
                        //because type promotion from short to half/bfloat16 is not technically the way the data
                        //would be expected to show up here.
                        if(!sourceType.equals("boolean") && !sourceType.equals("short") && dt == DataType.HALF && !sourceType.equals("bfloat16") && dt == DataType.BFLOAT16) {
                            System.out.println("Test case source type: " + sourceType + " data type : " + dt);
                            testDBOps(db1);
                            testDBOps(db2);
                        }
                    }
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAsBytes(Nd4jBackend backend) {
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
            Buffer buffer = bb;
            buffer.position(0);
            bb.put(b);

            Nd4j.getAffinityManager().tagLocation(arr2, AffinityManager.Location.HOST);

            assertEquals(arr.toString(), arr2.toString());
            assertEquals(arr, arr2);

            //Sanity check on data buffer getters:
            DataBuffer db = arr.data();
            DataBuffer db2 = arr2.data();
            for(int i = 0; i < 10; i++) {
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


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEnsureLocation(){
        //https://github.com/eclipse/deeplearning4j/issues/8783
        Nd4j.create(1);

        BytePointer bp = new BytePointer(5);


        Pointer ptr =Nd4j.getNativeOps().pointerForAddress(bp.address());
        DataBuffer buff = Nd4j.createBuffer(ptr, 5, DataType.INT8);


        INDArray arr2 = Nd4j.create(buff, new long[]{5}, new long[]{1}, 0, 'c', DataType.INT8);
        long before = arr2.data().pointer().address();
        Nd4j.getAffinityManager().ensureLocation(arr2, AffinityManager.Location.HOST);
        long after = arr2.data().pointer().address();

        assertEquals(before, after);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDataBufferSyncAfterResize(Nd4jBackend backend) {
        // Test that DataBuffer operations remain correct after creating arrays of different sizes
        // and triggering sync operations. This exercises the _primaryAllocBytes/_specialAllocBytes
        // tracking that prevents buffer overruns when setPrimaryBuffer/setSpecialBuffer are called
        // with different sizes.

        // Create a small array, write data, then create a larger array and verify no corruption
        INDArray small = Nd4j.createFromArray(1.0f, 2.0f, 3.0f);
        INDArray large = Nd4j.createFromArray(10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f);

        // Force sync to host
        Nd4j.getAffinityManager().ensureLocation(small, AffinityManager.Location.HOST);
        Nd4j.getAffinityManager().ensureLocation(large, AffinityManager.Location.HOST);

        // Verify data integrity
        assertEquals(1.0f, small.getFloat(0), 1e-5);
        assertEquals(2.0f, small.getFloat(1), 1e-5);
        assertEquals(3.0f, small.getFloat(2), 1e-5);
        assertEquals(80.0f, large.getFloat(7), 1e-5);

        // Create a dup (exercises copyBufferFrom with alloc tracking)
        INDArray largeDup = large.dup();
        Nd4j.getAffinityManager().ensureLocation(largeDup, AffinityManager.Location.HOST);
        assertEquals(large, largeDup);

        // Modify original and verify dup is independent
        large.putScalar(0, 999.0f);
        Nd4j.getAffinityManager().ensureLocation(large, AffinityManager.Location.HOST);
        assertEquals(999.0f, large.getFloat(0), 1e-5);
        assertEquals(10.0f, largeDup.getFloat(0), 1e-5);

        // Test with assign (exercises buffer copy paths)
        INDArray target = Nd4j.zeros(DataType.FLOAT, 8);
        target.assign(largeDup);
        Nd4j.getAffinityManager().ensureLocation(target, AffinityManager.Location.HOST);
        assertEquals(largeDup, target);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCopyBufferFp8AtOffsets(Nd4jBackend backend) {
        // DataBuffer::memcpy (copyBuffer, bitcast, DataBuffer.dup, DSP replica copies) dispatches FP8
        // outside SD_COMMON_TYPES. Bitcasting through INT8 keeps the check byte-exact, and nonzero
        // offsets on both sides expose a wrong element size.
        int length = 16;
        int count = 6;
        int sourceOffset = 3;
        int targetOffset = 5;
        byte[] sourceBytes = new byte[length];
        byte[] targetBytes = new byte[length];
        int[] expected = new int[length];
        for (int i = 0; i < length; i++) {
            sourceBytes[i] = (byte) (0x10 + i);
            targetBytes[i] = (byte) (0x40 + i);
            expected[i] = targetBytes[i];
        }
        for (int i = 0; i < count; i++) {
            expected[targetOffset + i] = sourceBytes[sourceOffset + i];
        }

        for (DataType fp8 : new DataType[]{DataType.FLOAT8, DataType.FLOAT8_E5M2}) {
            INDArray source = Nd4j.create(fp8, length);
            INDArray target = Nd4j.create(fp8, length);
            Nd4j.exec(new BitCast(Nd4j.createFromArray(sourceBytes), fp8, source));
            Nd4j.exec(new BitCast(Nd4j.createFromArray(targetBytes), fp8, target));

            Nd4j.getNativeOps().copyBuffer(target.data().opaqueBuffer(), count,
                    source.data().opaqueBuffer(), sourceOffset, targetOffset);

            INDArray targetAsBytes = Nd4j.create(DataType.INT8, length);
            Nd4j.exec(new BitCast(target, DataType.INT8, targetAsBytes));
            assertArrayEquals(expected, targetAsBytes.toIntVector(),
                    fp8 + " copy must replace exactly the addressed bytes");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testHalfFromFloatMatchesNativeCast(Nd4jBackend backend) {
        // toHalfs builds the HALF buffers Java fills from arrays (CUDA buffers, scalars,
        // constants). The exponents reach each rounding branch: at most half the smallest
        // subnormal, subnormals, the carry from the largest subnormal into the smallest normal,
        // normals, the 65504/65520 overflow edge and past it.
        assertFloatsMatchNativeCast(DataType.HALF, new int[]{101, 102, 103, 112, 113, 127, 142, 143},
                ArrayTypeConverters::toHalfs);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBfloat16FromFloatMatchesNativeCast(Nd4jBackend backend) {
        // toBfloats builds the BFLOAT16 buffers Java fills from arrays. bfloat16 keeps the float
        // exponent, so one rounding serves every binade; the exponents reach subnormals, the carry
        // into the smallest normal, normals, the round past the largest finite value to infinity,
        // infinity and every NaN.
        assertFloatsMatchNativeCast(DataType.BFLOAT16, new int[]{0, 1, 127, 254, 255},
                ArrayTypeConverters::toBfloats);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testHalfBfloat16BitsMatchNativeCast(Nd4jBackend backend) {
        // Every 16-bit pattern read as each type; the element index is the pattern.
        short[] patterns = new short[1 << 16];
        short[] halfToBfloat16 = new short[patterns.length];
        short[] bfloat16ToHalf = new short[patterns.length];
        float[] halfToFloat = new float[patterns.length];
        for (int i = 0; i < patterns.length; i++) {
            patterns[i] = (short) i;
            halfToBfloat16[i] = ArrayTypeConverters.toBFloat16(patterns[i]);
            bfloat16ToHalf[i] = ArrayTypeConverters.bfloat16ToShort(patterns[i]);
            halfToFloat[i] = ArrayTypeConverters.halfToFloat(patterns[i]);
        }
        INDArray halfs = bitsAs(patterns, DataType.HALF);
        assertBitsMatchNativeCast("HALF to BFLOAT16", halfs, DataType.BFLOAT16, halfToBfloat16);
        assertBitsMatchNativeCast("BFLOAT16 to HALF", bitsAs(patterns, DataType.BFLOAT16), DataType.HALF,
                bfloat16ToHalf);
        float[] expected = halfs.castTo(DataType.FLOAT).toFloatVector();
        for (int i = 0; i < patterns.length; i++) {
            if (Float.floatToRawIntBits(expected[i]) == Float.floatToRawIntBits(halfToFloat[i])
                    || (Float.isNaN(expected[i]) && Float.isNaN(halfToFloat[i]))) continue;
            fail(String.format("half 0x%04x casts to %s natively but converts to %s in Java",
                    i, expected[i], halfToFloat[i]));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalarKeepsNegativeZero(Nd4jBackend backend) {
        // A scalar op's value reaches a plan as tArgs[0] through getDouble(0). relu with cutoff -0
        // returns -0 for negative inputs, so the sign must survive the scalar's storage type.
        for (DataType dtype : new DataType[]{DataType.HALF, DataType.BFLOAT16, DataType.FLOAT, DataType.DOUBLE}) {
            INDArray scalar = Nd4j.scalar(dtype, -0.0);
            assertEquals(Double.doubleToRawLongBits(-0.0), Double.doubleToRawLongBits(scalar.getDouble(0)),
                    dtype + " scalar -0");
        }
    }

    /**
     * Every float of the given exponents, of both signs, and the specials, converted to the
     * target in Java must give the bits libnd4j's cast gives. The element index is the mantissa.
     */
    private static void assertFloatsMatchNativeCast(DataType target, int[] exponents,
                                                    Function<float[], short[]> javaConversion) {
        float[] values = new float[1 << 23];
        for (int exponent : exponents) {
            for (int sign = 0; sign < 2; sign++) {
                for (int mantissa = 0; mantissa < values.length; mantissa++) {
                    values[mantissa] = Float.intBitsToFloat((sign << 31) | (exponent << 23) | mantissa);
                }
                assertBitsMatchNativeCast("FLOAT exponent " + exponent + (sign == 0 ? "" : ", negative") + " to "
                        + target, Nd4j.createFromArray(values), target, javaConversion.apply(values));
            }
        }
        float[] specials = {0.0f, -0.0f, Float.MIN_VALUE, -Float.MIN_VALUE, Float.MAX_VALUE, -Float.MAX_VALUE,
                Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NaN, Float.intBitsToFloat(0xffc00000),
                Float.intBitsToFloat(0x7f800001)};
        assertBitsMatchNativeCast("FLOAT specials to " + target, Nd4j.createFromArray(specials), target,
                javaConversion.apply(specials));
    }

    /** NaN payloads differ between conversions, so any two NaNs match. */
    private static void assertBitsMatchNativeCast(String context, INDArray source, DataType target,
                                                  short[] javaBits) {
        INDArray nativeBits = Nd4j.create(DataType.SHORT, source.length());
        Nd4j.exec(new BitCast(source.castTo(target), DataType.SHORT, nativeBits));
        int[] expected = nativeBits.toIntVector();
        for (int i = 0; i < javaBits.length; i++) {
            int e = expected[i] & 0xffff;
            int a = javaBits[i] & 0xffff;
            if (e == a || (isNaN(target, e) && isNaN(target, a))) continue;
            fail(String.format("%s: element %d (%s) casts to 0x%04x natively but converts to 0x%04x in Java",
                    context, i, source.getDouble(i), e, a));
        }
    }

    private static INDArray bitsAs(short[] patterns, DataType type) {
        INDArray out = Nd4j.create(type, patterns.length);
        Nd4j.exec(new BitCast(Nd4j.createFromArray(patterns), type, out));
        return out;
    }

    private static boolean isNaN(DataType type, int bits) {
        int exponentMask = type == DataType.HALF ? 0x7c00 : 0x7f80;
        return (bits & exponentMask) == exponentMask && (bits & 0x7fff & ~exponentMask) != 0;
    }

    @Override
    public char ordering() {
        return 'c';
    }

}

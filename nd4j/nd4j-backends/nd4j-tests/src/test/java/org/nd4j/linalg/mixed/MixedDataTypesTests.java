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

package org.nd4j.linalg.mixed;

import com.google.flatbuffers.FlatBufferBuilder;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.graph.FlatArray;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.MirroringPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.impl.reduce.bool.IsInf;
import org.nd4j.linalg.api.ops.impl.reduce.bool.IsNaN;
import org.nd4j.linalg.api.ops.impl.reduce.longer.CountNonZero;
import org.nd4j.linalg.api.ops.impl.reduce3.CosineSimilarity;
import org.nd4j.linalg.api.ops.impl.transforms.custom.EqualTo;
import org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.api.memory.abstracts.Nd4jWorkspace;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.nativeblas.NativeOpsHolder;

import static org.junit.Assert.*;

/**
 * Basic tests for mixed data types
 * @author raver119@gmail.com
 */
@Slf4j
public class MixedDataTypesTests extends BaseNd4jTest {

    public MixedDataTypesTests(Nd4jBackend b){
        super(b);
    }

    @Override
    public char ordering(){
        return 'c';
    }

    @Test
    public void testBasicCreation_1() {
        val array = Nd4j.create(DataType.LONG, 3, 3);

        assertNotNull(array);
        assertEquals(9, array.length());
        assertEquals(DataType.LONG, array.dataType());
        assertEquals(DataType.LONG, ArrayOptionsHelper.dataType(array.shapeInfoJava()));
    }

    @Test
    public void testBasicCreation_2() {
        val array = Nd4j.create(DataType.SHORT, 3, 3);

        assertNotNull(array);
        assertEquals(9, array.length());
        assertEquals(DataType.SHORT, array.dataType());
        assertEquals(DataType.SHORT, ArrayOptionsHelper.dataType(array.shapeInfoJava()));
    }

    @Test
    public void testBasicCreation_3() {
        val array = Nd4j.create(DataType.HALF, 3, 3);

        assertNotNull(array);
        assertEquals(9, array.length());
        assertEquals(DataType.HALF, array.dataType());
        assertEquals(DataType.HALF, ArrayOptionsHelper.dataType(array.shapeInfoJava()));
    }

    @Test
    public void testBasicCreation_4() {
        val scalar = Nd4j.scalar(DataType.DOUBLE, 1.0);
        assertNotNull(scalar);
        assertEquals(0, scalar.rank());
        assertEquals(1, scalar.length());
        assertEquals(DataType.DOUBLE, scalar.dataType());
        assertEquals(1.0, scalar.getDouble(0), 1e-5);
    }

    @Test
    public void testBasicCreation_5() {
        val scalar = Nd4j.scalar(Integer.valueOf(1));
        assertNotNull(scalar);
        assertEquals(0, scalar.rank());
        assertEquals(1, scalar.length());
        assertEquals(DataType.INT, scalar.dataType());
        assertEquals(1.0, scalar.getInt(0), 1e-5);
    }

    @Test
    public void testBasicCreation_5_0() {
        val scalar = Nd4j.scalar(Long.valueOf(1));
        assertNotNull(scalar);
        assertEquals(0, scalar.rank());
        assertEquals(1, scalar.length());
        assertEquals(DataType.LONG, scalar.dataType());
        assertEquals(1.0, scalar.getInt(0), 1e-5);
    }

    @Test
    public void testBasicCreation_5_1() {
        val scalar = Nd4j.scalar(Double.valueOf(1));
        assertNotNull(scalar);
        assertEquals(0, scalar.rank());
        assertEquals(1, scalar.length());
        assertEquals(DataType.DOUBLE, scalar.dataType());
        assertEquals(1.0, scalar.getDouble(0), 1e-5);
    }

    @Test
    public void testBasicCreation_5_2() {
        val scalar = Nd4j.scalar(Float.valueOf(1));
        assertNotNull(scalar);
        assertEquals(0, scalar.rank());
        assertEquals(1, scalar.length());
        assertEquals(DataType.FLOAT, scalar.dataType());
        assertEquals(1.0, scalar.getDouble(0), 1e-5);
    }

    @Test
    public void testBasicCreation_5_3() {
        val scalar = Nd4j.scalar(Short.valueOf((short) 1));
        assertNotNull(scalar);
        assertEquals(0, scalar.rank());
        assertEquals(1, scalar.length());
        assertEquals(DataType.SHORT, scalar.dataType());
        assertEquals(1.0, scalar.getDouble(0), 1e-5);
    }

    @Test
    public void testBasicCreation_5_4() {
        val scalar = Nd4j.scalar(Byte.valueOf((byte) 1));
        assertNotNull(scalar);
        assertEquals(0, scalar.rank());
        assertEquals(1, scalar.length());
        assertEquals(DataType.BYTE, scalar.dataType());
        assertEquals(1.0, scalar.getDouble(0), 1e-5);
    }

    @Test
    public void testBasicCreation_6() {
        val scalar = Nd4j.scalar(1);
        assertNotNull(scalar);
        assertEquals(0, scalar.rank());
        assertEquals(1, scalar.length());
        assertEquals(DataType.INT, scalar.dataType());
        assertEquals(1.0, scalar.getInt(0), 1e-5);
    }

    @Test
    public void testBasicCreation_7() {
        val scalar = Nd4j.scalar(1L);
        assertNotNull(scalar);
        assertEquals(0, scalar.rank());
        assertEquals(1, scalar.length());
        assertEquals(DataType.LONG, scalar.dataType());
        assertEquals(1, scalar.getInt(0));
    }

    @Test
    public void testBasicOps_1() {
        val exp = new int[]{1,1,1,1,1,1,1,1,1};
        val array = Nd4j.create(DataType.INT, 3, 3);
        assertEquals(DataType.INT, array.dataType());
        array.assign(1);

        val vector = array.data().asInt();
        assertArrayEquals(exp, vector);
    }

    @Test
    public void testBasicOps_2() {
        val exp = new int[]{1,1,1,1,1,1,1,1,1};
        val arrayX = Nd4j.create(DataType.INT, 3, 3);
        val arrayY = Nd4j.create(new int[]{1,1,1,1,1,1,1,1,1}, new long[]{3, 3}, DataType.INT);

        arrayX.addi(arrayY);

        val vector = arrayX.data().asInt();
        assertArrayEquals(exp, vector);
    }

    @Test
    public void testBasicOps_3() {
        if (!NativeOpsHolder.getInstance().getDeviceNativeOps().isExperimentalEnabled())
            return;

        val exp = new int[]{1,1,1,1,1,1,1,1,1};
        val arrayX = Nd4j.create(DataType.INT, 3, 3);
        val arrayY = Nd4j.create(new int[]{1,1,1,1,1,1,1,1,1}, new long[]{3, 3}, DataType.LONG);

        val vectorY = arrayY.data().asInt();
        assertArrayEquals(exp, vectorY);

        arrayX.addi(arrayY);

        val vectorX = arrayX.data().asInt();
        assertArrayEquals(exp, vectorX);
    }

    @Test
    public void testBasicOps_4() {
        val arrayX = Nd4j.create(new int[]{7,8,7,9,1,1,1,1,1}, new long[]{3, 3}, DataType.LONG);

        val result = arrayX.maxNumber();
        val l = result.longValue();

        assertEquals(9L, l);
    }

    @Test
    public void testBasicOps_5() {
        val arrayX = Nd4j.create(new int[]{1, 2, 3, 4}, new  long[]{4}, DataType.INT);

        val result = arrayX.meanNumber().floatValue();

        assertEquals(2.5f, result, 1e-5);
    }

    @Test
    public void testBasicOps_6() {
        val arrayX = Nd4j.create(new int[]{1, 0, 0, 4}, new  long[]{4}, DataType.INT);

        val z = Nd4j.getExecutioner().exec(new CountNonZero(arrayX));

        assertEquals(DataType.LONG, z.dataType());
        val result = z.getInt(0);

        assertEquals(2, result);
    }

    @Test
    public void testBasicOps_7() {
        val arrayX = Nd4j.create(new float[]{1, 0, Float.NaN, 4}, new  long[]{4}, DataType.FLOAT);

        val z = Nd4j.getExecutioner().exec(new IsInf(arrayX));

        assertEquals(DataType.BOOL, z.dataType());
        val result = z.getInt(0);

        val z2 = Nd4j.getExecutioner().exec(new IsNaN(arrayX));
        assertEquals(DataType.BOOL, z2.dataType());
        val result2 = z2.getInt(0);

        assertEquals(1, result2);
    }

    @Test
    public void testBasicOps_8() {
        val arrayX = Nd4j.create(new int[]{1, 2, 3, 4}, new  long[]{4}, DataType.INT);
        val arrayY = Nd4j.create(new int[]{1, 0, 0, 4}, new  long[]{4}, DataType.INT);
        val exp = new long[]{1, 0, 0, 1};

        val result = Nd4j.getExecutioner().exec(new EqualTo(arrayX, arrayY, arrayX.ulike().castTo(DataType.BOOL)))[0];
        assertEquals(DataType.BOOL, result.dataType());
        val arr = result.data().asLong();

        assertArrayEquals(exp, arr);
    }

    @Test
    public void testBasicOps_9() {
        val arrayX = Nd4j.create(new int[]{1, 2, 3, 4}, new  long[]{4}, DataType.INT);
        val arrayY = Nd4j.create(new int[]{1, 2, 3, 4}, new  long[]{4}, DataType.INT);
        val exp = new long[]{1, 0, 0, 1};

        val op = new CosineSimilarity(arrayX, arrayY);
        val result = Nd4j.getExecutioner().exec(op);
        val arr = result.getDouble(0);

        assertEquals(1.0, arr, 1e-5);
    }

    @Test
    public void testNewAssign_1() {
        val arrayX = Nd4j.create(DataType.FLOAT, 5);
        val arrayY = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        val exp = Nd4j.create(new float[]{1.f, 2.f, 3.f, 4.f, 5.f});

        arrayX.assign(arrayY);

        assertEquals(exp, arrayX);
    }

    @Test
    public void testNewAssign_2() {
        val arrayX = Nd4j.create(DataType.INT, 5);
        val arrayY = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        val exp = Nd4j.create(new int[]{1, 2, 3, 4, 5}, new long[]{5}, DataType.INT);

        arrayX.assign(arrayY);

        assertEquals(exp, arrayX);
    }

    @Test
    public void testMethods_1() {
        val arrayX = Nd4j.create(new int[]{1, 2, 3, 4}, new  long[]{4}, DataType.INT);
        val arrayY = Nd4j.create(new int[]{1, 2, 3, 4}, new  long[]{4}, DataType.INT);
        val exp = Nd4j.create(new int[]{2, 4, 6, 8}, new  long[]{4}, DataType.INT);

        val arrayZ = arrayX.add(arrayY);
        assertEquals(DataType.INT, arrayZ.dataType());
        assertEquals(exp, arrayZ);
    }

    @Test
    public void testMethods_2() {
        if (!NativeOpsHolder.getInstance().getDeviceNativeOps().isExperimentalEnabled())
            return;

        val arrayX = Nd4j.create(new int[]{1, 2, 3, 4}, new  long[]{4}, DataType.INT);
        val arrayY = Nd4j.create(new double[]{1, 2, 3, 4}, new  long[]{4}, DataType.DOUBLE);
        val exp = Nd4j.create(new double[]{2, 4, 6, 8}, new  long[]{4}, DataType.DOUBLE);

        val arrayZ = arrayX.add(arrayY);

        assertEquals(DataType.DOUBLE, arrayZ.dataType());
        assertEquals(exp, arrayZ);
    }

    @Test
    public void testMethods_3() {
        if (!NativeOpsHolder.getInstance().getDeviceNativeOps().isExperimentalEnabled())
            return;

        val arrayX = Nd4j.create(new int[]{1, 2, 3, 4}, new  long[]{4}, DataType.INT);
        val arrayY = Nd4j.create(new double[]{0.5, 0.5, 0.5, 0.5}, new  long[]{4}, DataType.DOUBLE);
        val exp = Nd4j.create(new double[]{1.5, 2.5, 3.5, 4.5}, new  long[]{4}, DataType.DOUBLE);

        val arrayZ = arrayX.add(arrayY);

        assertEquals(DataType.DOUBLE, arrayZ.dataType());
        assertEquals(exp, arrayZ);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testTypesValidation_1() {
        val arrayX = Nd4j.create(new int[]{1, 2, 3, 4}, new  long[]{4}, DataType.LONG);
        val arrayY = Nd4j.create(new int[]{1, 2, 3, 4}, new  long[]{4}, DataType.INT);
        val exp = new long[]{1, 0, 0, 1};

        val op = new CosineSimilarity(arrayX, arrayY);
        val result = Nd4j.getExecutioner().exec(op);
    }

    @Test(expected = RuntimeException.class)
    public void testTypesValidation_2() {
        val arrayX = Nd4j.create(new int[]{1, 2, 3, 4}, new  long[]{4}, DataType.INT);
        val arrayY = Nd4j.create(new int[]{1, 0, 0, 4}, new  long[]{4}, DataType.LONG);
        val exp = new long[]{1, 0, 0, 1};

        val result = Nd4j.getExecutioner().exec(new EqualTo(arrayX, arrayY, arrayX.ulike().castTo(DataType.BOOL)))[0];
        val arr = result.data().asLong();

        assertArrayEquals(exp, arr);
    }

    @Test(expected = RuntimeException.class)
    public void testTypesValidation_3() {
        val arrayX = Nd4j.create(new int[]{1, 2, 3, 4}, new  long[]{4}, DataType.INT);

        val result = Nd4j.getExecutioner().exec((CustomOp) new SoftMax(arrayX, arrayX, -1));
    }

    public void testTypesValidation_4() {
        val arrayX = Nd4j.create(new int[]{1, 2, 3, 4}, new  long[]{4}, DataType.INT);
        val arrayY = Nd4j.create(new int[]{1, 0, 0, 4}, new  long[]{4}, DataType.DOUBLE);
        val arrayE = Nd4j.create(new int[]{2, 2, 3, 8}, new  long[]{4}, DataType.INT);

        arrayX.addi(arrayY);
        assertEquals(arrayE, arrayX);
    }


    @Test
    public void testFlatSerde_1() {
        val arrayX = Nd4j.create(new int[]{1, 2, 3, 4}, new  long[]{4}, DataType.INT);

        val builder = new FlatBufferBuilder(512);
        val flat = arrayX.toFlatArray(builder);
        builder.finish(flat);
        val db = builder.dataBuffer();

        val flatb = FlatArray.getRootAsFlatArray(db);

        val restored = Nd4j.createFromFlatArray(flatb);

        assertEquals(arrayX, restored);
    }

    @Test
    public void testFlatSerde_2() {
        val arrayX = Nd4j.create(new long[]{1, 2, 3, 4}, new  long[]{4}, DataType.LONG);

        val builder = new FlatBufferBuilder(512);
        val flat = arrayX.toFlatArray(builder);
        builder.finish(flat);
        val db = builder.dataBuffer();

        val flatb = FlatArray.getRootAsFlatArray(db);

        val restored = Nd4j.createFromFlatArray(flatb);

        assertEquals(arrayX, restored);
    }

    @Test
    public void testFlatSerde_3() {
        val arrayX = Nd4j.create(new boolean[]{true, false, true, true}, new  long[]{4}, DataType.BOOL);

        val builder = new FlatBufferBuilder(512);
        val flat = arrayX.toFlatArray(builder);
        builder.finish(flat);
        val db = builder.dataBuffer();

        val flatb = FlatArray.getRootAsFlatArray(db);

        val restored = Nd4j.createFromFlatArray(flatb);

        assertEquals(arrayX, restored);
    }

    @Test
    public void testBoolFloatCast2(){
        val first = Nd4j.zeros(DataType.FLOAT, 3, 5000);
        INDArray asBool = first.castTo(DataType.BOOL);
        INDArray not = Transforms.not(asBool);  //
        INDArray asFloat = not.castTo(DataType.FLOAT);

//        System.out.println(not);
//        System.out.println(asFloat);
        INDArray exp = Nd4j.ones(DataType.FLOAT, 3, 5000);
        assertEquals(DataType.FLOAT, exp.dataType());
        assertEquals(exp.dataType(), asFloat.dataType());

        val arrX = asFloat.data().asFloat();
        val arrE = exp.data().asFloat();
        assertArrayEquals(arrE, arrX, 1e-5f);

        assertEquals(exp, asFloat);
    }

    @Test
    public void testReduce3Large() {
        val arrayX = Nd4j.create(DataType.FLOAT, 10, 5000);
        val arrayY = Nd4j.create(DataType.FLOAT, 10, 5000);

        assertTrue(arrayX.equalsWithEps(arrayY, -1e-5f));
    }


    @Test
    public void testAssignScalarSimple(){
        for(DataType dt : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
            INDArray arr = Nd4j.scalar(dt, 10.0);
            arr.assign(2.0);
//            System.out.println(dt + " - value: " + arr + " - " + arr.getDouble(0));
        }
    }

    @Test
    public void testSimple(){
        Nd4j.create(1);
        for(DataType dt : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF, DataType.INT, DataType.LONG}) {
//            System.out.println("----- " + dt + " -----");
            INDArray arr = Nd4j.ones(dt,1, 5);
//            System.out.println("Ones: " + arr);
            arr.assign(1.0);
//            System.out.println("assign(1.0): " + arr);
//            System.out.println("DIV: " + arr.div(8));
//            System.out.println("MUL: " + arr.mul(8));
//            System.out.println("SUB: " + arr.sub(8));
//            System.out.println("ADD: " + arr.add(8));
//            System.out.println("RDIV: " + arr.rdiv(8));
//            System.out.println("RSUB: " + arr.rsub(8));
            arr.div(8);
            arr.mul(8);
            arr.sub(8);
            arr.add(8);
            arr.rdiv(8);
            arr.rsub(8);
        }
    }

    @Test
    public void testWorkspaceBool(){
        val conf = WorkspaceConfiguration.builder().minSize(10 * 1024 * 1024)
                .overallocationLimit(1.0).policyAllocation(AllocationPolicy.OVERALLOCATE)
                .policyLearning(LearningPolicy.FIRST_LOOP).policyMirroring(MirroringPolicy.FULL)
                .policySpill(SpillPolicy.EXTERNAL).build();

        val ws = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(conf, "WS");

        for( int i=0; i<10; i++ ) {
            try (val workspace = (Nd4jWorkspace)ws.notifyScopeEntered() ) {
                val bool = Nd4j.create(DataType.BOOL, 1, 10);
                val dbl = Nd4j.create(DataType.DOUBLE, 1, 10);

                val boolAttached = bool.isAttached();
                val doubleAttached = dbl.isAttached();

//                System.out.println(i + "\tboolAttached=" + boolAttached + ", doubleAttached=" + doubleAttached );
                //System.out.println("bool: " + bool);        //java.lang.IllegalStateException: Indexer must never be null
                //System.out.println("double: " + dbl);
            }
        }
    }

    @Test
    @Ignore("AB 2019/05/23 - Failing on linux-x86_64-cuda-9.2 - see issue #7657")
    public void testArrayCreationFromPointer() {
        val source = Nd4j.create(new double[]{1, 2, 3, 4, 5});

        val pAddress = source.data().addressPointer();
        val shape = source.shape();
        val stride = source.stride();
        val order = source.ordering();

        val buffer = Nd4j.createBuffer(pAddress, source.length(), source.dataType());
        val restored = Nd4j.create(buffer, shape, stride, 0, order, source.dataType());
        assertEquals(source, restored);

        assertArrayEquals(source.toDoubleVector(), restored.toDoubleVector(), 1e-5);

        assertEquals(source.getDouble(0), restored.getDouble(0), 1e-5);
    }

    @Test
    public void testBfloat16_1() {
        val x = Nd4j.create(DataType.BFLOAT16, 5);
        val y = Nd4j.createFromArray(new int[]{2, 2, 2, 2, 2}).castTo(DataType.BFLOAT16);

        x.addi(y);
        assertEquals(x, y);
    }

    @Test
    public void testUint16_1() {
        val x = Nd4j.create(DataType.UINT16, 5);
        val y = Nd4j.createFromArray(new int[]{2, 2, 2, 2, 2}).castTo(DataType.UINT16);

        x.addi(y);
        assertEquals(x, y);
    }

    @Test
    public void testUint32_1() {
        val x = Nd4j.create(DataType.UINT32, 5);
        val y = Nd4j.createFromArray(new int[]{2, 2, 2, 2, 2}).castTo(DataType.UINT32);

        x.addi(y);
        assertEquals(x, y);
    }

    @Test
    public void testUint64_1() {
        val x = Nd4j.create(DataType.UINT64, 5);
        val y = Nd4j.createFromArray(new int[]{2, 2, 2, 2, 2}).castTo(DataType.UINT64);

        x.addi(y);
        assertEquals(x, y);
    }
}

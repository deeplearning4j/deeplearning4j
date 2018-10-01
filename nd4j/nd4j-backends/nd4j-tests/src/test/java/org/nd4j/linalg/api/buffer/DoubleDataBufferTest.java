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

package org.nd4j.linalg.api.buffer;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.indexer.DoubleIndexer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.util.SerializationUtils;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;

import java.io.*;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Double data buffer tests
 *
 * This tests the double buffer data opType
 * Put all buffer related tests here
 *
 * @author Adam Gibson
 */
@RunWith(Parameterized.class)
public class DoubleDataBufferTest extends BaseNd4jTest {
    DataType initialType;

    public DoubleDataBufferTest(Nd4jBackend backend) {
        super(backend);
        initialType = Nd4j.dataType();
    }



    @Before
    public void before() {

        DataTypeUtil.setDTypeForContext(DataType.DOUBLE);
    }

    @After
    public void after() {
        DataTypeUtil.setDTypeForContext(initialType);
    }

    @Test
    public void testPointerCreation() {
        DoublePointer floatPointer = new DoublePointer(1, 2, 3, 4);
        Indexer indexer = DoubleIndexer.create(floatPointer);
        DataBuffer buffer = Nd4j.createBuffer(floatPointer, DataType.DOUBLE, 4, indexer);
        DataBuffer other = Nd4j.createBuffer(new double[] {1, 2, 3, 4});
        assertArrayEquals(other.asDouble(), buffer.asDouble(), 0.001);
    }

    @Test
    public void testGetSet() throws Exception {
        double[] d1 = new double[] {1, 2, 3, 4};
        DataBuffer d = Nd4j.createBuffer(d1);
        double[] d2 = d.asDouble();
        assertArrayEquals(d1, d2, 1e-1f);

    }



    @Test
    public void testSerialization2() throws Exception {
        INDArray[] arr = new INDArray[] {Nd4j.ones(1, 10),
                        //      Nd4j.ones(5,10).getRow(2)
        };

        for (INDArray a : arr) {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            try (ObjectOutputStream oos = new ObjectOutputStream(baos)) {
                oos.writeObject(a);
                oos.flush();
            }



            byte[] bytes = baos.toByteArray();

            ByteArrayInputStream bais = new ByteArrayInputStream(bytes);
            ObjectInputStream ois = new ObjectInputStream(bais);

            INDArray aDeserialized = (INDArray) ois.readObject();

            System.out.println(aDeserialized);
            assertEquals(Nd4j.ones(1, 10), aDeserialized);
        }
    }


    @Test
    public void testSerialization() {
        DataBuffer buf = Nd4j.createBuffer(5);
        String fileName = "buf.ser";
        File file = new File(fileName);
        file.deleteOnExit();
        SerializationUtils.saveObject(buf, file);
        DataBuffer buf2 = SerializationUtils.readObject(file);
        //assertEquals(buf, buf2);
        assertArrayEquals(buf.asDouble(), buf2.asDouble(), 0.001);

        Nd4j.alloc = DataBuffer.AllocationMode.DIRECT;
        buf = Nd4j.createBuffer(5);
        file.deleteOnExit();
        SerializationUtils.saveObject(buf, file);
        buf2 = SerializationUtils.readObject(file);
        //        assertEquals(buf, buf2);
        assertArrayEquals(buf.asDouble(), buf2.asDouble(), 0.001);
    }


    @Test
    public void testDup() throws Exception {
        double[] d1 = new double[] {1, 2, 3, 4};
        DataBuffer d = Nd4j.createBuffer(d1);
        DataBuffer d2 = d.dup();
        assertArrayEquals(d.asDouble(), d2.asDouble(), 0.0001f);
    }



    @Test
    public void testPut() throws Exception {
        double[] d1 = new double[] {1, 2, 3, 4};
        DataBuffer d = Nd4j.createBuffer(d1);
        d.put(0, 0.0);
        double[] result = new double[] {0, 2, 3, 4};
        d1 = d.asDouble();
        assertArrayEquals(d1, result, 1e-1f);
    }


    @Test
    public void testGetRange() throws Exception {
        DataBuffer buffer = Nd4j.linspace(1, 5, 5).data();
        double[] get = buffer.getDoublesAt(0, 3);
        double[] data = new double[] {1, 2, 3};
        assertArrayEquals(get, data, 1e-1f);


        double[] get2 = buffer.asDouble();
        double[] allData = buffer.getDoublesAt(0, (int) buffer.length());
        assertArrayEquals(get2, allData, 1e-1f);


    }

    @Test
    public void testGetOffsetRange() throws Exception {
        DataBuffer buffer = Nd4j.linspace(1, 5, 5).data();
        double[] get = buffer.getDoublesAt(1, 3);
        double[] data = new double[] {2, 3, 4};
        assertArrayEquals(get, data, 1e-1f);


        double[] allButLast = new double[] {2, 3, 4, 5};

        double[] allData = buffer.getDoublesAt(1, (int) buffer.length());
        assertArrayEquals(allButLast, allData, 1e-1f);

    }

    @Test
    public void testAssign() {
        DataBuffer assertion = Nd4j.createBuffer(new double[] {1, 2, 3});
        DataBuffer one = Nd4j.createBuffer(new double[] {1});
        DataBuffer twoThree = Nd4j.createBuffer(new double[] {2, 3});
        DataBuffer blank = Nd4j.createBuffer(new double[] {0, 0, 0});
        blank.assign(one, twoThree);
        assertArrayEquals(assertion.asDouble(), blank.asDouble(), 0.0001);
    }


    @Test
    public void testOffset() {
        DataBuffer create = Nd4j.createBuffer(new double[] {1, 2, 3, 4}, 2);
        assertEquals(2, create.length());
        assertEquals(4, create.underlyingLength());
        assertEquals(2, create.offset());
        assertEquals(3, create.getDouble(0), 1e-1);
        assertEquals(4, create.getDouble(1), 1e-1);

    }

    @Test
    public void testReallocation() {
        DataBuffer buffer = Nd4j.createBuffer(new double[] {1, 2, 3, 4});
        assertEquals(4, buffer.capacity());
        double[] old = buffer.asDouble();
        buffer.reallocate(6);
        assertEquals(6, buffer.capacity());
        assertArrayEquals(old, buffer.asDouble(), 1e-1);
    }

    @Test
    public void testReallocationWorkspace() {
        WorkspaceConfiguration initialConfig = WorkspaceConfiguration.builder().initialSize(10 * 1024L * 1024L)
                        .policyAllocation(AllocationPolicy.STRICT).policyLearning(LearningPolicy.NONE).build();
        MemoryWorkspace workspace = Nd4j.getWorkspaceManager().getAndActivateWorkspace(initialConfig, "SOME_ID");

        DataBuffer buffer = Nd4j.createBuffer(new double[] {1, 2, 3, 4});
        double[] old = buffer.asDouble();
        assertTrue(buffer.isAttached());
        assertEquals(4, buffer.capacity());
        buffer.reallocate(6);
        assertEquals(6, buffer.capacity());
        assertArrayEquals(old, buffer.asDouble(), 1e-1);
        workspace.close();

    }

    @Test
    public void testAddressPointer(){
        if( Nd4j.getExecutioner().type() !=  OpExecutioner.ExecutionerType.NATIVE_CPU ){
            return;
        }
        DataBuffer buffer = Nd4j.createBuffer(new double[] {1, 2, 3, 4});
        DataBuffer wrappedBuffer = Nd4j.createBuffer(buffer, 1, 2);

        DoublePointer pointer = (DoublePointer) wrappedBuffer.addressPointer();
        Assert.assertEquals(buffer.getDouble(1), pointer.get(0), 1e-1);
        Assert.assertEquals(buffer.getDouble(2), pointer.get(1), 1e-1);

        try {
            pointer.asBuffer().get(3); // Try to access element outside pointer capacity.
            Assert.fail("Accessing this address should not be allowed!");
        } catch (IndexOutOfBoundsException e) {
            // do nothing
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}

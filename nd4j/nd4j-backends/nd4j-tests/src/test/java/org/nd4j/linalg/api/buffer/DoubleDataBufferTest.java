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

package org.nd4j.linalg.api.buffer;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.indexer.DoubleIndexer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.io.TempDir;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.common.util.SerializationUtils;

import java.io.*;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Double data buffer tests
 *
 * This tests the double buffer data opType
 * Put all buffer related tests here
 *
 * @author Adam Gibson
 */

@Disabled("AB 2019/05/23 - Failing on linux-x86_64-cuda-9.2 - see issue #7657")
public class DoubleDataBufferTest extends BaseNd4jTestWithBackends {



    DataType initialType = Nd4j.dataType();



    @BeforeEach
    public void before(Nd4jBackend backend) {

        DataTypeUtil.setDTypeForContext(DataType.DOUBLE);
    }

    @AfterEach
    public void after(Nd4jBackend backend) {
        DataTypeUtil.setDTypeForContext(initialType);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPointerCreation(Nd4jBackend backend) {
        DoublePointer floatPointer = new DoublePointer(1, 2, 3, 4);
        Indexer indexer = DoubleIndexer.create(floatPointer);
        DataBuffer buffer = Nd4j.createBuffer(floatPointer, DataType.DOUBLE, 4, indexer);
        DataBuffer other = Nd4j.createBuffer(new double[] {1, 2, 3, 4});
        assertArrayEquals(other.asDouble(), buffer.asDouble(), 0.001);
    }

      @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetSet(Nd4jBackend backend) {
        double[] d1 = new double[] {1, 2, 3, 4};
        DataBuffer d = Nd4j.createBuffer(d1);
        double[] d2 = d.asDouble();
        assertArrayEquals(d1, d2, 1e-1f);

    }



      @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
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


      @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSerialization(@TempDir Path testDir) throws Exception {
        File dir = testDir.toFile();
        DataBuffer buf = Nd4j.createBuffer(5);
        String fileName = "buf.ser";
        File file = new File(dir, fileName);
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


      @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDup(Nd4jBackend backend) {
        double[] d1 = new double[] {1, 2, 3, 4};
        DataBuffer d = Nd4j.createBuffer(d1);
        DataBuffer d2 = d.dup();
        assertArrayEquals(d.asDouble(), d2.asDouble(), 0.0001f);
    }



      @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPut(Nd4jBackend backend) {
        double[] d1 = new double[] {1, 2, 3, 4};
        DataBuffer d = Nd4j.createBuffer(d1);
        d.put(0, 0.0);
        double[] result = new double[] {0, 2, 3, 4};
        d1 = d.asDouble();
        assertArrayEquals(d1, result, 1e-1f);
    }


      @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetRange(Nd4jBackend backend) {
        DataBuffer buffer = Nd4j.linspace(1, 5, 5, DataType.DOUBLE).data();
        double[] get = buffer.getDoublesAt(0, 3);
        double[] data = new double[] {1, 2, 3};
        assertArrayEquals(get, data, 1e-1f);


        double[] get2 = buffer.asDouble();
        double[] allData = buffer.getDoublesAt(0, (int) buffer.length());
        assertArrayEquals(get2, allData, 1e-1f);


    }

      @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetOffsetRange(Nd4jBackend backend) {
        DataBuffer buffer = Nd4j.linspace(1, 5, 5, DataType.DOUBLE).data();
        double[] get = buffer.getDoublesAt(1, 3);
        double[] data = new double[] {2, 3, 4};
        assertArrayEquals(get, data, 1e-1f);


        double[] allButLast = new double[] {2, 3, 4, 5};

        double[] allData = buffer.getDoublesAt(1, (int) buffer.length());
        assertArrayEquals(allButLast, allData, 1e-1f);

    }

      @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAssign(Nd4jBackend backend) {
        DataBuffer assertion = Nd4j.createBuffer(new double[] {1, 2, 3});
        DataBuffer one = Nd4j.createBuffer(new double[] {1});
        DataBuffer twoThree = Nd4j.createBuffer(new double[] {2, 3});
        DataBuffer blank = Nd4j.createBuffer(new double[] {0, 0, 0});
        blank.assign(one, twoThree);
        assertArrayEquals(assertion.asDouble(), blank.asDouble(), 0.0001);
    }


      @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testOffset(Nd4jBackend backend) {
        DataBuffer create = Nd4j.createBuffer(new double[] {1, 2, 3, 4}, 2);
        assertEquals(2, create.length());
        assertEquals(0, create.offset());
        assertEquals(3, create.getDouble(0), 1e-1);
        assertEquals(4, create.getDouble(1), 1e-1);

    }

      @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReallocation(Nd4jBackend backend) {
        DataBuffer buffer = Nd4j.createBuffer(new double[] {1, 2, 3, 4});
        assertEquals(4, buffer.capacity());
        double[] old = buffer.asDouble();
        buffer.reallocate(6);
        assertEquals(6, buffer.capacity());
        assertArrayEquals(old, Arrays.copyOf(buffer.asDouble(), 4), 1e-1);
    }

      @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReallocationWorkspace(Nd4jBackend backend) {
        WorkspaceConfiguration initialConfig = WorkspaceConfiguration.builder().initialSize(10 * 1024L * 1024L)
                .policyAllocation(AllocationPolicy.STRICT).policyLearning(LearningPolicy.NONE).build();
        MemoryWorkspace workspace = Nd4j.getWorkspaceManager().getAndActivateWorkspace(initialConfig, "SOME_ID");

        DataBuffer buffer = Nd4j.createBuffer(new double[] {1, 2, 3, 4});
        double[] old = buffer.asDouble();
        assertTrue(buffer.isAttached());
        assertEquals(4, buffer.capacity());
        buffer.reallocate(6);
        assertEquals(6, buffer.capacity());
        assertArrayEquals(old, Arrays.copyOf(buffer.asDouble(), 4), 1e-1);
        workspace.close();

    }

      @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAddressPointer(){
        if( Nd4j.getExecutioner().type() !=  OpExecutioner.ExecutionerType.NATIVE_CPU ){
            return;
        }
        DataBuffer buffer = Nd4j.createBuffer(new double[] {1, 2, 3, 4});
        DataBuffer wrappedBuffer = Nd4j.createBuffer(buffer, 1, 2);

        DoublePointer pointer = (DoublePointer) wrappedBuffer.addressPointer();
        assertEquals(buffer.getDouble(1), pointer.get(0), 1e-1);
        assertEquals(buffer.getDouble(2), pointer.get(1), 1e-1);

        try {
            pointer.asBuffer().get(3); // Try to access element outside pointer capacity.
            fail("Accessing this address should not be allowed!");
        } catch (IndexOutOfBoundsException e) {
            // do nothing
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}

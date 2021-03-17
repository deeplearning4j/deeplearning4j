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

import lombok.val;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.indexer.FloatIndexer;
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
import java.nio.ByteBuffer;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Float data buffer tests
 *
 * This tests the float buffer data opType
 * Put all buffer related tests here
 *
 * @author Adam Gibson
 */
@Disabled("AB 2019/05/21 - Failing on linux-x86_64-cuda-9.2 - see issue #7657")
public class FloatDataBufferTest extends BaseNd4jTestWithBackends {

    DataType initialType = Nd4j.dataType();

    @BeforeEach
    public void before() {
        DataTypeUtil.setDTypeForContext(DataType.FLOAT);
        System.out.println("DATATYPE HERE: " + Nd4j.dataType());
    }

    @AfterEach
    public void after() {
        DataTypeUtil.setDTypeForContext(initialType);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPointerCreation(Nd4jBackend backend) {
        FloatPointer floatPointer = new FloatPointer(1, 2, 3, 4);
        Indexer indexer = FloatIndexer.create(floatPointer);
        DataBuffer buffer = Nd4j.createBuffer(floatPointer, DataType.FLOAT, 4, indexer);
        DataBuffer other = Nd4j.createBuffer(new float[] {1, 2, 3, 4});
        assertArrayEquals(other.asFloat(), buffer.asFloat(), 0.001f);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetSet(Nd4jBackend backend) {
        float[] d1 = new float[] {1, 2, 3, 4};
        DataBuffer d = Nd4j.createBuffer(d1);
        float[] d2 = d.asFloat();
        assertArrayEquals( d1, d2, 1e-1f,getFailureMessage());

    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSerialization(@TempDir Path tempDir,Nd4jBackend backend) throws Exception {
        File dir = tempDir.toFile();
        DataBuffer buf = Nd4j.createBuffer(5);
        String fileName = "buf.ser";
        File file = new File(dir, fileName);
        file.deleteOnExit();
        SerializationUtils.saveObject(buf, file);
        DataBuffer buf2 = SerializationUtils.readObject(file);
        //        assertEquals(buf, buf2);
        assertArrayEquals(buf.asFloat(), buf2.asFloat(), 0.0001f);

        Nd4j.alloc = DataBuffer.AllocationMode.DIRECT;
        buf = Nd4j.createBuffer(5);
        file.deleteOnExit();
        SerializationUtils.saveObject(buf, file);
        buf2 = SerializationUtils.readObject(file);
        //assertEquals(buf, buf2);
        assertArrayEquals(buf.asFloat(), buf2.asFloat(), 0.0001f);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDup(Nd4jBackend backend) {
        float[] d1 = new float[] {1, 2, 3, 4};
        DataBuffer d = Nd4j.createBuffer(d1);
        DataBuffer d2 = d.dup();
        assertArrayEquals(d.asFloat(), d2.asFloat(), 0.001f);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testToNio(Nd4jBackend backend) {
        DataBuffer buff = Nd4j.createTypedBuffer(new double[] {1, 2, 3, 4}, DataType.FLOAT);
        assertEquals(4, buff.length());
        if (buff.allocationMode() == DataBuffer.AllocationMode.HEAP)
            return;

        ByteBuffer nio = buff.asNio();
        assertEquals(16, nio.capacity());

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPut(Nd4jBackend backend) {
        float[] d1 = new float[] {1, 2, 3, 4};
        DataBuffer d = Nd4j.createBuffer(d1);
        d.put(0, 0.0);
        float[] result = new float[] {0, 2, 3, 4};
        d1 = d.asFloat();
        assertArrayEquals(d1, result, 1e-1f,getFailureMessage());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetRange(Nd4jBackend backend) {
        DataBuffer buffer = Nd4j.linspace(1, 5, 5).data();
        float[] get = buffer.getFloatsAt(0, 3);
        float[] data = new float[] {1, 2, 3};
        assertArrayEquals(get, data, 1e-1f,getFailureMessage());


        float[] get2 = buffer.asFloat();
        float[] allData = buffer.getFloatsAt(0, (int) buffer.length());
        assertArrayEquals(get2, allData, 1e-1f,getFailureMessage());


    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetOffsetRange(Nd4jBackend backend) {
        DataBuffer buffer = Nd4j.linspace(1, 5, 5).data();
        float[] get = buffer.getFloatsAt(1, 3);
        float[] data = new float[] {2, 3, 4};
        assertArrayEquals(get, data, 1e-1f,getFailureMessage());


        float[] allButLast = new float[] {2, 3, 4, 5};

        float[] allData = buffer.getFloatsAt(1, (int) buffer.length());
        assertArrayEquals(allButLast, allData, 1e-1f,getFailureMessage());


    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAsBytes(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(5);
        byte[] d = arr.data().asBytes();
        assertEquals(4 * 5, d.length,getFailureMessage());
        INDArray rand = Nd4j.rand(3, 3);
        rand.data().asBytes();

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAssign(Nd4jBackend backend) {
        DataBuffer assertion = Nd4j.createBuffer(new double[] {1, 2, 3});
        DataBuffer one = Nd4j.createBuffer(new double[] {1});
        DataBuffer twoThree = Nd4j.createBuffer(new double[] {2, 3});
        DataBuffer blank = Nd4j.createBuffer(new double[] {0, 0, 0});
        blank.assign(one, twoThree);
        assertArrayEquals(assertion.asFloat(), blank.asFloat(), 0.0001f);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReadWrite(Nd4jBackend backend) throws Exception {
        DataBuffer assertion = Nd4j.createBuffer(new double[] {1, 2, 3});
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(bos);
        assertion.write(dos);

        DataBuffer clone = assertion.dup();
        val stream = new DataInputStream(new ByteArrayInputStream(bos.toByteArray()));
        val header = BaseDataBuffer.readHeader(stream);
        assertion.read(stream, header.getLeft(), header.getMiddle(), header.getRight());
        assertArrayEquals(assertion.asFloat(), clone.asFloat(), 0.0001f);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testOffset(Nd4jBackend backend) {
        DataBuffer create = Nd4j.createBuffer(new float[] {1, 2, 3, 4}, 2);
        assertEquals(2, create.length());
        assertEquals(0, create.offset());
        assertEquals(3, create.getDouble(0), 1e-1);
        assertEquals(4, create.getDouble(1), 1e-1);

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReallocation(Nd4jBackend backend) {
        DataBuffer buffer = Nd4j.createBuffer(new float[] {1, 2, 3, 4});
        assertEquals(4, buffer.capacity());
        float[] old = buffer.asFloat();
        buffer.reallocate(6);
        float[] newBuf = buffer.asFloat();
        assertEquals(6, buffer.capacity());
        assertArrayEquals(old, newBuf, 1e-4F);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReallocationWorkspace(Nd4jBackend backend) {
        WorkspaceConfiguration initialConfig = WorkspaceConfiguration.builder().initialSize(10 * 1024L * 1024L)
                        .policyAllocation(AllocationPolicy.STRICT).policyLearning(LearningPolicy.NONE).build();
        MemoryWorkspace workspace = Nd4j.getWorkspaceManager().getAndActivateWorkspace(initialConfig, "SOME_ID");

        DataBuffer buffer = Nd4j.createBuffer(new float[] {1, 2, 3, 4});
        assertTrue(buffer.isAttached());
        float[] old = buffer.asFloat();
        assertEquals(4, buffer.capacity());
        buffer.reallocate(6);
        assertEquals(6, buffer.capacity());
        float[] newBuf = buffer.asFloat();
        assertArrayEquals(old, newBuf, 1e-4F);
        workspace.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAddressPointer(Nd4jBackend backend){
        if( Nd4j.getExecutioner().type() !=  OpExecutioner.ExecutionerType.NATIVE_CPU ){
            return;
        }

        DataBuffer buffer = Nd4j.createBuffer(new float[] {1, 2, 3, 4});
        DataBuffer wrappedBuffer = Nd4j.createBuffer(buffer, 1, 2);

        FloatPointer pointer = (FloatPointer) wrappedBuffer.addressPointer();
        assertEquals(buffer.getFloat(1), pointer.get(0), 1e-1);
        assertEquals(buffer.getFloat(2), pointer.get(1), 1e-1);

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

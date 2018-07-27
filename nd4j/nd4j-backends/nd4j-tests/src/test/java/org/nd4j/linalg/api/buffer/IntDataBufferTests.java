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


import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.*;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Tests for INT INDArrays and DataBuffers serialization
 *
 * @author raver119@gmail.com
 */
public class IntDataBufferTests extends BaseNd4jTest {

    public IntDataBufferTests(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testBasicSerde1() throws Exception {


        DataBuffer dataBuffer = Nd4j.createBuffer(new int[] {1, 2, 3, 4, 5});
        DataBuffer shapeBuffer = Nd4j.getShapeInfoProvider().createShapeInformation(new int[] {1, 5}).getFirst();
        INDArray intArray = Nd4j.createArrayFromShapeBuffer(dataBuffer, shapeBuffer);

        File tempFile = File.createTempFile("test", "test");
        tempFile.deleteOnExit();

        Nd4j.saveBinary(intArray, tempFile);

        InputStream stream = new FileInputStream(tempFile);
        BufferedInputStream bis = new BufferedInputStream(stream);
        DataInputStream dis = new DataInputStream(bis);

        INDArray loaded = Nd4j.read(dis);

        assertEquals(DataBuffer.Type.INT, loaded.data().dataType());
        assertEquals(DataBuffer.Type.LONG, loaded.shapeInfoDataBuffer().dataType());

        assertEquals(intArray.data().length(), loaded.data().length());

        assertArrayEquals(intArray.data().asInt(), loaded.data().asInt());
    }


    @Test(expected = ND4JIllegalStateException.class)
    public void testOpDiscarded() throws Exception {
        DataBuffer dataBuffer = Nd4j.createBuffer(new int[] {1, 2, 3, 4, 5});
        DataBuffer shapeBuffer = Nd4j.getShapeInfoProvider().createShapeInformation(new int[] {1, 5}).getFirst();
        INDArray intArray = Nd4j.createArrayFromShapeBuffer(dataBuffer, shapeBuffer);

        intArray.add(10f);
    }

    @Test
    public void testReallocation() {
        DataBuffer buffer = Nd4j.createBuffer(new int[] {1, 2, 3, 4});
        assertEquals(4, buffer.capacity());
        int[] old = buffer.asInt();
        buffer.reallocate(6);
        assertEquals(6, buffer.capacity());
        assertArrayEquals(old, buffer.asInt());
    }

    @Test
    public void testReallocationWorkspace() {
        WorkspaceConfiguration initialConfig = WorkspaceConfiguration.builder().initialSize(10 * 1024L * 1024L)
                        .policyAllocation(AllocationPolicy.STRICT).policyLearning(LearningPolicy.NONE).build();
        MemoryWorkspace workspace = Nd4j.getWorkspaceManager().getAndActivateWorkspace(initialConfig, "SOME_ID");

        DataBuffer buffer = Nd4j.createBuffer(new int[] {1, 2, 3, 4});
        int[] old = buffer.asInt();
        assertTrue(buffer.isAttached());
        assertEquals(4, buffer.capacity());
        buffer.reallocate(6);
        assertEquals(6, buffer.capacity());
        assertArrayEquals(old, buffer.asInt());
        workspace.close();
    }

    @Override
    public char ordering() {
        return 'c';
    }
}

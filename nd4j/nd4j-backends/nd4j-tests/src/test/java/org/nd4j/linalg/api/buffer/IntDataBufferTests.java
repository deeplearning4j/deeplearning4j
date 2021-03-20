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
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.*;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

@NativeTag
public class IntDataBufferTests extends BaseNd4jTestWithBackends {


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBasicSerde1() throws Exception {


        DataBuffer dataBuffer = Nd4j.createBuffer(new int[] {1, 2, 3, 4, 5});
        DataBuffer shapeBuffer = Nd4j.getShapeInfoProvider().createShapeInformation(new long[] {1, 5}, DataType.INT).getFirst();
        INDArray intArray = Nd4j.createArrayFromShapeBuffer(dataBuffer, shapeBuffer);

        File tempFile = File.createTempFile("test", "test");
        tempFile.deleteOnExit();

        Nd4j.saveBinary(intArray, tempFile);

        InputStream stream = new FileInputStream(tempFile);
        BufferedInputStream bis = new BufferedInputStream(stream);
        DataInputStream dis = new DataInputStream(bis);

        INDArray loaded = Nd4j.read(dis);

        assertEquals(DataType.INT, loaded.data().dataType());
        assertEquals(DataType.LONG, loaded.shapeInfoDataBuffer().dataType());

        assertEquals(intArray.data().length(), loaded.data().length());

        assertArrayEquals(intArray.data().asInt(), loaded.data().asInt());
    }

/*
    @Test(expected = ND4JIllegalStateException.class)
    public void testOpDiscarded() throws Exception {
        DataBuffer dataBuffer = Nd4j.createBuffer(new int[] {1, 2, 3, 4, 5});
        DataBuffer shapeBuffer = Nd4j.getShapeInfoProvider().createShapeInformation(new long[] {1, 5}, DataType.INT).getFirst();
        INDArray intArray = Nd4j.createArrayFromShapeBuffer(dataBuffer, shapeBuffer);

        intArray.add(10f);
    }
    */

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReallocation(Nd4jBackend backend) {
        DataBuffer buffer = Nd4j.createBuffer(new int[] {1, 2, 3, 4});
        assertEquals(4, buffer.capacity());
        buffer.reallocate(6);
        val old = buffer.asInt();
        assertEquals(6, buffer.capacity());
        val newContent = buffer.asInt();
        assertEquals(6, newContent.length);
        assertArrayEquals(old, Arrays.copyOf(newContent, old.length));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReallocationWorkspace(Nd4jBackend backend) {
        WorkspaceConfiguration initialConfig = WorkspaceConfiguration.builder().initialSize(10 * 1024L * 1024L)
                .policyAllocation(AllocationPolicy.STRICT).policyLearning(LearningPolicy.NONE).build();
        MemoryWorkspace workspace = Nd4j.getWorkspaceManager().getAndActivateWorkspace(initialConfig, "SOME_ID");

        DataBuffer buffer = Nd4j.createBuffer(new int[] {1, 2, 3, 4});
        val old = buffer.asInt();
        assertTrue(buffer.isAttached());
        assertEquals(4, buffer.capacity());
        buffer.reallocate(6);
        assertEquals(6, buffer.capacity());
        val newContent = buffer.asInt();
        assertEquals(6, newContent.length);
        assertArrayEquals(old, Arrays.copyOf(newContent, old.length));
        workspace.close();
    }

    @Override
    public char ordering() {
        return 'c';
    }
}

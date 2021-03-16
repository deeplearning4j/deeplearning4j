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

package org.nd4j.aeron.ipc.chunk;

import org.agrona.DirectBuffer;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.aeron.ipc.NDArrayMessage;
import org.nd4j.aeron.util.BufferUtil;
import org.nd4j.linalg.factory.Nd4j;

import javax.annotation.concurrent.NotThreadSafe;
import java.nio.ByteBuffer;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

@NotThreadSafe
@Disabled("Tests are too flaky")
public class NDArrayMessageChunkTests extends BaseND4JTest {

    @Test
    public void testChunkSerialization() {
        NDArrayMessage message = NDArrayMessage.wholeArrayUpdate(Nd4j.ones(1000));
        int chunkSize = 128;
        int numChunks = NDArrayMessage.numChunksForMessage(message, chunkSize);
        NDArrayMessageChunk[] chunks = NDArrayMessage.chunks(message, chunkSize);
        assertEquals(numChunks, chunks.length);
        for (int i = 1; i < numChunks; i++) {
            assertEquals(chunks[0].getMessageType(), chunks[i].getMessageType());
            assertEquals(chunks[0].getId(), chunks[i].getId());
            assertEquals(chunks[0].getChunkSize(), chunks[i].getChunkSize());
            assertEquals(chunks[0].getNumChunks(), chunks[i].getNumChunks());
        }

        ByteBuffer[] concat = new ByteBuffer[chunks.length];
        for (int i = 0; i < concat.length; i++)
            concat[i] = chunks[i].getData();


        DirectBuffer buffer = NDArrayMessage.toBuffer(message);
        //test equality of direct byte buffer contents vs chunked
        ByteBuffer byteBuffer = buffer.byteBuffer();
        ByteBuffer concatAll = BufferUtil.concat(concat, buffer.capacity());
        byte[] arrays = new byte[byteBuffer.capacity()];
        byteBuffer.rewind();
        byteBuffer.get(arrays);
        byte[] arrays2 = new byte[concatAll.capacity()];
        concatAll.rewind();
        concatAll.get(arrays2);
        assertArrayEquals(arrays, arrays2);
        NDArrayMessage message1 = NDArrayMessage.fromChunks(chunks);
        assertEquals(message, message1);

    }

}

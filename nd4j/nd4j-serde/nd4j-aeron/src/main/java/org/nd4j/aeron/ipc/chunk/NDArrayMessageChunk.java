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

package org.nd4j.aeron.ipc.chunk;

import lombok.Builder;
import lombok.Data;
import org.nd4j.aeron.ipc.AeronNDArraySubscriber;
import org.nd4j.aeron.ipc.NDArrayMessage;

import java.io.Serializable;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * An NDArrayMessageChunk
 * represents a chunked {@link NDArrayMessage}
 * that needs to be reassembled.
 *
 * This chunking is for large messages
 * that need to be segmented to be sent over the wire.
 *
 * An {@link AeronNDArraySubscriber}
 * would use this information to assemble a contiguous buffer
 * to be used for assembling a large {@link NDArrayMessage}
 *
 * Of note is that this chunk will also contain the needed data
 * for assembling in addition to the desired metadata such as the chunk size.
 *
 * A chunk has an idea which is used to track the chunk
 * across fragmentations, an index fr determining ordering
 * for re assembling, and metadata about the chunk such as
 * the chunk size and number of chunks
 *
 *
 *
 * @author Adam Gibson
 */
@Data
@Builder
public class NDArrayMessageChunk implements Serializable {
    //id of the chunk (meant for tracking for reassembling)
    private String id;
    //the chunk size (message size over the network)
    private int chunkSize;
    //the message opType, this should be chunked
    private NDArrayMessage.MessageType messageType;
    //the number of chunks for reassembling the message
    private int numChunks;
    //the index of this particular chunk
    private int chunkIndex;
    //the actual chunk data
    private ByteBuffer data;


    /**
     * Returns the overall size for an {@link NDArrayMessageChunk}.
     * The size of a message chunk is:
     * idLengthSize(4) + messageTypeSize(4) + indexSize(4) + chunkSizeSize(4) +  numChunksSize(4) + chunk.getData().limit() + chunk.getId().getBytes().length
     * Many of these are flat out integers and are mainly variables for accounting purposes and ease of readbility
     * @param chunk the size of a message chunk
     * @return the size of an {@link ByteBuffer} for the given {@link NDArrayMessageChunk}
     */
    public static int sizeForMessage(NDArrayMessageChunk chunk) {
        int messageTypeSize = 4;
        int indexSize = 4;
        int numChunksSize = 4;
        int chunkSizeSize = 4;
        int idLengthSize = 4;
        return idLengthSize + messageTypeSize + indexSize + chunkSizeSize + numChunksSize + chunk.getData().limit()
                        + chunk.getId().getBytes().length;

    }

    /**
     * Convert an ndarray message chunk to a buffer.
     * @param chunk the chunk to convert
     * @return an {@link ByteBuffer} based on the
     * passed in message chunk.
     */
    public static ByteBuffer toBuffer(NDArrayMessageChunk chunk) {
        ByteBuffer ret = ByteBuffer.allocateDirect(sizeForMessage(chunk)).order(ByteOrder.nativeOrder());
        //the messages opType enum as an int
        ret.putInt(chunk.getMessageType().ordinal());
        //the number of chunks this chunk is apart of
        ret.putInt(chunk.getNumChunks());
        //the chunk size
        ret.putInt(chunk.getChunkSize());
        //the length of the id (for self describing purposes)
        ret.putInt(chunk.getId().getBytes().length);
        // the actual id as a string
        ret.put(chunk.getId().getBytes());
        //the chunk index
        ret.putInt(chunk.getChunkIndex());
        //the actual data
        ret.put(chunk.getData());
        return ret;
    }

    /**
     * Returns a chunk given the passed in {@link ByteBuffer}
     * NOTE THAT THIS WILL MODIFY THE PASSED IN BYTEBUFFER's POSITION.
     *
     * @param byteBuffer the byte buffer to extract the chunk from
     * @return the ndarray message chunk based on the passed in {@link ByteBuffer}
     */
    public static NDArrayMessageChunk fromBuffer(ByteBuffer byteBuffer, NDArrayMessage.MessageType type) {
        int numChunks = byteBuffer.getInt();
        int chunkSize = byteBuffer.getInt();
        int idLength = byteBuffer.getInt();
        byte[] id = new byte[idLength];
        byteBuffer.get(id);
        String idString = new String(id);
        int index = byteBuffer.getInt();
        ByteBuffer firstData = byteBuffer.slice();
        NDArrayMessageChunk chunk = NDArrayMessageChunk.builder().chunkSize(chunkSize).numChunks(numChunks)
                        .data(firstData).messageType(type).id(idString).chunkIndex(index).build();
        return chunk;

    }

}

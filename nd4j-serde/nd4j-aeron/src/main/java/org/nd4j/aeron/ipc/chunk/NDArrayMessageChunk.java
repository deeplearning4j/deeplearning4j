package org.nd4j.aeron.ipc.chunk;

import lombok.Builder;
import lombok.Data;
import org.nd4j.aeron.ipc.AeronNDArraySubscriber;
import org.nd4j.aeron.ipc.NDArrayMessage;

import java.nio.ByteBuffer;

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
 * for assembling in addition to the desired metdata such as the chunk size.
 *
 *
 * @author Adam Gibson
 */
@Data
@Builder
public class NDArrayMessageChunk {
    private String id;
    private int chunkSize;
    private NDArrayMessage.MessageType messageType;
    private int numChunks;
    private int chunkIndex;
    private ByteBuffer data;


    /**
     * Returns the overall size for an {@link NDArrayMessageChunk}.
     * The size of a message chunk is:
     * messageTypeSize(4) + indexSize(4) + chunkSizeSize(4) +  numChunksSize(4)  + chunk.getData().capacity()
     * @param chunk the size of a message chunk
     * @return the size of an {@link ByteBuffer} for the given {@link NDArrayMessageChunk}
     */
    public static int sizeForMessage(NDArrayMessageChunk chunk) {
        int messageTypeSize = 4;
        int indexSize = 4;
        int numChunksSize = 4;
        int chunkSizeSize = 4;
        return messageTypeSize + indexSize + chunkSizeSize +  numChunksSize + chunk.getData().limit() + chunk.getId().getBytes().length;

    }

    /**
     * Convert an ndarray message chunk to a buffer.
     * @param chunk the chunk to convert
     * @return an {@link ByteBuffer} based on the
     * passed in message chunk.
     */
    public static ByteBuffer toBuffer(NDArrayMessageChunk chunk) {
        ByteBuffer ret = ByteBuffer.allocateDirect(sizeForMessage(chunk));
        ret.putInt(chunk.getMessageType().ordinal());
        ret.putInt(chunk.getNumChunks());
        ret.putInt(chunk.getChunkSize());
        ret.putInt(chunk.getId().getBytes().length);
        ret.putInt(chunk.getChunkIndex());
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
    public static NDArrayMessageChunk fromBuffer(ByteBuffer byteBuffer) {
        NDArrayMessage.MessageType type = NDArrayMessage.MessageType.values()[byteBuffer.getInt()];
        if(type != NDArrayMessage.MessageType.CHUNKED)
            throw new IllegalStateException("Messages must all be of type chunked");
        int numChunks = byteBuffer.getInt();
        int chunkSize = byteBuffer.getInt();
        int idLength = byteBuffer.getInt();
        byte[] id = new byte[idLength];
        byteBuffer.get(id);
        String idString = new String(id);
        int index = byteBuffer.getInt();
        ByteBuffer firstData = byteBuffer.get(new byte[chunkSize],index,chunkSize);
        NDArrayMessageChunk chunk = NDArrayMessageChunk.builder()
                .chunkSize(chunkSize).numChunks(numChunks).data(firstData)
                .messageType(type).id(idString).chunkIndex(index).build();
        return chunk;

    }

}

package org.nd4j.aeron.ipc;

import io.aeron.logbuffer.FragmentHandler;
import io.aeron.logbuffer.Header;
import lombok.extern.slf4j.Slf4j;
import org.agrona.DirectBuffer;
import org.nd4j.aeron.ipc.chunk.ChunkAccumulator;
import org.nd4j.aeron.ipc.chunk.InMemoryChunkAccumulator;
import org.nd4j.aeron.ipc.chunk.NDArrayMessageChunk;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;


/**
 * NDArray fragment handler
 * for listening to an aeron queue
 *
 * @author Adam Gibson
 */
@Slf4j
public class NDArrayFragmentHandler implements FragmentHandler {
    private NDArrayCallback ndArrayCallback;
    private ChunkAccumulator chunkAccumulator = new InMemoryChunkAccumulator();

    public NDArrayFragmentHandler(NDArrayCallback ndArrayCallback) {
        this.ndArrayCallback = ndArrayCallback;
    }

    /**
     * Callback for handling
     * fragments of data being read from a log.
     *
     * @param buffer containing the data.
     * @param offset at which the data begins.
     * @param length of the data in bytes.
     * @param header representing the meta data for the data.
     */
    @Override
    public void onFragment(DirectBuffer buffer, int offset, int length, Header header) {
        ByteBuffer byteBuffer = buffer.byteBuffer();
        boolean byteArrayInput = false;
        if (byteBuffer == null) {
            byteArrayInput = true;
            byte[] destination = new byte[length];
            ByteBuffer wrap = ByteBuffer.wrap(buffer.byteArray());
            wrap.get(destination, offset, length);
            byteBuffer = ByteBuffer.wrap(destination).order(ByteOrder.nativeOrder());
        }


        //only applicable for direct buffers where we don't wrap the array
        if (!byteArrayInput) {
            byteBuffer.position(offset);
            byteBuffer.order(ByteOrder.nativeOrder());
        }

        int messageTypeIndex = byteBuffer.getInt();
        if (messageTypeIndex >= NDArrayMessage.MessageType.values().length)
            throw new IllegalStateException(
                            "Illegal index on message opType. Likely corrupt message. Please check the serialization of the bytebuffer. Input was bytebuffer: "
                                            + byteArrayInput);
        NDArrayMessage.MessageType messageType = NDArrayMessage.MessageType.values()[messageTypeIndex];

        if (messageType == NDArrayMessage.MessageType.CHUNKED) {
            NDArrayMessageChunk chunk = NDArrayMessageChunk.fromBuffer(byteBuffer, messageType);
            if (chunk.getNumChunks() < 1)
                throw new IllegalStateException("Found invalid number of chunks " + chunk.getNumChunks()
                                + " on chunk index " + chunk.getChunkIndex());
            chunkAccumulator.accumulateChunk(chunk);
            log.info("Number of chunks " + chunk.getNumChunks() + " and number of chunks " + chunk.getNumChunks()
                            + " for id " + chunk.getId() + " is " + chunkAccumulator.numChunksSoFar(chunk.getId()));

            if (chunkAccumulator.allPresent(chunk.getId())) {
                NDArrayMessage message = chunkAccumulator.reassemble(chunk.getId());
                ndArrayCallback.onNDArrayMessage(message);
            }
        } else {
            NDArrayMessage message = NDArrayMessage.fromBuffer(buffer, offset);
            ndArrayCallback.onNDArrayMessage(message);
        }


    }
}

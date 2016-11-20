package org.nd4j.aeron.ipc;

import org.junit.Test;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;



import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 11/20/16.
 */
public class NDArrayMessageChunkTests {

    @Test
    public void testChunkSerialization() {
        NDArrayMessage message = NDArrayMessage.wholeArrayUpdate(Nd4j.ones(1000));
        int chunkSize = 128;
        int numChunks = NDArrayMessage.numChunksForMessage(message,chunkSize);
        NDArrayMessageChunk[] chunks = NDArrayMessage.chunks(message,chunkSize);
        assertEquals(numChunks,chunks.length);
        for(int i = 1; i < numChunks; i++) {
            assertEquals(chunks[0].getMessageType(),chunks[i].getMessageType());
            assertEquals(chunks[0].getId(),chunks[i].getId());
            assertEquals(chunks[0].getChunkSize(),chunks[i].getChunkSize());
            assertEquals(chunks[0].getNumChunks(),chunks[i].getNumChunks());
        }

        NDArrayMessage message1 = NDArrayMessage.fromChunks(chunks);
        assertEquals(message,message1);

    }

}

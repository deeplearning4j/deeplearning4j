package org.nd4j.aeron.ipc.chunk;

import org.junit.Test;
import org.nd4j.aeron.ipc.NDArrayMessage;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 11/20/16.
 */
public class ChunkAccumulatorTests {

    @Test
    public void testAccumulator() {
        ChunkAccumulator chunkAccumulator = new InMemoryChunkAccumulator();
        NDArrayMessage message = NDArrayMessage.wholeArrayUpdate(Nd4j.ones(1000));
        int chunkSize = 128;
        NDArrayMessageChunk[] chunks = NDArrayMessage.chunks(message, chunkSize);
        for (int i = 0; i < chunks.length; i++) {
            chunkAccumulator.accumulateChunk(chunks[i]);
        }

        NDArrayMessage message1 = chunkAccumulator.reassemble(chunks[0].getId());
        assertEquals(message, message1);
    }

}

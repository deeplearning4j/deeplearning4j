package org.nd4j.aeron.ipc.chunk;

import org.nd4j.aeron.ipc.NDArrayMessage;

import java.util.ArrayList;
import java.util.List;

/**
 * Accumulate {@link NDArrayMessageChunk} and reassemble them in to
 * {@link NDArrayMessage}.
 *
 * @author Aadm Gibson
 */
public interface ChunkAccumulator {
    /**
     * Reassemble an ndarray message
     * from a set of chunks
     *
     * Note that once reassemble is called,
     * the associated chunk lists will automatically
     * be removed from storage.
     *
     *
     * @param id the id to reassemble
     * @return the reassembled message
     */
    NDArrayMessage reassemble(String id);

    /**
     * Accumulate chunks
     * @param chunk the chunk to accumulate
     */
    void accumulateChunk(NDArrayMessageChunk chunk);
}

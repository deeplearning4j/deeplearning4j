package org.nd4j.aeron.ipc.chunk;

import com.google.common.collect.Maps;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.aeron.ipc.NDArrayMessage;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Accumulate chunks and reassemble them.
 *
 * @author Adam Gibson
 */
@Slf4j
public class InMemoryChunkAccumulator implements ChunkAccumulator {
    private Map<String,List<NDArrayMessageChunk>> chunks = Maps.newConcurrentMap();

    /**
     * Returns the number of chunks
     * accumulated for a given id so far
     *
     * @param id the id to get the
     *           number of chunks for
     * @return the number of chunks accumulated
     * for a given id so far
     */
    @Override
    public int numChunksSoFar(String id) {
        if(!chunks.containsKey(id))
            return 0;
        return chunks.get(id).size();
    }

    /**
     * Returns true if all chunks are present
     *
     * @param id the id to check for
     * @return true if all the chunks are present,false otherwise
     */
    @Override
    public boolean allPresent(String id) {
        if(!chunks.containsKey(id))
            return false;
        List<NDArrayMessageChunk> chunkList = chunks.get(id);
        return chunkList.size() == chunkList.get(0).getNumChunks();
    }

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
     public NDArrayMessage reassemble(String id) {
        List<NDArrayMessageChunk> chunkList = chunks.get(id);
        if(chunkList.size() != chunkList.get(0).getNumChunks())
            throw new IllegalStateException("Unable to reassmemble message chunk " + id + " missing " + (chunkList.get(0).getNumChunks() - chunkList.size()) + "chunks");
        NDArrayMessageChunk[] inOrder = new NDArrayMessageChunk[chunkList.size()];
        for(NDArrayMessageChunk chunk : chunkList) {
            inOrder[chunk.getChunkIndex()] = chunk;
        }


        NDArrayMessage message = NDArrayMessage.fromChunks(inOrder);
        chunkList.clear();
        chunks.remove(id);
        return message;

    }

    /**
     * Accumulate chunks
     * @param chunk
     */
    public void accumulateChunk(NDArrayMessageChunk chunk) {
        String id = chunk.getId();
        if(!chunks.containsKey(id)) {
            List<NDArrayMessageChunk> list = new ArrayList<>();
            list.add(chunk);
            chunks.put(id,list);
        }
        else {
            List<NDArrayMessageChunk> chunkList = chunks.get(id);
            chunkList.add(chunk);
        }

        log.debug("Accumulating chunk for id " + chunk.getId());


    }

}

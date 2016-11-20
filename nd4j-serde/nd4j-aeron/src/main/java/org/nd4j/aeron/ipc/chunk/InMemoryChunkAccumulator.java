package org.nd4j.aeron.ipc.chunk;

import com.google.common.collect.Maps;
import org.nd4j.aeron.ipc.NDArrayMessage;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Accumulate chunks and reassemble them.
 *
 * @author Adam Gibson
 */
public class InMemoryChunkAccumulator implements ChunkAccumulator {
    private Map<String,List<NDArrayMessageChunk>> chunks = Maps.newConcurrentMap();
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


    }

}

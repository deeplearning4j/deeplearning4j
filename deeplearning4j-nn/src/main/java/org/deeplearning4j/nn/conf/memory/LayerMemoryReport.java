package org.deeplearning4j.nn.conf.memory;

import lombok.NonNull;
import org.nd4j.linalg.api.buffer.DataBuffer;

/**
 * Created by Alex on 13/07/2017.
 */
public class LayerMemoryReport implements MemoryReport {
    @Override
    public long getTotalMemoryBytes(int minibatchSize) {
        return 0;
    }

    @Override
    public long getTotalMemoryBytes(int minibatchSize, @NonNull DataBuffer.Type dataType) {
        return 0;
    }

    @Override
    public long getMemoryBytes(MemoryType memoryType, int minibatchSize) {
        return 0;
    }

    @Override
    public long getMemoryBytes(MemoryType memoryType, int minibatchSize, DataBuffer.Type dataType) {
        return 0;
    }
}

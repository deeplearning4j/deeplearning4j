package org.deeplearning4j.nn.conf.memory;

import lombok.NonNull;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;

/**
 * Created by Alex on 13/07/2017.
 */
public abstract class MemoryReport {

    public long getTotalMemoryBytes(int minibatchSize){
        return getTotalMemoryBytes(minibatchSize, DataTypeUtil.getDtypeFromContext());
    }

    public abstract long getTotalMemoryBytes(int minibatchSize, @NonNull DataBuffer.Type dataType );

    public long getMemoryBytes(MemoryType memoryType, int minibatchSize ){
        return getMemoryBytes(memoryType, minibatchSize, DataTypeUtil.getDtypeFromContext());
    }

    public abstract long getMemoryBytes(MemoryType memoryType, int minibatchSize, DataBuffer.Type dataType );

    public abstract String toString();

}

package org.deeplearning4j.nn.conf.memory;

import lombok.NonNull;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;

/**
 * Created by Alex on 13/07/2017.
 */
public abstract class MemoryReport {

    /**
     *
     * @return Class that the memory report was generated for
     */
    public abstract Class<?> getReportClass();

    /**
     * Name of the object that the memory report was generated for
     * @return
     */
    public abstract String getName();

    public long getTotalMemoryBytes(int minibatchSize, MemoryUseMode memoryUseMode){
        return getTotalMemoryBytes(minibatchSize, memoryUseMode, DataTypeUtil.getDtypeFromContext());
    }

    public long getTotalMemoryBytes(int minibatchSize, MemoryUseMode memoryUseMode, @NonNull DataBuffer.Type dataType) {
        int bytesPerElement = getBytesPerElement(dataType);

        long totalBytes = 0;
        for(MemoryType mt : MemoryType.values()){
            totalBytes += getMemoryBytes(mt, minibatchSize, memoryUseMode, dataType);
        }

        return totalBytes;
    }

    public long getMemoryBytes(MemoryType memoryType, int minibatchSize, MemoryUseMode memoryUseMode ){
        return getMemoryBytes(memoryType, minibatchSize, memoryUseMode, DataTypeUtil.getDtypeFromContext());
    }

    public abstract long getMemoryBytes(MemoryType memoryType, int minibatchSize, MemoryUseMode memoryUseMode, DataBuffer.Type dataType );

    public abstract String toString();

    protected int getBytesPerElement(DataBuffer.Type dataType){
        switch (dataType){
            case DOUBLE:
                return 8;
            case FLOAT:
                return 4;
            case HALF:
                return 2;
            default:
                throw new UnsupportedOperationException("Data type not supported: " + dataType);
        }
    }

}

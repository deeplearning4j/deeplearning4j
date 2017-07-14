package org.deeplearning4j.nn.conf.memory;

import lombok.NonNull;
import org.deeplearning4j.nn.conf.CacheMode;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by Alex on 13/07/2017.
 */
public abstract class MemoryReport {

    public static final Map<CacheMode,Integer> CACHE_MODE_ALL_ZEROS = getAllZerosMap();

    private static Map<CacheMode,Integer> getAllZerosMap(){
        Map<CacheMode,Integer> map = new HashMap<>();
        for(CacheMode c : CacheMode.values()){
            map.put(c, 0);
        }

        return Collections.unmodifiableMap(map);
    }

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

    public long getTotalMemoryBytes(int minibatchSize, @NonNull MemoryUseMode memoryUseMode, @NonNull CacheMode cacheMode){
        return getTotalMemoryBytes(minibatchSize, memoryUseMode, cacheMode, DataTypeUtil.getDtypeFromContext());
    }

    public long getTotalMemoryBytes(int minibatchSize, @NonNull MemoryUseMode memoryUseMode, @NonNull CacheMode cacheMode,
                                    @NonNull DataBuffer.Type dataType) {
        int bytesPerElement = getBytesPerElement(dataType);

        long totalBytes = 0;
        for(MemoryType mt : MemoryType.values()){
            totalBytes += getMemoryBytes(mt, minibatchSize, memoryUseMode, cacheMode, dataType);
        }

        return totalBytes;
    }

    public long getMemoryBytes(MemoryType memoryType, int minibatchSize, MemoryUseMode memoryUseMode, CacheMode cacheMode ){
        return getMemoryBytes(memoryType, minibatchSize, memoryUseMode, cacheMode, DataTypeUtil.getDtypeFromContext());
    }

    public abstract long getMemoryBytes(MemoryType memoryType, int minibatchSize, MemoryUseMode memoryUseMode,
                                        CacheMode cacheMode, DataBuffer.Type dataType );

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

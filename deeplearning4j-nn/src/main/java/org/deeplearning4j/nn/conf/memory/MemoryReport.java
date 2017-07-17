package org.deeplearning4j.nn.conf.memory;

import lombok.NonNull;
import org.deeplearning4j.nn.conf.CacheMode;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * For the purposes of estimating memory use under different situations, we consider there to be 3 types of memory:<br>
 * Standard memory, working memory and Cached memory. Each type has the concept of 'fixed' size memory (independent
 * of minibatch size) and 'variable' memory (total use depends on minibatch size; memory reported is for one example).<br>
 * <br>
 * <br>
 * The following breakdown of memory types will be used:<br>
 * <ul>
 * <li>Standard memory</li>
 * <ul>
 * <li>Fixed size (parameters, parameter gradients, updater state)</li>
 * <li>Variable size (activations, activation gradients)</li>
 * </ul>
 * <li>Working memory (may be reused via workspace or garbage collected)</li>
 * <ul>
 * <li>Fixed size (may be different for train vs. inference)</li>
 * <li>Variable size (may be different for train vs. inference)</li>
 * </ul>
 * <li>Cached memory (only used for training mode)</li>
 * <ul>
 * <li>Fixed size (as a function of CacheMode)</li>
 * <li>Variable size (as a function of CacheMode)</li>
 * </ul>
 * </ul>
 * <br>
 * <br>
 * For MemoryUseMode (X = train or inference), for a given cache mode CM and minibatch size M and layers L:<br>
 * TotalMemory(X,CM,M) = sum_L ( StandardFixedMem(X) + M * StandardVariableMem(X) )<br>
 *  + max_L ( WorkingFixedMem(X) + M * WorkingVariableMem(X) )<br>
 *  + sum_L ( CachedFixedMem(X,CM) + M * CachedVariableMem(X,CM))<br>
 * <br>
 * Note 1: CachedFixedMem(inference,any) = 0 and CachedVariableMem(inference,any) = 0. i.e., cache is a train-only
 * feature.<br>
 * Note 2: Reported memory figures are given in NDArray size unit - thus 1 refers to 1 float or 1 double value,
 * depending on the data type setting.
 * <br>
 *
 * @author Alex Black
 */
public abstract class MemoryReport {

    public static final Map<CacheMode, Integer> CACHE_MODE_ALL_ZEROS = getAllZerosMap();

    private static Map<CacheMode, Integer> getAllZerosMap() {
        Map<CacheMode, Integer> map = new HashMap<>();
        for (CacheMode c : CacheMode.values()) {
            map.put(c, 0);
        }

        return Collections.unmodifiableMap(map);
    }

    /**
     * @return Class that the memory report was generated for
     */
    public abstract Class<?> getReportClass();

    /**
     * Name of the object that the memory report was generated for
     *
     * @return
     */
    public abstract String getName();

    public long getTotalMemoryBytes(int minibatchSize, @NonNull MemoryUseMode memoryUseMode, @NonNull CacheMode cacheMode) {
        return getTotalMemoryBytes(minibatchSize, memoryUseMode, cacheMode, DataTypeUtil.getDtypeFromContext());
    }

    public long getTotalMemoryBytes(int minibatchSize, @NonNull MemoryUseMode memoryUseMode, @NonNull CacheMode cacheMode,
                                    @NonNull DataBuffer.Type dataType) {
        long totalBytes = 0;
        for (MemoryType mt : MemoryType.values()) {
            totalBytes += getMemoryBytes(mt, minibatchSize, memoryUseMode, cacheMode, dataType);
        }

        return totalBytes;
    }

    public long getMemoryBytes(MemoryType memoryType, int minibatchSize, MemoryUseMode memoryUseMode, CacheMode cacheMode) {
        return getMemoryBytes(memoryType, minibatchSize, memoryUseMode, cacheMode, DataTypeUtil.getDtypeFromContext());
    }

    public abstract long getMemoryBytes(MemoryType memoryType, int minibatchSize, MemoryUseMode memoryUseMode,
                                        CacheMode cacheMode, DataBuffer.Type dataType);

    public abstract String toString();

    protected int getBytesPerElement(DataBuffer.Type dataType) {
        switch (dataType) {
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

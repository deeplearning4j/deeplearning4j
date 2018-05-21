package org.deeplearning4j.nn.conf.memory;

import lombok.EqualsAndHashCode;
import lombok.NonNull;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;
import org.nd4j.shade.jackson.core.JsonProcessingException;

import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * A MemoryReport is designed to represent the estimated memory usage of a model, as a function of:<br>
 * - Training vs. Inference usage of the network<br>
 * - Minibatch size<br>
 * - ND4J DataType setting<br>
 * - Cache mode<br>
 * Note that the memory use estimate may not be exact, as may not take into account all possible memory use;
 * Furthermore, memory may exceed this value depending on, for example, garbage collection.<br>
 * <br>
 * <br>
 * <br>
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
 *  + max_L ( WorkingFixedMem(X,CM) + M * WorkingVariableMem(X,CM) )<br>
 *  + sum_L ( CachedFixedMem(X,CM) + M * CachedVariableMem(X,CM))<br>
 * <br>
 * Note 1: CachedFixedMem(INFERENCE,any) = 0 and CachedVariableMem(INFERENCE,any) = 0. i.e., cache is a train-only
 * feature.<br>
 * Note 2: Working memory may depend on cache mode: if we cache something, we have less computation to do later, and
 *         hence less working memory.<br>
 * Note 3: Reported memory figures are given in NDArray size unit - thus 1 refers to 1 float or 1 double value,
 * depending on the data type setting.
 * <br>
 *
 * @author Alex Black
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY)
@EqualsAndHashCode
public abstract class MemoryReport {

    /**
     * A simple Map containing all zeros for each CacheMode key
     */
    public static final Map<CacheMode, Long> CACHE_MODE_ALL_ZEROS = getAllZerosMap();

    private static Map<CacheMode, Long> getAllZerosMap() {
        Map<CacheMode, Long> map = new HashMap<>();
        for (CacheMode c : CacheMode.values()) {
            map.put(c, 0L);
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
     * @return Name of the object
     */
    public abstract String getName();

    /**
     * Get the total memory use in bytes for the given configuration (using the current ND4J data type)
     *
     * @param minibatchSize Mini batch size to estimate the memory for
     * @param memoryUseMode The memory use mode (training or inference)
     * @param cacheMode     The CacheMode to use
     * @return The estimated total memory consumption in bytes
     */
    public long getTotalMemoryBytes(int minibatchSize, @NonNull MemoryUseMode memoryUseMode,
                    @NonNull CacheMode cacheMode) {
        return getTotalMemoryBytes(minibatchSize, memoryUseMode, cacheMode, DataTypeUtil.getDtypeFromContext());
    }

    /**
     * Get the total memory use in bytes for the given configuration
     *
     * @param minibatchSize Mini batch size to estimate the memory for
     * @param memoryUseMode The memory use mode (training or inference)
     * @param cacheMode     The CacheMode to use
     * @param dataType      Nd4j datatype
     * @return The estimated total memory consumption in bytes
     */
    public abstract long getTotalMemoryBytes(int minibatchSize, @NonNull MemoryUseMode memoryUseMode,
                    @NonNull CacheMode cacheMode, @NonNull DataBuffer.Type dataType);

    /**
     * Get the memory estimate (in bytes) for the specified type of memory, using the current ND4J data type
     *
     * @param memoryType    Type of memory to get the estimate for invites
     * @param minibatchSize Mini batch size to estimate the memory for
     * @param memoryUseMode The memory use mode (training or inference)
     * @param cacheMode     The CacheMode to use
     * @return              Estimated memory use for the given memory type
     */
    public long getMemoryBytes(MemoryType memoryType, int minibatchSize, MemoryUseMode memoryUseMode,
                    CacheMode cacheMode) {
        return getMemoryBytes(memoryType, minibatchSize, memoryUseMode, cacheMode, DataTypeUtil.getDtypeFromContext());
    }

    /**
     * Get the memory estimate (in bytes) for the specified type of memory
     *
     * @param memoryType    Type of memory to get the estimate for invites
     * @param minibatchSize Mini batch size to estimate the memory for
     * @param memoryUseMode The memory use mode (training or inference)
     * @param cacheMode     The CacheMode to use
     * @param dataType      Nd4j datatype
     * @return              Estimated memory use for the given memory type
     */
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

    /**
     * Get a map of CacheMode with all keys associated with the specified value
     *
     * @param value Value for all keys
     * @return Map
     */
    public static Map<CacheMode, Long> cacheModeMapFor(long value) {
        if (value == 0) {
            return CACHE_MODE_ALL_ZEROS;
        }
        Map<CacheMode, Long> m = new HashMap<>();
        for (CacheMode cm : CacheMode.values()) {
            m.put(cm, value);
        }
        return m;
    }

    public String toJson() {
        try {
            return NeuralNetConfiguration.mapper().writeValueAsString(this);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    public String toYaml() {
        try {
            return NeuralNetConfiguration.mapperYaml().writeValueAsString(this);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    public static MemoryReport fromJson(String json) {
        try {
            return NeuralNetConfiguration.mapper().readValue(json, MemoryReport.class);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static MemoryReport fromYaml(String yaml) {
        try {
            return NeuralNetConfiguration.mapperYaml().readValue(yaml, MemoryReport.class);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

}

/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.conf.memory;

import lombok.EqualsAndHashCode;
import lombok.NonNull;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;
import org.nd4j.shade.jackson.core.JsonProcessingException;

import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

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
                    @NonNull CacheMode cacheMode, @NonNull DataType dataType);

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
                    CacheMode cacheMode, DataType dataType);

    public abstract String toString();

    protected int getBytesPerElement(DataType dataType) {
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

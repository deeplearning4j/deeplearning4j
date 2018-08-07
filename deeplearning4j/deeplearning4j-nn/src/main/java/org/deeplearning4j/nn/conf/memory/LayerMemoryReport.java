/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.nn.conf.memory;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.nd4j.linalg.api.buffer.DataBuffer;

import java.util.HashMap;
import java.util.Map;

/**
 * A {@link MemoryReport} Designed to report estimated memory use for a single layer or graph vertex.
 *
 * @author Alex Black
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
public class LayerMemoryReport extends MemoryReport {

    private String layerName;
    private Class<?> layerType;
    private InputType inputType;
    private InputType outputType;

    //Standard memory (in terms of total ND4J array length)
    private long parameterSize;
    private long updaterStateSize;

    //Working memory (in ND4J array length)
    //Note that *working* memory may be reduced by caching (which is only used during train mode)
    private long workingMemoryFixedInference;
    private long workingMemoryVariableInference;
    private Map<CacheMode, Long> workingMemoryFixedTrain;
    private Map<CacheMode, Long> workingMemoryVariableTrain;

    //Cache memory, by cache mode:
    Map<CacheMode, Long> cacheModeMemFixed;
    Map<CacheMode, Long> cacheModeMemVariablePerEx;

    protected LayerMemoryReport(Builder b) {
        this.layerName = b.layerName;
        this.layerType = b.layerType;
        this.inputType = b.inputType;
        this.outputType = b.outputType;

        this.parameterSize = b.parameterSize;
        this.updaterStateSize = b.updaterStateSize;

        this.workingMemoryFixedInference = b.workingMemoryFixedInference;
        this.workingMemoryVariableInference = b.workingMemoryVariableInference;
        this.workingMemoryFixedTrain = b.workingMemoryFixedTrain;
        this.workingMemoryVariableTrain = b.workingMemoryVariableTrain;

        this.cacheModeMemFixed = b.cacheModeMemFixed;
        this.cacheModeMemVariablePerEx = b.cacheModeMemVariablePerEx;
    }

    @Override
    public Class<?> getReportClass() {
        return layerType;
    }

    @Override
    public String getName() {
        return layerName;
    }

    @Override
    public long getTotalMemoryBytes(int minibatchSize, @NonNull MemoryUseMode memoryUseMode,
                    @NonNull CacheMode cacheMode, @NonNull DataBuffer.Type dataType) {
        long total = 0;
        for (MemoryType mt : MemoryType.values()) {
            total += getMemoryBytes(mt, minibatchSize, memoryUseMode, cacheMode, dataType);
        }
        return total;
    }

    @Override
    public long getMemoryBytes(MemoryType memoryType, int minibatchSize, MemoryUseMode memoryUseMode,
                    CacheMode cacheMode, DataBuffer.Type dataType) {
        int bytesPerElement = getBytesPerElement(dataType);
        switch (memoryType) {
            case PARAMETERS:
                return parameterSize * bytesPerElement;
            case PARAMATER_GRADIENTS:
                if (memoryUseMode == MemoryUseMode.INFERENCE) {
                    return 0;
                }
                return parameterSize * bytesPerElement;
            case ACTIVATIONS:
                return minibatchSize * outputType.arrayElementsPerExample() * bytesPerElement;
            case ACTIVATION_GRADIENTS:
                if (memoryUseMode == MemoryUseMode.INFERENCE) {
                    return 0;
                }
                //Activation gradients produced by this layer: epsilons to layer below -> equal to input size
                return minibatchSize * inputType.arrayElementsPerExample() * bytesPerElement;
            case UPDATER_STATE:
                if (memoryUseMode == MemoryUseMode.INFERENCE) {
                    return 0;
                }
                return updaterStateSize * bytesPerElement;
            case WORKING_MEMORY_FIXED:
                if (memoryUseMode == MemoryUseMode.INFERENCE) {
                    return workingMemoryFixedInference * bytesPerElement;
                } else {
                    return workingMemoryFixedTrain.get(cacheMode) * bytesPerElement;
                }
            case WORKING_MEMORY_VARIABLE:
                if (memoryUseMode == MemoryUseMode.INFERENCE) {
                    return workingMemoryVariableInference * bytesPerElement;
                } else {
                    return minibatchSize * workingMemoryVariableTrain.get(cacheMode) * bytesPerElement;
                }
            case CACHED_MEMORY_FIXED:
                if (memoryUseMode == MemoryUseMode.INFERENCE) {
                    return 0;
                } else {
                    return cacheModeMemFixed.get(cacheMode) * bytesPerElement;
                }
            case CACHED_MEMORY_VARIABLE:
                if (memoryUseMode == MemoryUseMode.INFERENCE) {
                    return 0;
                } else {
                    return minibatchSize * cacheModeMemVariablePerEx.get(cacheMode) * bytesPerElement;
                }
            default:
                throw new IllegalStateException("Unknown memory type: " + memoryType);
        }
    }

    @Override
    public String toString() {
        return "LayerMemoryReport(layerName=" + layerName + ",layerType=" + layerType.getSimpleName() + ")";
    }

    /**
     * Multiply all memory usage by the specified scaling factor
     *
     * @param scale Scale factor to multiply all memory usage by
     */
    public void scale(int scale){
        parameterSize *= scale;
        updaterStateSize *= scale;
        workingMemoryFixedInference *= scale;
        workingMemoryVariableInference *= scale;
        cacheModeMemFixed = scaleEntries(cacheModeMemFixed, scale);
        cacheModeMemVariablePerEx = scaleEntries(cacheModeMemVariablePerEx, scale);
    }

    private static Map<CacheMode,Long> scaleEntries(Map<CacheMode, Long> in, int scale){
        if(in == null)
            return null;

        Map<CacheMode,Long> out = new HashMap<>();
        for(Map.Entry<CacheMode,Long> e : in.entrySet()){
            out.put(e.getKey(), scale * e.getValue());
        }

        return out;
    }

    public static class Builder {

        private String layerName;
        private Class<?> layerType;
        private InputType inputType;
        private InputType outputType;

        //Standard memory (in terms of total ND4J array length)
        private long parameterSize;
        private long updaterStateSize;

        //Working memory (in ND4J array length)
        //Note that *working* memory may be reduced by caching (which is only used during train mode)
        private long workingMemoryFixedInference;
        private long workingMemoryVariableInference;
        private Map<CacheMode, Long> workingMemoryFixedTrain;
        private Map<CacheMode, Long> workingMemoryVariableTrain;

        //Cache memory, by cache mode:
        Map<CacheMode, Long> cacheModeMemFixed;
        Map<CacheMode, Long> cacheModeMemVariablePerEx;

        /**
         *
         * @param layerName  Name of the layer or graph vertex
         * @param layerType  Type of the layer or graph vertex
         * @param inputType  Input type to the layer/vertex
         * @param outputType Output type from the layer/vertex
         */
        public Builder(String layerName, Class<?> layerType, InputType inputType, InputType outputType) {
            this.layerName = layerName;
            this.layerType = layerType;
            this.inputType = inputType;
            this.outputType = outputType;
        }

        /**
         * Report the standard memory
         *
         * @param parameterSize    Number of parameters
         * @param updaterStateSize Size for the updater array
         */
        public Builder standardMemory(long parameterSize, long updaterStateSize) {
            this.parameterSize = parameterSize;
            this.updaterStateSize = updaterStateSize;
            return this;
        }

        /**
         * Report the working memory size, for both inference and training
         *
         * @param fixedInference         Number of elements used for inference ( independent of minibatch size)
         * @param variableInferencePerEx Number of elements used for inference, for each example
         * @param fixedTrain             Number of elements used for training (independent of minibatch size)
         * @param variableTrainPerEx     Number of elements used for training, for each example
         */
        public Builder workingMemory(long fixedInference, long variableInferencePerEx, long fixedTrain,
                        long variableTrainPerEx) {
            return workingMemory(fixedInference, variableInferencePerEx, MemoryReport.cacheModeMapFor(fixedTrain),
                            MemoryReport.cacheModeMapFor(variableTrainPerEx));
        }

        /**
         * Report the working memory requirements, for both inference and training. As noted in {@link MemoryReport}
         * Working memory is memory That will be allocated in a ND4J workspace, or can be garbage collected at any
         * points after the method returns.
         *
         * @param fixedInference         Number of elements of working memory used for inference (independent of minibatch size)
         * @param variableInferencePerEx Number of elements of working memory used for inference, for each example
         * @param fixedTrain             Number of elements of working memory used for training (independent of
         *                               minibatch size), for each cache mode
         * @param variableTrainPerEx     Number of elements of working memory used for training, for each example, for
         *                               each cache mode
         */
        public Builder workingMemory(long fixedInference, long variableInferencePerEx, Map<CacheMode, Long> fixedTrain,
                        Map<CacheMode, Long> variableTrainPerEx) {
            this.workingMemoryFixedInference = fixedInference;
            this.workingMemoryVariableInference = variableInferencePerEx;
            this.workingMemoryFixedTrain = fixedTrain;
            this.workingMemoryVariableTrain = variableTrainPerEx;
            return this;
        }

        /**
         * Reports the cached/cacheable memory requirements. This method assumes the caseload memory is the same for
         * all cases, i.e., typically used with zeros (Layers that do not use caching)
         *
         *
         * @param cacheModeMemoryFixed         Number of elements of cache memory, independent of the mini batch size
         * @param cacheModeMemoryVariablePerEx Number of elements of cache memory, for each example
         */
        public Builder cacheMemory(long cacheModeMemoryFixed, long cacheModeMemoryVariablePerEx) {
            return cacheMemory(MemoryReport.cacheModeMapFor(cacheModeMemoryFixed),
                            MemoryReport.cacheModeMapFor(cacheModeMemoryVariablePerEx));
        }

        /**
         * Reports the cached/cacheable memory requirements.
         *
         * @param cacheModeMemoryFixed         Number of elements of cache memory, independent of the mini batch size
         * @param cacheModeMemoryVariablePerEx Number of elements of cache memory, for each example
         */
        public Builder cacheMemory(Map<CacheMode, Long> cacheModeMemoryFixed,
                        Map<CacheMode, Long> cacheModeMemoryVariablePerEx) {
            this.cacheModeMemFixed = cacheModeMemoryFixed;
            this.cacheModeMemVariablePerEx = cacheModeMemoryVariablePerEx;
            return this;
        }

        public LayerMemoryReport build() {
            return new LayerMemoryReport(this);
        }
    }
}

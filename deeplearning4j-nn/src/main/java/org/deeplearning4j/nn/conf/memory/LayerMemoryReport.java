package org.deeplearning4j.nn.conf.memory;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.nd4j.linalg.api.buffer.DataBuffer;

import java.util.Map;

/**
 * Created by Alex on 13/07/2017.
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
    private Map<CacheMode,Long> workingMemoryFixedTrain;
    private Map<CacheMode,Long> workingMemoryVariableTrain;

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
    public long getMemoryBytes(MemoryType memoryType, int minibatchSize, MemoryUseMode memoryUseMode, CacheMode cacheMode, DataBuffer.Type dataType) {
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
        //TODO
        return null;
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
        private Map<CacheMode,Long> workingMemoryFixedTrain;
        private Map<CacheMode,Long> workingMemoryVariableTrain;

        //Cache memory, by cache mode:
        Map<CacheMode, Long> cacheModeMemFixed;
        Map<CacheMode, Long> cacheModeMemVariablePerEx;


        public Builder(String layerName, Class<?> layerType, InputType inputType, InputType outputType) {
            this.layerName = layerName;
            this.layerType = layerType;
            this.inputType = inputType;
            this.outputType = outputType;
        }

        public Builder standardMemory(long parameterSize, long updaterStateSize) {
            this.parameterSize = parameterSize;
            this.updaterStateSize = updaterStateSize;
            return this;
        }

        public Builder workingMemory(long fixedInference, long variableInferencePerEx, long fixedTrain, long variableTrainPerEx) {
            return workingMemory(fixedInference, variableInferencePerEx, MemoryReport.cacheModeMapFor(fixedInference), MemoryReport.cacheModeMapFor(variableInferencePerEx) );
        }

        public Builder workingMemory(long fixedInference, long variableInferencePerEx, Map<CacheMode, Long> fixedTrain, Map<CacheMode, Long> variableTrainPerEx) {
            this.workingMemoryFixedInference = fixedInference;
            this.workingMemoryVariableInference = variableInferencePerEx;
            this.workingMemoryFixedTrain = fixedTrain;
            this.workingMemoryVariableTrain = variableTrainPerEx;
            return this;
        }

        public Builder cacheMemory(Map<CacheMode, Long> cacheModeMemoryFixed, Map<CacheMode, Long> cacheModeMemoryVariablePerEx) {
            this.cacheModeMemFixed = cacheModeMemoryFixed;
            this.cacheModeMemVariablePerEx = cacheModeMemoryVariablePerEx;
            return this;
        }

        public LayerMemoryReport build() {
            return new LayerMemoryReport(this);
        }
    }
}

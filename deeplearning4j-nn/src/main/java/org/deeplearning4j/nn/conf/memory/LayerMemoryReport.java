package org.deeplearning4j.nn.conf.memory;

import lombok.Getter;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.nd4j.linalg.api.buffer.DataBuffer;

import java.util.Map;

/**
 * Created by Alex on 13/07/2017.
 */
@Getter
public class LayerMemoryReport extends MemoryReport {

    private String layerName;
    private Class<?> layerType;
    private InputType inputType;
    private InputType outputType;

    //Standard memory (in terms of total ND4J array length)
    private long parameterSize;
    private long activationSizePerEx;
    private long updaterStateSize;

    //Working memory (in ND4J array length)
    private long workinMemoryFixedInference;
    private long workingMemoryVariableInference;
    private long workinMemoryFixedTrain;
    private long workingMemoryVariableTrain;

    //Cache memory, by cache mode:
    Map<CacheMode,Long> cacheModeMemFixed;
    Map<CacheMode,Long> cacheModeMemVariablePerEx;

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
        switch (memoryType){
            case PARAMETERS:
                return parameterSize * bytesPerElement;
            case PARAMATER_GRADIENTS:
                if(memoryUseMode == MemoryUseMode.INFERENCE){
                    return 0;
                }
                return parameterSize * bytesPerElement;
            case ACTIVATIONS:
                return minibatchSize * activationSizePerEx * bytesPerElement;
            case ACTIVATION_GRADIENTS:
                if(memoryUseMode == MemoryUseMode.INFERENCE){
                    return 0;
                }
                return minibatchSize * activationSizePerEx * bytesPerElement;
            case UPDATER_STATE:
                if(memoryUseMode == MemoryUseMode.INFERENCE){
                    return 0;
                }
                return updaterStateSize * bytesPerElement;
            case INFERENCE_WORKING_MEM:
                return minibatchSize * inferenceWorkingSizePerEx * bytesPerElement;
            case TRAINING_WORKING_MEM:
                int totalPerEx = trainingWorkingSizePerEx.get(cacheMode) + trainingWorkingSizeCachedPerEx.get(cacheMode);
                return minibatchSize * totalPerEx * bytesPerElement;
            default:
                throw new IllegalStateException("Unknown memory type: " + memoryType);
        }
    }

    @Override
    public String toString() {
        return null;
    }


    public static class Builder {

        private String layerName;
        private Class<?> layerType;
        private InputType inputType;
        private InputType outputType;

        //Standard memory (in terms of total ND4J array length)
        private long parameterSize;
        private long activationSizePerEx;
        private long updaterStateSize;

        //Working memory (in ND4J array length)
        private long workingMemoryFixedInference;
        private long workingMemoryVariableInference;
        private long workingMemoryFixedTrain;
        private long workingMemoryVariableTrain;

        //Cache memory, by cache mode:
        Map<CacheMode,Long> cacheModeMemFixed;
        Map<CacheMode,Long> cacheModeMemVariablePerEx;


        public Builder(String layerName, Class<?> layerType, InputType inputType, InputType outputType){
            this.layerName = layerName;
            this.layerType = layerType;
            this.inputType = inputType;
            this.outputType = outputType;
        }

        public Builder standardMemory(long parameterSize, long updaterStateSizePerEx, long activationSizePerEx ){
            this.parameterSize = parameterSize;
            this.updaterStateSize = updaterStateSizePerEx;
            this.activationSizePerEx = activationSizePerEx;
            return this;
        }

        public Builder workingMemory(long fixedInference, long variableInferencePerEx, long fixedTrain, long variableTrainPerEx){
            this.workingMemoryFixedInference = fixedInference;
            this.workingMemoryVariableInference = variableInferencePerEx;
            this.workingMemoryFixedTrain = fixedTrain;
            this.workingMemoryVariableTrain = variableTrainPerEx;
            return this;
        }

        public Builder cacheMemory(Map<CacheMode,Long> cacheModeMemoryFixed, Map<CacheMode,Long> cacheModeMemoryVariablePerEx){
            this.cacheModeMemFixed = cacheModeMemoryFixed;
            this.cacheModeMemVariablePerEx = cacheModeMemoryVariablePerEx;
            return this;
        }
    }
}

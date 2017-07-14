package org.deeplearning4j.nn.conf.memory;

import lombok.Builder;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.nd4j.linalg.api.buffer.DataBuffer;

import java.util.Map;

/**
 * Created by Alex on 13/07/2017.
 */
public class LayerMemoryReport extends MemoryReport {

    private final String layerName;
    private final Class<?> layerType;
    private final InputType inputType;
    private final InputType outputType;
    private final int parameterSize;
    private final int activationSizePerEx;
    private final int updaterStateSize;
    //Assume no cache is used for inference
    private final int inferenceWorkingSizePerEx;
    //Train working memory (ex. activations, etc) that is either GC'd or in workspace
    private final Map<CacheMode,Integer> trainingWorkingSizePerEx;
    //Train working memory (ex. activations, etc) that is cached, hence cannot be GC'd or reused via workspace
    //Note that this is in ADDITION to the non-cached memory
    private final Map<CacheMode,Integer> trainingWorkingSizeCachedPerEx;

    @Builder
    public LayerMemoryReport(String layerName, Class<?> layerType, InputType inputType, InputType outputType, int parameterSize,
                             int activationSizePerEx, int updaterStateSize, int inferenceWorkingSizePerEx,
                             Map<CacheMode,Integer> trainingWorkingSizePerEx, Map<CacheMode, Integer> trainingWorkingSizeCachedPerEx){
        this.layerName = layerName;
        this.layerType = layerType;
        this.inputType = inputType;
        this.outputType = outputType;
        this.parameterSize = parameterSize;
        this.activationSizePerEx = activationSizePerEx;
        this.updaterStateSize = updaterStateSize;
        this.inferenceWorkingSizePerEx = inferenceWorkingSizePerEx;
        this.trainingWorkingSizePerEx = trainingWorkingSizePerEx;
        this.trainingWorkingSizeCachedPerEx = trainingWorkingSizeCachedPerEx;
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
}

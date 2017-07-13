package org.deeplearning4j.nn.conf.memory;

import lombok.Builder;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.nd4j.linalg.api.buffer.DataBuffer;

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
    private final int fwdPassWorkingSize;
    private final int backwardPassWorkingSize;

    @Builder
    public LayerMemoryReport(String layerName, Class<?> layerType, InputType inputType, InputType outputType, int parameterSize,
                             int activationSizePerEx, int updaterStateSize, int fwdPassWorkingSize, int backwardPassWorkingSize){
        this.layerName = layerName;
        this.layerType = layerType;
        this.inputType = inputType;
        this.outputType = outputType;
        this.parameterSize = parameterSize;
        this.activationSizePerEx = activationSizePerEx;
        this.updaterStateSize = updaterStateSize;
        this.fwdPassWorkingSize = fwdPassWorkingSize;
        this.backwardPassWorkingSize = backwardPassWorkingSize;
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
    public long getMemoryBytes(MemoryType memoryType, int minibatchSize, MemoryUseMode memoryUseMode, DataBuffer.Type dataType) {
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
            case FORWARD_PASS_WORKING_MEM:
                return fwdPassWorkingSize * bytesPerElement;
            case BACKWARD_PASS_WORKING_MEM:
                return backwardPassWorkingSize * bytesPerElement;
            default:
                throw new IllegalStateException("Unknown memory type: " + memoryType);
        }
    }

    @Override
    public String toString() {
        return null;
    }
}

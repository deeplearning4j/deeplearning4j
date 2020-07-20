package org.nd4j.linalg.api.ops.impl.layers.recurrent.outputs;

import lombok.Getter;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMLayer;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMDataFormat;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMDirectionMode;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMLayerConfig;

/**
 * The outputs of a LSTM layer ({@link LSTMLayer}.
 */
@Getter
public class LSTMLayerOutputs {

    /**
     * The LSTM layer data format ({@link LSTMDataFormat}.
     */
    private LSTMDataFormat dataFormat;


    /**
     * output h:
     * [sL, bS, nOut]    when directionMode <= 2 && dataFormat == 0
     * [bS, sL, nOut]    when directionMode <= 2 && dataFormat == 1
     * [bS, nOut, sL]    when directionMode <= 2 && dataFormat == 2
     * [sL, bS, 2*nOut]  when directionMode == 3 && dataFormat == 0
     * [bS, sL, 2*nOut]  when directionMode == 3 && dataFormat == 1
     * [bS, 2*nOut, sL]  when directionMode == 3 && dataFormat == 2
     * [sL, 2, bS, nOut] when directionMode == 4 && dataFormat == 3
     * numbers mean index in corresponding enums {@link LSTMDataFormat} and {@link LSTMDirectionMode}
     */
    private SDVariable timeSeriesOutput;

    /**
     * cell state at last step cL:
     * [bS, nOut]   when directionMode FWD or BWD
     * 2, bS, nOut] when directionMode  BIDIR_SUM, BIDIR_CONCAT or BIDIR_EXTRA_DIM
     */
    private SDVariable lastCellStateOutput;

    /**
     * output at last step hL:
     * [bS, nOut]   when directionMode FWD or BWD
     * 2, bS, nOut] when directionMode  BIDIR_SUM, BIDIR_CONCAT or BIDIR_EXTRA_DIM
     */
    private SDVariable lastTimeStepOutput;


    public LSTMLayerOutputs(SDVariable[] outputs, LSTMLayerConfig lstmLayerConfig) {
        Preconditions.checkArgument(outputs.length > 0 && outputs.length <= 3,
                "Must have from 1 to 3 LSTM layer outputs, got %s", outputs.length);

        int i = 0;
        timeSeriesOutput = lstmLayerConfig.isRetFullSequence() ? outputs[i++] : null;
        lastTimeStepOutput = lstmLayerConfig.isRetLastH() ? outputs[i++] : null;
        lastCellStateOutput = lstmLayerConfig.isRetLastC() ? outputs[i++] : null;


        this.dataFormat = lstmLayerConfig.getLstmdataformat();
    }


    /**
     * Get h, the output of the cell for all time steps.
     * <p>
     * Shape depends on data format defined in {@link LSTMLayerConfig }:<br>
     * for unidirectional:
     * TNS: shape [timeLength, numExamples, inOutSize] - sometimes referred to as "time major"<br>
     * NST: shape [numExamples, inOutSize, timeLength]<br>
     * NTS: shape [numExamples, timeLength, inOutSize] <br>
     * for bidirectional:
     * T2NS: 3 = [timeLength, 2, numExamples, inOutSize] (for ONNX)
     */
    public SDVariable getOutput() {
        Preconditions.checkArgument(timeSeriesOutput != null, "retFullSequence was setted as false in LSTMLayerConfig");
        return timeSeriesOutput;
    }

    public SDVariable getLastState() {
        Preconditions.checkArgument(lastCellStateOutput != null, "retLastC was setted as false in LSTMLayerConfig");
        return lastCellStateOutput;
    }

    public SDVariable getLastOutput() {
        Preconditions.checkArgument(lastTimeStepOutput != null, "retLastH was setted as false in LSTMLayerConfig");
        return lastTimeStepOutput;
    }

}

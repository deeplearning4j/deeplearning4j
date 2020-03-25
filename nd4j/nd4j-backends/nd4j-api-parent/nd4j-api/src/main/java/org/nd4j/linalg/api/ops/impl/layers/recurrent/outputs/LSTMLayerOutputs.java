package org.nd4j.linalg.api.ops.impl.layers.recurrent.outputs;

import java.util.Arrays;
import java.util.List;
import lombok.AccessLevel;
import lombok.Getter;
import org.nd4j.autodiff.samediff.SDIndex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMLayer;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMDataFormat;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.RnnDataFormat;

/**
 * The outputs of a LSTM layer ({@link LSTMLayer}.
 */
@Getter
public class LSTMLayerOutputs {

    private LSTMDataFormat dataFormat;

    /**
     * Output - input modulation gate activations.
     * Shape depends on data format (in layer config):<br>
     * TNS -> [timeSteps, batchSize, numUnits]<br>
     * NST -> [batchSize, numUnits, timeSteps]<br>
     * NTS -> [batchSize, timeSteps, numUnits]<br>
     */
    private SDVariable i;

    /**
     * Activations, cell state (pre tanh).
     * Shape depends on data format (in layer config):<br>
     * TNS -> [timeSteps, batchSize, numUnits]<br>
     * NST -> [batchSize, numUnits, timeSteps]<br>
     * NTS -> [batchSize, timeSteps, numUnits]<br>
     */
    private SDVariable c;

    /**
     * Output - forget gate activations.
     * Shape depends on data format (in layer config):<br>
     * TNS -> [timeSteps, batchSize, numUnits]<br>
     * NST -> [batchSize, numUnits, timeSteps]<br>
     * NTS -> [batchSize, timeSteps, numUnits]<br>
     */
    private SDVariable f;

    /**
     * Output - output gate activations.
     * Shape depends on data format (in layer config):<br>
     * TNS -> [timeSteps, batchSize, numUnits]<br>
     * NST -> [batchSize, numUnits, timeSteps]<br>
     * NTS -> [batchSize, timeSteps, numUnits]<br>
     */
    private SDVariable o;

    /**
     * Output - input gate activations.
     * Shape depends on data format (in layer config):<br>
     * TNS -> [timeSteps, batchSize, numUnits]<br>
     * NST -> [batchSize, numUnits, timeSteps]<br>
     * NTS -> [batchSize, timeSteps, numUnits]<br>
     */
    private SDVariable z;

    /**
     * Cell state, post tanh.
     * Shape depends on data format (in layer config):<br>
     * TNS -> [timeSteps, batchSize, numUnits]<br>
     * NST -> [batchSize, numUnits, timeSteps]<br>
     * NTS -> [batchSize, timeSteps, numUnits]<br>
     */
    private SDVariable h;

    /**
     * Current cell output.
     * Shape depends on data format (in layer config):<br>
     * TNS -> [timeSteps, batchSize, numUnits]<br>
     * NST -> [batchSize, numUnits, timeSteps]<br>
     * NTS -> [batchSize, timeSteps, numUnits]<br>
     */
    private SDVariable y;

    public LSTMLayerOutputs(SDVariable[] outputs, LSTMDataFormat dataFormat){
        Preconditions.checkArgument(outputs.length == 7,
                "Must have 7 LSTM layer outputs, got %s", outputs.length);

        i = outputs[0];
        c = outputs[1];
        f = outputs[2];
        o = outputs[3];
        z = outputs[4];
        h = outputs[5];
        y = outputs[6];
        this.dataFormat = dataFormat;
    }

    /**
     * Get all outputs returned by the cell.
     */
    public List<SDVariable> getAllOutputs(){
        return Arrays.asList(i, c, f, o, z, h, y);
    }

    /**
     * Get y, the output of the cell for all time steps.
     * 
     * Shape depends on data format (in layer config):<br>
     * TNS -> [timeSteps, batchSize, numUnits]<br>
     * NST -> [batchSize, numUnits, timeSteps]<br>
     * NTS -> [batchSize, timeSteps, numUnits]<br>
     */
    public SDVariable getOutput(){
        return y;
    }

    /**
     * Get c, the cell's state for all time steps.
     *
     * Shape depends on data format (in layer config):<br>
     * TNS -> [timeSteps, batchSize, numUnits]<br>
     * NST -> [batchSize, numUnits, timeSteps]<br>
     * NTS -> [batchSize, timeSteps, numUnits]<br>
     */
    public SDVariable getState(){
        return c;
    }

    private SDVariable lastOutput = null;

    /**
     * Get y, the output of the cell, for the last time step.
     *
     * Has shape [batchSize, numUnits].
     */
    public SDVariable getLastOutput(){
        if(lastOutput != null)
            return lastOutput;

        switch (dataFormat){
            case TNS:
                lastOutput = getOutput().get(SDIndex.point(-1), SDIndex.all(), SDIndex.all());
                break;
            case NST:
                lastOutput = getOutput().get(SDIndex.all(), SDIndex.all(), SDIndex.point(-1));
                break;
            case NTS:
                lastOutput = getOutput().get(SDIndex.all(), SDIndex.point(-1), SDIndex.all());
                break;
        }
        return lastOutput;
    }

    private SDVariable lastState = null;

    /**
     * Get c, the state of the cell, for the last time step.
     *
     * Has shape [batchSize, numUnits].
     */
    public SDVariable getLastState(){
        if(lastState != null)
            return lastState;

        switch (dataFormat){
            case TNS:
                lastState = getState().get(SDIndex.point(-1), SDIndex.all(), SDIndex.all());
                break;
            case NST:
                lastState = getState().get(SDIndex.all(), SDIndex.all(), SDIndex.point(-1));
                break;
            case NTS:
                lastState = getState().get(SDIndex.all(), SDIndex.point(-1), SDIndex.all());
                break;
        }
        return lastState;
    }


}

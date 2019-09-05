package org.nd4j.linalg.api.ops.impl.layers.recurrent.outputs;

import java.util.Arrays;
import java.util.List;
import lombok.Getter;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.GRUCell;

/**
 * The outputs of a GRU cell ({@link GRUCell}.
 */
@Getter
public class SRUCellOutputs {


    /**
     * Current cell output [batchSize, numUnits].
     */
    private SDVariable h;

    /**
     * Current cell state [batchSize, numUnits].
     */
    private SDVariable c;

    public SRUCellOutputs(SDVariable[] outputs){
        Preconditions.checkArgument(outputs.length == 2,
                "Must have 2 SRU cell outputs, got %s", outputs.length);

        h = outputs[0];
        c = outputs[1];
    }

    /**
     * Get all outputs returned by the cell.
     */
    public List<SDVariable> getAllOutputs(){
        return Arrays.asList(h, c);
    }

    /**
     * Get h, the output of the cell.
     *
     * Has shape [batchSize, inSize].
     */
    public SDVariable getOutput(){
        return h;
    }

    /**
     * Get c, the state of the cell.
     *
     * Has shape [batchSize, inSize].
     */
    public SDVariable getState(){
        return c;
    }

}

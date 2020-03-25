package org.nd4j.linalg.api.ops.impl.layers.recurrent.weights;

import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMBlockCell;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMLayer;

/**
 * The weight configuration of a LSTM layer.  For {@link LSTMLayer} and {@link LSTMBlockCell}.
 *
 * @author Alex Black
 */
@EqualsAndHashCode(callSuper = true)
@Data
@Builder
public class LSTMLayerWeights extends RNNWeights {

    /**
     * Input to hidden weights with a shape of [inSize, 4*numUnits].
     *
     * Input to hidden and hidden to hidden are concatenated in dimension 0,
     * so the input to hidden weights are [:inSize, :] and the hidden to hidden weights are [inSize:, :].
     */
    private SDVariable weights;
    private INDArray iWeights;

    /**
     * hidden to hidden weights (aka "recurrent weights", with a shape of [numUnits, 4*numUnits].
     *
     */
    private SDVariable rWeights;
    private INDArray irWeights;

    /**
     * Peephole weights, with a shape of [3*numUnits].
     */
    private SDVariable peepholeWeights;
    private INDArray iPeepholeWeights;

    /**
     * Input to hidden and hidden to hidden biases, with shape [4*numUnits].
     */
    private SDVariable bias;
    private INDArray iBias;

    @Override
    public SDVariable[] args() {
        return filterNonNull(weights, rWeights, peepholeWeights, bias);
    }

    @Override
    public INDArray[] arrayArgs() {
        return filterNonNull(iWeights, irWeights, iPeepholeWeights, iBias);
    }

    @Override
    public SDVariable[] argsWithInputs(SDVariable... inputs){
        Preconditions.checkArgument(inputs.length == 4, "Expected 4 inputs, got %s", inputs.length);   //Order: x, seqLen, yLast, cLast
        //lstmLayer c++ op expects: x, Wx, Wr, Wp, b, seqLen, yLast, cLast
        return new SDVariable[]{inputs[0], weights, rWeights, peepholeWeights, bias, inputs[1], inputs[2], inputs[3]};
    }

    @Override
    public INDArray[] argsWithInputs(INDArray... inputs) {
        Preconditions.checkArgument(inputs.length == 4, "Expected 4 inputs, got %s", inputs.length);   //Order: x, seqLen, yLast, cLast
        //lstmLayer c++ op expects: x, Wx, Wr, Wp, b, seqLen, yLast, cLast
        return new INDArray[]{inputs[0], iWeights, irWeights, iPeepholeWeights, iBias, inputs[1], inputs[2], inputs[3]};
    }


    public boolean hasBias() {
        return (bias!=null||iBias!=null);
    }

    public boolean hasPH() {
        return (peepholeWeights!=null||iPeepholeWeights!=null);
    }

}

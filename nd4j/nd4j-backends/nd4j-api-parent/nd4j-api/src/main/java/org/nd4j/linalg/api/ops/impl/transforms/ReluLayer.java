package org.nd4j.linalg.api.ops.impl.transforms;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ops.DynamicCustomOp;


/**
 * Composed op: relu((X, W) + b)
 *
 * @author Max Pumperla
 */
@NoArgsConstructor
public class ReluLayer extends DynamicCustomOp {


    public ReluLayer(SameDiff sameDiff, SDVariable input, SDVariable weights, SDVariable bias) {
        super(null, sameDiff, new SDVariable[] {input, weights, bias}, false);

    }

    @Override
    public String opName() {
        return "relu_layer";
    }


    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow name found for shape " + opName());
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx name found for shape " + opName());
    }

}

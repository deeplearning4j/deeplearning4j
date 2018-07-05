package org.nd4j.linalg.api.ops.impl.transforms;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;

import java.util.Collections;
import java.util.List;


/**
 * Composed op: relu((X, W) + b)
 *
 * @author Max Pumperla
 */
@NoArgsConstructor
public class ReluLayer extends XwPlusB {


    public ReluLayer(SameDiff sameDiff, SDVariable input, SDVariable weights, SDVariable bias) {
        super(sameDiff, input, weights, bias);

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

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradient) {
        //TODO a native implementation would be faster
        //Backprop through ReLU, then it's same as XwPlusB
        SDVariable[] args = args();
        SDVariable xwb = sameDiff.linear(args[0], args[1], (args.length == 2 ? null : args[2]));
        SDVariable grad = gradient.get(0).mul(sameDiff.step(xwb, 0));
        return super.doDiff(Collections.singletonList(grad));
    }

}

package org.nd4j.linalg.api.ops.impl.controlflow;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.List;

/**
 *
 */
@NoArgsConstructor
public class Select extends DynamicCustomOp {
    public Select(SameDiff sameDiff, SDVariable[] args) {
        super(null, sameDiff, args);
    }

    public Select(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, List<Integer> iArguments) {
        super(opName, inputs, outputs, tArguments, iArguments);
    }

    public Select(String opName, INDArray[] inputs, INDArray[] outputs) {
        super(opName, inputs, outputs);
    }

    public Select(String opName, SameDiff sameDiff, SDVariable[] args, boolean inPlace) {
        super(opName, sameDiff, args, inPlace);
    }

    @Override
    public String opName() {
        return "select";
    }
}

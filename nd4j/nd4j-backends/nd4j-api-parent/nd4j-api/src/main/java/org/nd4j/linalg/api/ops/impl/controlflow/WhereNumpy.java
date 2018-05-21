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
public class WhereNumpy extends DynamicCustomOp {
    public WhereNumpy(SameDiff sameDiff, SDVariable[] args) {
        super(null, sameDiff, args);
    }

    public WhereNumpy(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, List<Integer> iArguments) {
        super(opName, inputs, outputs, tArguments, iArguments);
    }

    public WhereNumpy(INDArray[] inputs, INDArray[] outputs) {
        super(null, inputs, outputs);
    }

    public WhereNumpy(SameDiff sameDiff, SDVariable[] args, boolean inPlace) {
        super(null, sameDiff, args, inPlace);
    }

    @Override
    public String opName() {
        return "where_np";
    }
}

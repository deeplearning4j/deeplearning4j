package org.nd4j.linalg.api.ops.impl.loss;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * L2 loss op wrapper
 */
@NoArgsConstructor
public class L2Loss extends DynamicCustomOp {

    public L2Loss(SameDiff sameDiff, SDVariable[] args) {
        super(null, sameDiff, args);
    }

    @Override
    public List<long[]> calculateOutputShape() {
        return Collections.singletonList(new long[0]);
    }

    @Override
    public String opName() {
        return "l2_loss";
    }

    @Override
    public String tensorflowName() {
        return "L2Loss";
    }
}

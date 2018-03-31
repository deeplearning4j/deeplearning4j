package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.List;

public class MatrixSetDiag extends DynamicCustomOp {
    @Override
    public String tensorflowName() {
        return "MatrixSetDiag";
    }

    @Override
    public String opName() {
        return "matrix_set_diag";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        throw new UnsupportedOperationException();
    }
}

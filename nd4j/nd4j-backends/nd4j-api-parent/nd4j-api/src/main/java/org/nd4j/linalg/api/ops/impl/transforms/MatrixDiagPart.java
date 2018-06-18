package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.List;

public class MatrixDiagPart extends DynamicCustomOp {

    public MatrixDiagPart() {
        //
    }

    public MatrixDiagPart(SameDiff sameDiff, SDVariable in, boolean inPlace) {
        super(null, sameDiff, new SDVariable[]{in}, inPlace);
    }


    @Override
    public String opName() {
        return "matrix_diag_part";
    }

    @Override
    public String tensorflowName() {
        return "MatrixDiagPart";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        throw new UnsupportedOperationException("Not implemented yet");
    }
}

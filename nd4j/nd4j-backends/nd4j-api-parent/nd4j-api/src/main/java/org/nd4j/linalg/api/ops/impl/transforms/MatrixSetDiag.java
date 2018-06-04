package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.List;

public class MatrixSetDiag extends DynamicCustomOp {

    public MatrixSetDiag(SameDiff sameDiff, SDVariable in, SDVariable diag, boolean inPlace) {
        super(null, sameDiff, new SDVariable[]{in, diag}, inPlace);
    }

    public MatrixSetDiag(){ }

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
        SDVariable grad = i_v.get(0);
        SDVariable in1Grad = f().setDiag(grad, sameDiff.zerosLike(arg(1)));
        SDVariable in2Grad = f().diagPart(grad);
        return Arrays.asList(in1Grad, in2Grad);
    }
}

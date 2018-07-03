package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.List;

/**
 * Matrix Determinant op
 *
 * Given input with shape [..., N, N] output the determinant for each sub-matrix.
 *
 * @author Alex Black
 */
public class MatrixDeterminant extends DynamicCustomOp {

    public MatrixDeterminant() {
        //
    }

    public MatrixDeterminant(SameDiff sameDiff, SDVariable in, boolean inPlace) {
        super(null, sameDiff, new SDVariable[]{in}, inPlace);
    }


    @Override
    public String opName() {
        return "matrix_determinant";
    }

    @Override
    public String tensorflowName() {
        return "MatrixDeterminant";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        //return Collections.singletonList(sameDiff.setDiag(sameDiff.zerosLike(arg()), i_v.get(0)));    //Incorrect - needs to be broadcast
        return null;
    }
}

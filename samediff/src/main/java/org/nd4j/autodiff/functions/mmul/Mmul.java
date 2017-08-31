package org.nd4j.autodiff.functions.mmul;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.DifferentialFunctionFactory;
import org.nd4j.autodiff.graph.Graph;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SDGraph;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.shape.Shape;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;

/**
 *  Specialized matrix multiply operations.
 *  Many people know this as "gemm"
 *
 *
 */

public class Mmul extends TensorMmul<ArrayField> {

    private MMulTranspose mMulTranspose;
    public Mmul(SameDiff sameDiff,
                DifferentialFunction<ArrayField> i_v1,
                DifferentialFunction<ArrayField> i_v2,
                MMulTranspose mMulTranspose) {
        super(sameDiff,
                i_v1,
                i_v2, new int[][] {
                        {1},{0}
                });

        this.mMulTranspose = mMulTranspose;
    }


    public Mmul(SameDiff sameDiff,
                DifferentialFunction<ArrayField> i_v1,
                DifferentialFunction<ArrayField> i_v2) {
        super(sameDiff,
                i_v1,
                i_v2, new int[][] {
                        {1},{0}
                });
    }

    @Override
    public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v1) {
        List<DifferentialFunction<ArrayField>> ret = new ArrayList<>();
        DifferentialFunction<ArrayField> gradWrtX = f().reshape(f().mmul(i_v1.get(0),rarg()),larg().getResultShape());
        DifferentialFunction<ArrayField> gradWrtY = f().reshape(f().mmul(larg(),i_v1.get(0)),rarg().getResultShape());
        ret.add(gradWrtX);
        ret.add(gradWrtY);
        larg().setGradient(gradWrtX);
        rarg().setGradient(gradWrtY);
        return ret;
    }

    @Override
    protected void addEdges(SameDiff sameDiff,
                            DifferentialFunction<ArrayField> i_v1,
                            DifferentialFunction<ArrayField> i_v2,
                            String opName) {
        ArrayField arrayField = i_v1.getValue(true);
        ArrayField secondVal = i_v2.getValue(true);
        //skip empty dimensions
        addEdges(sameDiff,i_v1,i_v2,opName,
                OpState.OpType.ACCUMULATION,
                Shape.getMatrixMultiplyShape(arrayField.getInput().getShape(),
                        secondVal.getInput().getShape()));
    }



    /**
     * Get the value of this function
     *
     * @return
     */
    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().mmul(larg(),rarg());
    }


    @Override
    public String functionName() {
        return "mmul";
    }


}

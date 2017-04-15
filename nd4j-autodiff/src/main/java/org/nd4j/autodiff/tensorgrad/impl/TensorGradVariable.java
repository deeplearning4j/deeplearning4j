package org.nd4j.autodiff.tensorgrad.impl;

import lombok.Builder;
import lombok.Data;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.Variable;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.tensorgrad.TensorGrad;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

/**
 * Created by agibsonccc on 4/9/17.
 */
@Data
@Builder
public class TensorGradVariable {
    private INDArray arr;
    private Variable<ArrayField> arrayField;
    private DifferentialFunction<ArrayField> differentialFunction;
    private String varName;
    private TensorGrad tensorGrad;


    public int[] getShape() {
        if(arrayField != null)
            return arrayField.getValue().getInput().getShape();
        else {
            OpState opState =  differentialFunction.getOpState();
            if(opState == null)
                throw new IllegalStateException("Unable to determine shape!");
            return opState.getResult().getShape();
        }
    }


    public boolean isAllocated() {
        return arr != null;
    }

    public void allocate() {
        arr = Nd4j.create(getShape());
    }



    public TensorGradVariable add(TensorGradVariable tensorGradVariable) {
        return TensorGradVariable.builder()
                .varName(varName + " + " + tensorGradVariable.getVarName())
                .arr(null)
                .differentialFunction(tensorGradVariable.getArrayField().plus(arrayField))
                .build();
    }

    public TensorGradVariable sub(TensorGradVariable tensorGradVariable) {
        return TensorGradVariable.builder()
                .varName(varName + " - " + tensorGradVariable.getVarName())
                .arr(null)
                .differentialFunction(tensorGradVariable.getArrayField().minus(arrayField))
                .build();
    }

    public TensorGradVariable div(TensorGradVariable tensorGradVariable) {
        return TensorGradVariable.builder()
                .varName(varName + " / " + tensorGradVariable.getVarName())
                .arr(null)
                .differentialFunction(tensorGradVariable.getArrayField().div(arrayField))
                .build();
    }

    public TensorGradVariable mul(TensorGradVariable tensorGradVariable) {
        return TensorGradVariable.builder()
                .varName(varName + " * " + tensorGradVariable.getVarName())
                .arr(null)
                .differentialFunction(tensorGradVariable.getArrayField().mul(arrayField))
                .build();
    }

}

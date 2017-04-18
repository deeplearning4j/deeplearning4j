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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by agibsonccc on 4/9/17.
 */
@Data
public class TensorGradVariable extends TensorGradFunction implements Serializable {
    private INDArray arr;
    private Variable<ArrayField> arrayField;
    private String varName;
    private TensorGrad tensorGrad;

    @Builder
    private TensorGradVariable(DifferentialFunction<ArrayField> differentialFunction,
                               String varName,
                               INDArray arr,
                               TensorGrad tensorGrad,
                               Variable<ArrayField> arrayField) {
        super(differentialFunction);
        this.differentialFunction = differentialFunction;
        this.varName = varName;
        this.arr = arr;
        this.arrayField = arrayField;
        this.tensorGrad = tensorGrad;
    }

    public String getFormula() {
        List<Variable<ArrayField>> ret = new ArrayList<>();
        if(arrayField != null)
            return arrayField.getFormula(ret);
        else {
            return this.differentialFunction.getFormula(ret);
        }
    }

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
        if(arr == null)
            arr = Nd4j.createUninitialized(getShape());
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

    @Override
    public String toString() {
        return "TensorGradVariable{" +
                "varName='" + varName + '\'' +
                '}';
    }


    public static class TensorGradVariableBuilder extends TensorGradFunction.TensorGradFunctionBuilder {

    }

}

package org.nd4j.autodiff.tensorgrad.impl;

import lombok.Builder;
import lombok.Data;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.Variable;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.tensorgrad.TensorGrad;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 *
 */
@Data
public class TensorGradVariable extends TensorGradFunction implements Serializable {
    private INDArray arr;
    private Variable<ArrayField> arrayField;
    private String varName;
    private TensorGrad tensorGrad;
    private int[] shape;

    @Builder
    private TensorGradVariable(DifferentialFunction<ArrayField> differentialFunction,
                               String varName,
                               INDArray arr,
                               TensorGrad tensorGrad,
                               Variable<ArrayField> arrayField,
                               int[] shape) {
        super(differentialFunction);
        this.shape = shape;
        this.differentialFunction = differentialFunction;
        this.varName = varName;
        this.arr = arr;
        this.arrayField = arrayField;
        this.tensorGrad = tensorGrad;
    }


    /**
     *
     * @return
     */
    public NDArrayInformation getInfo() {
        return getArrayField().getM_x().getInput();
    }

    /**
     *
     * @return
     */
    public String getFormula() {
        List<Variable<ArrayField>> ret = new ArrayList<>();
        if(arrayField != null)
            return arrayField.getFormula(ret);
        else {
            return this.differentialFunction.getFormula(ret);
        }
    }

    /**
     *
     * @return
     */
    public int[] getShape() {
        if(shape != null)
            return shape;

        if(arrayField != null)
            return arrayField.getValue().getInput().getShape();

        else {
            OpState opState =  differentialFunction.getOpState();
            if(opState == null) {
                return differentialFunction.getValue().getInput().getShape();
            }
            return opState.getResult().getShape();
        }
    }


    /**
     *
     * @return
     */
    public boolean isAllocated() {
        return arr != null;
    }

    /**
     *
     */
    public void allocate() {
        if(arr == null)
            arr = Nd4j.createUninitialized(getShape());
    }


    /**
     *
     * @return
     */
    public TensorGradVariable dup() {
        return TensorGradVariable.builder()
                .differentialFunction(differentialFunction)
                .arrayField(arrayField)
                .varName(varName)
                .tensorGrad(tensorGrad)
                .arr(arr.dup())
                .build();
    }

    public TensorGradVariable rsub(TensorGradVariable tensorGradVariable) {
        return TensorGradVariable.builder()
                .varName(varName + " + " + tensorGradVariable.getVarName())
                .arr(null).tensorGrad(tensorGradVariable.getTensorGrad())
                .differentialFunction(getFunction(tensorGradVariable).rsub(getFunction(this)))
                .build();
    }

    public TensorGradVariable rdiv(TensorGradVariable tensorGradVariable) {
        return TensorGradVariable.builder()
                .varName(varName + " + " + tensorGradVariable.getVarName())
                .arr(null).tensorGrad(tensorGradVariable.getTensorGrad())
                .differentialFunction(getFunction(tensorGradVariable).rdiv(getFunction(this)))
                .build();
    }

    public TensorGradVariable add(TensorGradVariable tensorGradVariable) {
        return TensorGradVariable.builder()
                .varName(varName + " + " + tensorGradVariable.getVarName())
                .arr(null).tensorGrad(tensorGradVariable.getTensorGrad())
                .differentialFunction(getFunction(tensorGradVariable).add(getFunction(this)))
                .build();
    }

    public TensorGradVariable sub(TensorGradVariable tensorGradVariable) {
        return TensorGradVariable.builder()
                .varName(varName + " - " + tensorGradVariable.getVarName())
                .arr(null).tensorGrad(tensorGradVariable.getTensorGrad())
                .differentialFunction(getFunction(tensorGradVariable).sub(getFunction(this)))
                .build();
    }

    public TensorGradVariable div(TensorGradVariable tensorGradVariable) {
        return TensorGradVariable.builder()
                .varName(varName + " / " + tensorGradVariable.getVarName())
                .arr(null).tensorGrad(tensorGradVariable.getTensorGrad())
                .differentialFunction(getFunction(tensorGradVariable).div(getFunction(this)))
                .build();
    }

    public TensorGradVariable mul(TensorGradVariable tensorGradVariable) {
        return TensorGradVariable.builder()
                .varName(varName + " * " + tensorGradVariable.getVarName())
                .arr(null).tensorGrad(tensorGradVariable.getTensorGrad())
                .differentialFunction(getFunction(tensorGradVariable).mul(getFunction(this)))
                .build();
    }


    private DifferentialFunction<ArrayField> getFunction(TensorGradVariable variable) {
        return variable.getArrayField() == null ? variable.getDifferentialFunction() : variable.getArrayField();
    }

    @Override
    public String toString() {
        return "TensorGradVariable{" +
                "varName='" + varName + '\'' +
                '}';
    }


    //lombok for inheritance purposes, do not remove
    @SuppressWarnings("unused")
    public static class TensorGradVariableBuilder extends TensorGradFunction.TensorGradFunctionBuilder {


    }

}

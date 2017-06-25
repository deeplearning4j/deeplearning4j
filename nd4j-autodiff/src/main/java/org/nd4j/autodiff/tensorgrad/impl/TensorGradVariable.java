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

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        if (!super.equals(o)) return false;

        TensorGradVariable that = (TensorGradVariable) o;

        if (arr != null ? !arr.equals(that.arr) : that.arr != null) return false;
        if (arrayField != null ? !arrayField.equals(that.arrayField) : that.arrayField != null) return false;
        if (varName != null ? !varName.equals(that.varName) : that.varName != null) return false;
        return Arrays.equals(shape, that.shape);
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + (arr != null ? arr.hashCode() : 0);
        result = 31 * result + (arrayField != null ? arrayField.hashCode() : 0);
        result = 31 * result + (varName != null ? varName.hashCode() : 0);
        result = 31 * result + Arrays.hashCode(shape);
        return result;
    }

    public NDArrayInformation getInfo() {
        return getArrayField().getM_x().getInput();
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
        if(shape != null)
            return shape;

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
                .arr(null)
                .differentialFunction(getFunction(tensorGradVariable).rsub(getFunction(this)))
                .build();
    }

    public TensorGradVariable rdiv(TensorGradVariable tensorGradVariable) {
        return TensorGradVariable.builder()
                .varName(varName + " + " + tensorGradVariable.getVarName())
                .arr(null)
                .differentialFunction(getFunction(tensorGradVariable).rdiv(getFunction(this)))
                .build();
    }

    public TensorGradVariable add(TensorGradVariable tensorGradVariable) {
        return TensorGradVariable.builder()
                .varName(varName + " + " + tensorGradVariable.getVarName())
                .arr(null)
                .differentialFunction(getFunction(tensorGradVariable).add(getFunction(this)))
                .build();
    }

    public TensorGradVariable sub(TensorGradVariable tensorGradVariable) {
        return TensorGradVariable.builder()
                .varName(varName + " - " + tensorGradVariable.getVarName())
                .arr(null)
                .differentialFunction(getFunction(tensorGradVariable).sub(getFunction(this)))
                .build();
    }

    public TensorGradVariable div(TensorGradVariable tensorGradVariable) {
        return TensorGradVariable.builder()
                .varName(varName + " / " + tensorGradVariable.getVarName())
                .arr(null)
                .differentialFunction(getFunction(tensorGradVariable).div(getFunction(this)))
                .build();
    }

    public TensorGradVariable mul(TensorGradVariable tensorGradVariable) {
        return TensorGradVariable.builder()
                .varName(varName + " * " + tensorGradVariable.getVarName())
                .arr(null)
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

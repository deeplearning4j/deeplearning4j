package org.nd4j.autodiff.samediff.impl;

import lombok.Builder;
import lombok.Data;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.Variable;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 *
 */
@Data
public class SameDiffVariable extends SameDiffFunction implements Serializable {
    private INDArray arr;
    private Variable<ArrayField> arrayField;
    private String varName;
    private SameDiff sameDiff;
    private int[] shape;

    @Builder
    private SameDiffVariable(DifferentialFunction<ArrayField> differentialFunction,
                             String varName,
                             INDArray arr,
                             SameDiff sameDiff,
                             Variable<ArrayField> arrayField,
                             int[] shape) {
        super(differentialFunction);
        this.shape = shape;
        this.differentialFunction = differentialFunction;
        this.varName = varName;
        this.arr = arr;
        this.arrayField = arrayField;
        this.sameDiff = sameDiff;
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
        if(differentialFunction == null)
            throw new IllegalStateException("Unable to infer shape. Function is null.");
        OpState opState =  differentialFunction.getOpState();
        if(opState == null) {
            return differentialFunction.getValue().getInput().getShape();
        }

        return opState.getResult().getShape();

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
    public SameDiffVariable dup() {
        return SameDiffVariable.builder()
                .differentialFunction(differentialFunction)
                .arrayField(arrayField)
                .varName(varName)
                .tensorGrad(sameDiff)
                .arr(arr.dup())
                .build();
    }

    private int[] getTransformOutputShape(SameDiffVariable other) {
        if(shape == null)
            return other.getShape();
        if(ArrayUtil.prod(shape) == 1) {
            return other.getShape();
        }

        return getShape();
    }


    public SameDiffVariable rsub(SameDiffVariable tensorGradVariable) {
        assertShapeEquals(tensorGradVariable);

        return SameDiffVariable.builder()
                .varName(varName + " + " + tensorGradVariable.getVarName())
                .arr(null).tensorGrad(tensorGradVariable.getTensorGrad()).shape(getTransformOutputShape(tensorGradVariable))
                .differentialFunction(getFunction(tensorGradVariable).rsub(getFunction(this)))
                .build();
    }

    public SameDiffVariable rdiv(SameDiffVariable tensorGradVariable) {
        assertShapeEquals(tensorGradVariable);

        return SameDiffVariable.builder()
                .varName(varName + " + " + tensorGradVariable.getVarName())
                .arr(null).tensorGrad(tensorGradVariable.getTensorGrad()).shape(getTransformOutputShape(tensorGradVariable))
                .differentialFunction(getFunction(tensorGradVariable).rdiv(getFunction(this)))
                .build();
    }

    public SameDiffVariable add(SameDiffVariable tensorGradVariable) {
        assertShapeEquals(tensorGradVariable);

        return SameDiffVariable.builder()
                .varName(varName + " + " + tensorGradVariable.getVarName())
                .arr(null).tensorGrad(tensorGradVariable.getTensorGrad()).shape(getTransformOutputShape(tensorGradVariable))
                .differentialFunction(getFunction(tensorGradVariable).add(getFunction(this)))
                .build();
    }

    public SameDiffVariable sub(SameDiffVariable tensorGradVariable) {
        assertShapeEquals(tensorGradVariable);

        DifferentialFunction<ArrayField> left = getFunction(tensorGradVariable);
        DifferentialFunction<ArrayField> right = getFunction(this);
        DifferentialFunction<ArrayField> result = left.sub(right);
        return SameDiffVariable.builder()
                .varName(varName + " - " + tensorGradVariable.getVarName())
                .arr(null).tensorGrad(tensorGradVariable.getTensorGrad()).shape(getTransformOutputShape(tensorGradVariable))
                .differentialFunction(result)
                .build();
    }

    public SameDiffVariable div(SameDiffVariable tensorGradVariable) {
        assertShapeEquals(tensorGradVariable);

        return SameDiffVariable.builder()
                .varName(varName + " / " + tensorGradVariable.getVarName())
                .arr(null).tensorGrad(tensorGradVariable.getTensorGrad()).shape(getTransformOutputShape(tensorGradVariable))
                .differentialFunction(getFunction(tensorGradVariable).div(getFunction(this)))
                .build();
    }

    public SameDiffVariable mul(SameDiffVariable tensorGradVariable) {
        assertShapeEquals(tensorGradVariable);

        DifferentialFunction<ArrayField> left = getFunction(tensorGradVariable);
        DifferentialFunction<ArrayField> right = getFunction(this);
        DifferentialFunction<ArrayField> result = left.mul(right);
        return SameDiffVariable.builder()
                .varName(varName + " * " + tensorGradVariable.getVarName())
                .arr(null).tensorGrad(tensorGradVariable.getTensorGrad())
                .shape(getTransformOutputShape(tensorGradVariable))
                .differentialFunction(result)
                .build();
    }

    private void assertShapeEquals(SameDiffVariable variable) {
        if(!Arrays.equals(shape,variable.getShape()) && ArrayUtil.prod(variable.getShape()) != 1) {
            throw new IllegalArgumentException("Input shape must be the same as this shape " + Arrays.toString(shape) + " and shape was " + Arrays.toString(variable.getShape()));
        }
    }


    private DifferentialFunction<ArrayField> getFunction(SameDiffVariable variable) {
        return variable.getDifferentialFunction() != null ? variable.getDifferentialFunction() : variable.getArrayField();
    }

    @Override
    public String toString() {
        return "SameDiffVariable{" +
                "varName='" + varName + '\'' +
                '}';
    }


    //lombok for inheritance purposes, do not remove
    @SuppressWarnings("unused")
    public static class TensorGradVariableBuilder extends SameDiffFunction.TensorGradFunctionBuilder {


    }

}

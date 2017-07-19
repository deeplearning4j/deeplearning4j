package org.nd4j.autodiff.samediff.impl;

import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.Variable;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.OpState;
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
public class SDVariable  implements Serializable {
    private INDArray arr;
    private Variable<ArrayField> arrayField;
    private String varName;
    private SameDiff sameDiff;
    private int[] shape;
    protected DifferentialFunction<ArrayField> differentialFunction;

    @Builder
    private SDVariable(DifferentialFunction<ArrayField> differentialFunction,
                       String varName,
                       INDArray arr,
                       SameDiff sameDiff,
                       Variable<ArrayField> arrayField,
                       int[] shape) {
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
    public SDVariable dup() {
        return SDVariable.builder()
                .differentialFunction(differentialFunction)
                .arrayField(arrayField)
                .varName(varName)
                .sameDiff(sameDiff)
                .arr(arr.dup())
                .build();
    }

    private int[] getTransformOutputShape(SDVariable other) {
        if(shape == null)
            return other.getShape();
        if(ArrayUtil.prod(shape) == 1) {
            return other.getShape();
        }

        return getShape();
    }


    public SDVariable rsub(SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);

        return SDVariable.builder()
                .varName(varName + " + " + sameDiffVariable.getVarName())
                .arr(null).sameDiff(sameDiffVariable.getSameDiff()).shape(getTransformOutputShape(sameDiffVariable))
                .differentialFunction(getFunction(sameDiffVariable).rsub(getFunction(this)))
                .build();
    }

    public SDVariable rdiv(SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);

        return SDVariable.builder()
                .varName(varName + " + " + sameDiffVariable.getVarName())
                .arr(null).sameDiff(sameDiffVariable.getSameDiff()).shape(getTransformOutputShape(sameDiffVariable))
                .differentialFunction(getFunction(sameDiffVariable).rdiv(getFunction(this)))
                .build();
    }

    public SDVariable add(SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);

        return SDVariable.builder()
                .varName(varName + " + " + sameDiffVariable.getVarName())
                .arr(null).sameDiff(sameDiffVariable.getSameDiff()).shape(getTransformOutputShape(sameDiffVariable))
                .differentialFunction(getFunction(sameDiffVariable).add(getFunction(this)))
                .build();
    }

    public SDVariable sub(SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);

        DifferentialFunction<ArrayField> left = getFunction(sameDiffVariable);
        DifferentialFunction<ArrayField> right = getFunction(this);
        DifferentialFunction<ArrayField> result = left.sub(right);
        return SDVariable.builder()
                .varName(varName + " - " + sameDiffVariable.getVarName())
                .arr(null).sameDiff(sameDiffVariable.getSameDiff()).shape(getTransformOutputShape(sameDiffVariable))
                .differentialFunction(result)
                .build();
    }

    public SDVariable div(SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);

        return SDVariable.builder()
                .varName(varName + " / " + sameDiffVariable.getVarName())
                .arr(null).sameDiff(sameDiffVariable.getSameDiff()).shape(getTransformOutputShape(sameDiffVariable))
                .differentialFunction(getFunction(sameDiffVariable).div(getFunction(this)))
                .build();
    }

    public SDVariable mul(SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);

        DifferentialFunction<ArrayField> left = getFunction(sameDiffVariable);
        DifferentialFunction<ArrayField> right = getFunction(this);
        DifferentialFunction<ArrayField> result = left.mul(right);
        return SDVariable.builder()
                .varName(varName + " * " + sameDiffVariable.getVarName())
                .arr(null).sameDiff(sameDiffVariable.getSameDiff())
                .shape(getTransformOutputShape(sameDiffVariable))
                .differentialFunction(result)
                .build();
    }

    private void assertShapeEquals(SDVariable variable) {
        if(!Arrays.equals(shape,variable.getShape()) && ArrayUtil.prod(variable.getShape()) != 1) {
            throw new IllegalArgumentException("Input shape must be the same as this shape " + Arrays.toString(shape) + " and shape was " + Arrays.toString(variable.getShape()));
        }
    }


    private DifferentialFunction<ArrayField> getFunction(SDVariable variable) {
        return variable.getDifferentialFunction() != null ? variable.getDifferentialFunction() : variable.getArrayField();
    }

    @Override
    public String toString() {
        return "SDVariable{" +
                "varName='" + varName + '\'' +
                '}';
    }




}

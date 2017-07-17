package org.nd4j.autodiff.tensorgrad.impl;

import com.kitfox.svg.A;
import lombok.Builder;
import lombok.Data;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.Variable;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.tensorgrad.TensorGrad;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
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
    public TensorGradVariable dup() {
        return TensorGradVariable.builder()
                .differentialFunction(differentialFunction)
                .arrayField(arrayField)
                .varName(varName)
                .tensorGrad(tensorGrad)
                .arr(arr.dup())
                .build();
    }

    private int[] getTransformOutputShape(TensorGradVariable other) {
        if(shape == null)
            return other.getShape();
        if(ArrayUtil.prod(shape) == 1) {
            return other.getShape();
        }

        return getShape();
    }


    public TensorGradVariable rsub(TensorGradVariable tensorGradVariable) {
        assertShapeEquals(tensorGradVariable);

        return TensorGradVariable.builder()
                .varName(varName + " + " + tensorGradVariable.getVarName())
                .arr(null).tensorGrad(tensorGradVariable.getTensorGrad()).shape(getTransformOutputShape(tensorGradVariable))
                .differentialFunction(getFunction(tensorGradVariable).rsub(getFunction(this)))
                .build();
    }

    public TensorGradVariable rdiv(TensorGradVariable tensorGradVariable) {
        assertShapeEquals(tensorGradVariable);

        return TensorGradVariable.builder()
                .varName(varName + " + " + tensorGradVariable.getVarName())
                .arr(null).tensorGrad(tensorGradVariable.getTensorGrad()).shape(getTransformOutputShape(tensorGradVariable))
                .differentialFunction(getFunction(tensorGradVariable).rdiv(getFunction(this)))
                .build();
    }

    public TensorGradVariable add(TensorGradVariable tensorGradVariable) {
        assertShapeEquals(tensorGradVariable);

        return TensorGradVariable.builder()
                .varName(varName + " + " + tensorGradVariable.getVarName())
                .arr(null).tensorGrad(tensorGradVariable.getTensorGrad()).shape(getTransformOutputShape(tensorGradVariable))
                .differentialFunction(getFunction(tensorGradVariable).add(getFunction(this)))
                .build();
    }

    public TensorGradVariable sub(TensorGradVariable tensorGradVariable) {
        assertShapeEquals(tensorGradVariable);

        DifferentialFunction<ArrayField> left = getFunction(tensorGradVariable);
        DifferentialFunction<ArrayField> right = getFunction(this);
        DifferentialFunction<ArrayField> result = left.sub(right);
        return TensorGradVariable.builder()
                .varName(varName + " - " + tensorGradVariable.getVarName())
                .arr(null).tensorGrad(tensorGradVariable.getTensorGrad()).shape(getTransformOutputShape(tensorGradVariable))
                .differentialFunction(result)
                .build();
    }

    public TensorGradVariable div(TensorGradVariable tensorGradVariable) {
        assertShapeEquals(tensorGradVariable);

        return TensorGradVariable.builder()
                .varName(varName + " / " + tensorGradVariable.getVarName())
                .arr(null).tensorGrad(tensorGradVariable.getTensorGrad()).shape(getTransformOutputShape(tensorGradVariable))
                .differentialFunction(getFunction(tensorGradVariable).div(getFunction(this)))
                .build();
    }

    public TensorGradVariable mul(TensorGradVariable tensorGradVariable) {
        assertShapeEquals(tensorGradVariable);

        DifferentialFunction<ArrayField> left = getFunction(tensorGradVariable);
        DifferentialFunction<ArrayField> right = getFunction(this);
        DifferentialFunction<ArrayField> result = left.mul(right);
        return TensorGradVariable.builder()
                .varName(varName + " * " + tensorGradVariable.getVarName())
                .arr(null).tensorGrad(tensorGradVariable.getTensorGrad())
                .shape(getTransformOutputShape(tensorGradVariable))
                .differentialFunction(result)
                .build();
    }

    private void assertShapeEquals(TensorGradVariable variable) {
        if(!Arrays.equals(shape,variable.getShape()) && ArrayUtil.prod(variable.getShape()) != 1) {
            throw new IllegalArgumentException("Input shape must be the same as this shape " + Arrays.toString(shape) + " and shape was " + Arrays.toString(variable.getShape()));
        }
    }


    private DifferentialFunction<ArrayField> getFunction(TensorGradVariable variable) {
        return variable.getDifferentialFunction() != null ? variable.getDifferentialFunction() : variable.getArrayField();
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

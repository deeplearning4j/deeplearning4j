package org.nd4j.autodiff.samediff.impl;

import lombok.Builder;
import lombok.Data;
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
import java.util.Map;

/**
 *
 * A variable representing a component within a
 * {@@link SameDiff} graph.
 *
 * SDVariable is used for symbolic declaration
 * of equations.
 *
 * @author Adam Gibson
 *
 */
@Data
public class SDVariable  implements Serializable {
    private INDArray arr;
    private Variable<ArrayField> arrayField;
    private String varName;
    private SameDiff sameDiff;
    private int[] shape;
    private SDVariable gradient;
    private int vertexId;
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
        if(differentialFunction != null)
            this.vertexId = differentialFunction.getVertexId();
        else if(arrayField != null)
            this.vertexId = arrayField.getVertexId();

    }


    /**
     * Nicer looking alias
     * for the gradient variable.
     * The gradient variable is meant to be an
     * a variable representation
     * of the gradient represented
     * in the underlying {@link DifferentialFunction}
     * @return
     */
    public SDVariable gradient() {
        return getGradient();
    }

    /**
     * A getter for the variable gradient.
     * Note here that a lazy initialization of the
     * gradient variable will happen if the gradient
     * isn't present at this variable's initialization
     * but is set later.
     * @return
     */
    public SDVariable getGradient() {
        if(gradient == null && differentialFunction != null && differentialFunction.getGradient() != null) {
            this.gradient = differentialFunction != null && differentialFunction.getGradient() != null ? SDVariable.builder()
                    .sameDiff(sameDiff)
                    .differentialFunction(differentialFunction.getGradient())
                    .varName(varName + "-grad")
                    .shape(differentialFunction.getGradient() != null ? differentialFunction.getGradient().getResultShape() : null)
                    .build() : null;
        }

        else if(gradient == null && arrayField != null && arrayField.getGradient() != null) {
            this.gradient = arrayField != null && arrayField.getGradient() != null ? SDVariable.builder()
                    .sameDiff(sameDiff)
                    .differentialFunction(arrayField.getGradient())
                    .varName(varName + "-grad")
                    .shape(arrayField.getGradient() != null ? arrayField.getGradient().getResultShape() : null)
                    .build() : null;
        }

        return gradient;
    }

    public void setGradient(SDVariable gradient) {
        this.gradient = gradient;
    }

    /**
     *
     * @return
     */
    public NDArrayInformation getInfo() {
        if(getArrayField() == null)
            return null;
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
     * Invokes this wrt itself starting as 1.
     * @return
     */
    public SDVariable backward() {
        if(ArrayUtil.prod(getShape()) != 1) {
            throw new IllegalStateException("Backward invocations must involve calling a scalar.");
        }

        return sameDiff.grad(this,this);
    }

    /**
     * Returns the shape of this variable
     * @return
     */
    public int[] getShape() {
        if(shape != null)
            return shape;
        if(differentialFunction == null)
            throw new IllegalStateException("Unable to infer shape. Function is null.");
        OpState opState =  differentialFunction.getOpState();
        if(opState == null) {
            return differentialFunction.getValue(true).getInput().getShape();
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
            arr = Nd4j.zeros(getShape());
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
                .arr(arr != null ? arr.dup() : null)
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





    //scalars

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rsub(double sameDiffVariable) {
        DifferentialFunction<ArrayField> function = getFunction(this).rsub(sameDiffVariable);

        return SDVariable.builder()
                .varName(varName + " + " + "scalar")
                .arr(null).sameDiff(getSameDiff())
                .shape(getTransformOutputShape(this))
                .differentialFunction(function)
                .build();
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rdiv(double sameDiffVariable) {
        DifferentialFunction<ArrayField> function = getFunction(this)
                .rdiv(sameDiffVariable);

        return SDVariable.builder()
                .varName(varName + " + " + "scalar")
                .arr(null).sameDiff(getSameDiff())
                .shape(getTransformOutputShape(this))
                .differentialFunction(function)
                .build();
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable add(double sameDiffVariable) {
        DifferentialFunction<ArrayField> function = getFunction(this).add(sameDiffVariable);
        return SDVariable.builder()
                .varName(varName + " + " + "scalar")
                .arr(null).sameDiff(getSameDiff())
                .shape(getTransformOutputShape(this))
                .differentialFunction(function)
                .build();
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable sub(double sameDiffVariable) {
        DifferentialFunction<ArrayField> right = getFunction(this);
        DifferentialFunction<ArrayField> result = right.sub(sameDiffVariable);
        return SDVariable.builder()
                .varName(varName + " - " + "scalar")
                .arr(null).sameDiff(getSameDiff())
                .shape(getTransformOutputShape(this))
                .differentialFunction(result)
                .build();
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable div(double sameDiffVariable) {
        DifferentialFunction<ArrayField> function = getFunction(this)
                .div(sameDiffVariable);

        return SDVariable.builder()
                .varName(varName + " / " + "scalar")
                .arr(null).sameDiff(getSameDiff())
                .shape(getTransformOutputShape(this))
                .differentialFunction(function)
                .build();
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable mul(double sameDiffVariable) {
        DifferentialFunction<ArrayField> function = getFunction(this)
                .mul(sameDiffVariable);
        return SDVariable.builder()
                .varName(varName + " * " + "scalar")
                .arr(null).sameDiff(getSameDiff())
                .shape(getTransformOutputShape(this))
                .differentialFunction(function)
                .build();
    }


    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rsubi(double sameDiffVariable) {
        DifferentialFunction<ArrayField> function = getFunction(this)
                .rsubi(sameDiffVariable);

        return SDVariable.builder()
                .varName(varName + " - " + "scalar")
                .arr(null).sameDiff(getSameDiff())
                .shape(getTransformOutputShape(this))
                .differentialFunction(function)
                .build();
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rdivi(double sameDiffVariable) {
        DifferentialFunction<ArrayField> function = getFunction(this)
                .rdivi(sameDiffVariable);

        return SDVariable.builder()
                .varName(varName + " / " + "scalar")
                .arr(null).sameDiff(getSameDiff())
                .shape(getTransformOutputShape(this))
                .differentialFunction(function)
                .build();
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable addi(double sameDiffVariable) {
        DifferentialFunction<ArrayField> function = getFunction(this).
                addi(sameDiffVariable);

        return SDVariable.builder()
                .varName(varName + " + " + "scalar")
                .arr(null).sameDiff(getSameDiff())
                .shape(getTransformOutputShape(this))
                .differentialFunction(function)
                .build();
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable subi(double sameDiffVariable) {
        DifferentialFunction<ArrayField> function = getFunction(this)
                .subi(sameDiffVariable);

        return SDVariable.builder()
                .varName(varName + " - " + "scalar")
                .arr(null).sameDiff(getSameDiff())
                .shape(getTransformOutputShape(this))
                .differentialFunction(function)
                .build();
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable divi(double sameDiffVariable) {
        DifferentialFunction<ArrayField> function = getFunction(this)
                .divi(sameDiffVariable);

        return SDVariable.builder()
                .varName(varName + " / " + "scalar")
                .arr(null)
                .sameDiff(getSameDiff())
                .shape(getTransformOutputShape(this))
                .differentialFunction(function)
                .build();
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable muli(double sameDiffVariable) {
        DifferentialFunction<ArrayField> function = getFunction(this).muli(sameDiffVariable);

        return SDVariable.builder()
                .varName(varName + " * " + "scalar")
                .arr(null).sameDiff(getSameDiff())
                .shape(getTransformOutputShape(this))
                .differentialFunction(function)
                .build();
    }



    //end scalars


    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rsub(SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);

        return SDVariable.builder()
                .varName(varName + " + " + sameDiffVariable.getVarName())
                .arr(null).sameDiff(sameDiffVariable.getSameDiff()).shape(getTransformOutputShape(sameDiffVariable))
                .differentialFunction(getFunction(sameDiffVariable).rsub(getFunction(this)))
                .build();
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rdiv(SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);

        return SDVariable.builder()
                .varName(varName + " + " + sameDiffVariable.getVarName())
                .arr(null).sameDiff(sameDiffVariable.getSameDiff()).shape(getTransformOutputShape(sameDiffVariable))
                .differentialFunction(getFunction(sameDiffVariable).rdiv(getFunction(this)))
                .build();
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable add(SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);

        return SDVariable.builder()
                .varName(varName + " + " + sameDiffVariable.getVarName())
                .arr(null).sameDiff(sameDiffVariable.getSameDiff()).shape(getTransformOutputShape(sameDiffVariable))
                .differentialFunction(getFunction(sameDiffVariable).add(getFunction(this)))
                .build();
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
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

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable div(SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);

        return SDVariable.builder()
                .varName(varName + " / " + sameDiffVariable.getVarName())
                .arr(null).sameDiff(sameDiffVariable.getSameDiff()).shape(getTransformOutputShape(sameDiffVariable))
                .differentialFunction(getFunction(sameDiffVariable).div(getFunction(this)))
                .build();
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
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


    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rsubi(SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);

        return SDVariable.builder()
                .varName(varName + " + " + sameDiffVariable.getVarName())
                .arr(null).sameDiff(sameDiffVariable.getSameDiff()).shape(getTransformOutputShape(sameDiffVariable))
                .differentialFunction(getFunction(sameDiffVariable).rsubi(getFunction(this)))
                .build();
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rdivi(SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);

        return SDVariable.builder()
                .varName(varName + " + " + sameDiffVariable.getVarName())
                .arr(null).sameDiff(sameDiffVariable.getSameDiff()).shape(getTransformOutputShape(sameDiffVariable))
                .differentialFunction(getFunction(sameDiffVariable).rdivi(getFunction(this)))
                .build();
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable addi(SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);

        return SDVariable.builder()
                .varName(varName + " + " + sameDiffVariable.getVarName())
                .arr(null).sameDiff(sameDiffVariable.getSameDiff()).shape(getTransformOutputShape(sameDiffVariable))
                .differentialFunction(getFunction(sameDiffVariable).addi(getFunction(this)))
                .build();
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable subi(SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);

        DifferentialFunction<ArrayField> left = getFunction(sameDiffVariable);
        DifferentialFunction<ArrayField> right = getFunction(this);
        DifferentialFunction<ArrayField> result = left.subi(right);
        return SDVariable.builder()
                .varName(varName + " - " + sameDiffVariable.getVarName())
                .arr(null).sameDiff(sameDiffVariable.getSameDiff()).shape(getTransformOutputShape(sameDiffVariable))
                .differentialFunction(result)
                .build();
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable divi(SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);

        return SDVariable.builder()
                .varName(varName + " / " + sameDiffVariable.getVarName())
                .arr(null).sameDiff(sameDiffVariable.getSameDiff()).shape(getTransformOutputShape(sameDiffVariable))
                .differentialFunction(getFunction(sameDiffVariable).divi(getFunction(this)))
                .build();
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable muli(SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);

        DifferentialFunction<ArrayField> left = getFunction(sameDiffVariable);
        DifferentialFunction<ArrayField> right = getFunction(this);
        DifferentialFunction<ArrayField> result = left.muli(right);
        return SDVariable.builder()
                .varName(varName + " * " + sameDiffVariable.getVarName())
                .arr(null).sameDiff(sameDiffVariable.getSameDiff())
                .shape(getTransformOutputShape(sameDiffVariable))
                .differentialFunction(result)
                .build();
    }


    /**
     * Evaluate the result of this variable
     * @return
     */
    public INDArray eval() {
        SameDiff exec = sameDiff.dup();
        exec.defineFunction("output", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable define(SameDiff sameDiff, Map<String, INDArray> inputs) {
                return SDVariable.this;
            }
        });

        SDVariable output = exec.invokeFunctionOn("output",exec);
        return output.getSameDiff().execAndEndResult();
    }





    private void assertShapeEquals(SDVariable variable) {
        if(!Arrays.equals(shape,variable.getShape()) && ArrayUtil.prod(variable.getShape()) != 1) {
            throw new IllegalArgumentException("Input shape must be the same as this shape " + Arrays.toString(shape) + " and shape was " + Arrays.toString(variable.getShape()));
        }
    }


    /**
     * Return the underlying differential
     * function
     * or array field.
     * @param variable
     * @return
     */
    public static DifferentialFunction<ArrayField> getFunction(SDVariable variable) {
        if(variable == null)
            throw new IllegalArgumentException("Unable to get function for null variable");
        return variable.getDifferentialFunction() != null ? variable.getDifferentialFunction() : variable.getArrayField();
    }

    @Override
    public String toString() {
        return "SDVariable{" +
                "varName='" + varName + '\'' +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        if (!super.equals(o)) return false;

        SDVariable variable = (SDVariable) o;

        if (arr != null ? !arr.equals(variable.arr) : variable.arr != null) return false;
        if (arrayField != null ? !arrayField.equals(variable.arrayField) : variable.arrayField != null) return false;
        if (varName != null ? !varName.equals(variable.varName) : variable.varName != null) return false;
        if (!Arrays.equals(shape, variable.shape)) return false;
        return differentialFunction != null ? differentialFunction.equals(variable.differentialFunction) : variable.differentialFunction == null;
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + (arr != null ? arr.hashCode() : 0);
        result = 31 * result + (arrayField != null ? arrayField.hashCode() : 0);
        result = 31 * result + (varName != null ? varName.hashCode() : 0);
        result = 31 * result + Arrays.hashCode(shape);
        result = 31 * result + (differentialFunction != null ? differentialFunction.hashCode() : 0);
        return result;
    }
}

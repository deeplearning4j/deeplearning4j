package org.nd4j.autodiff.samediff;

import com.google.common.base.Preconditions;
import lombok.*;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.shade.jackson.annotation.JsonIgnore;
import org.nd4j.weightinit.WeightInitScheme;
import org.nd4j.weightinit.impl.ZeroInitScheme;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.io.Serializable;
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
@NoArgsConstructor
public class SDVariable extends DifferentialFunction implements Serializable {


    @Getter
    @Setter
    private String varName;
    @Getter
    @Setter
    protected WeightInitScheme weightInitScheme;
    @Getter
    @Setter
    @JsonIgnore
    protected int depth;



    @Builder
    private SDVariable(String varName,
                       SameDiff sameDiff,
                       int[] shape,
                       WeightInitScheme weightInitScheme) {
        if(shape != null && shape.length >= 2)
            sameDiff.putShapeForVarName(varName,Shape.resolveNegativeShapeIfNeccessary(new int[shape.length],shape));
        this.varName = varName;
        this.weightInitScheme = weightInitScheme;

        if(weightInitScheme == null) {
            this.weightInitScheme = new ZeroInitScheme('f');
        }

        this.sameDiff = sameDiff;


    }

    @Override
    public String opName() {
        return "variable";
    }

    @Override
    public SDVariable[] outputVariables() {
        return new SDVariable[0];
    }


    @Override
    public boolean isVariable() {
        return true;
    }




    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {

    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {

    }


    @Override
    public void initWithArrays(Map<String, INDArray> arrayMap, Object... extraArgs) {
        //no-op
    }


    /**
     * Allocate and return a  new array
     * based on the vertex id and weight initialization.
     * @return the allocated array
     */
    public INDArray storeAndAllocateNewArray() {
        val shape = sameDiff.getShapeForVarName(getVarName());
        if(shape == null) {
            throw new ND4JIllegalStateException("Unable to allocate new array. No shape found for variable " + varName);
        }

        val arr = getWeightInitScheme().create(shape);
        sameDiff.putArrayForVarName(getVarName(),arr);
        return arr;
    }

    /**
     * A getter for the allocated ndarray
     * with this {@link SDVariable}.
     *
     * This getter will lazy initialize an array if one is not found
     * based on the associated shape and {@link WeightInitScheme}
     * if neither are found, an {@link ND4JIllegalStateException}
     * is thrown.
     *
     * If a {@link DifferentialFunction} is defined, note that
     * its getArr() method is called instead.
     * @return the {@link INDArray} associated with this variable.
     */
    public INDArray getArr() {
        if(sameDiff.arrayAlreadyExistsForVarName(getVarName()))
            return sameDiff.getArrForVarName(getVarName());

        //initialize value if it's actually a scalar constant (zero or 1 typically...)
        if(getScalarValue() != null && ArrayUtil.prod(getShape()) == 1) {
            INDArray arr = Nd4j.valueArrayOf(getShape(),
                    getScalarValue().doubleValue());
            sameDiff.associateArrayWithVariable(arr,this);
        }
        else if(getShape() == null)
            return null;

        else {
            INDArray newAlloc = getWeightInitScheme().create(getShape());
            sameDiff.associateArrayWithVariable(newAlloc,this);

        }

        return sameDiff.getArrForVarName(getVarName());
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
        return sameDiff.getGradForVariable(getVarName());
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new ND4JIllegalStateException("Unable to differentiate a variable! Must be a function.");
    }





    /**
     * Returns the shape of this variable
     * @return
     */
    public int[] getShape() {
        return sameDiff.getShapeForVarName(getVarName());
    }



    /**
     *
     * @return
     */
    public SDVariable dup() {
        return sameDiff.var(this);
    }



    //scalars

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rsub(double sameDiffVariable) {
        return rsub(sameDiff.generateVariableName("rsub",false,this),sameDiffVariable);
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rdiv(double sameDiffVariable) {
        return rdiv(sameDiff.generateVariableName("rdiv",false,this),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable add(double sameDiffVariable) {
        return add(sameDiff.generateVariableName("add",false,this),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable sub(double sameDiffVariable) {
        return sub(sameDiff.generateVariableName("sub",false,this),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable div(double sameDiffVariable) {
        return div(sameDiff.generateVariableName("div",false,this),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable mul(double sameDiffVariable) {
        return mul(sameDiff.generateVariableName("mul",false,this),sameDiffVariable);

    }


    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rsubi(double sameDiffVariable) {
        return rsubi(sameDiff.generateVariableName("rsubi",false,this),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rdivi(double sameDiffVariable) {
        return rdivi(sameDiff.generateVariableName("rdivi",false,this),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable addi(double sameDiffVariable) {
        return addi(sameDiff.generateVariableName("addi",false,this),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable subi(double sameDiffVariable) {
        return subi(sameDiff.generateVariableName("subi",false,this),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable divi(double sameDiffVariable) {
        return divi(sameDiff.generateVariableName("divi",false,this),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable muli(double sameDiffVariable) {
        return muli(sameDiff.generateVariableName("muli",false,this),sameDiffVariable);

    }



    //end scalars


    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rsub(SDVariable sameDiffVariable) {
        return rsub(sameDiff.generateVariableName("rsub",false,this,sameDiffVariable),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rdiv(SDVariable sameDiffVariable) {
        return rdiv(sameDiff.generateVariableName("rdiv",false,this,sameDiffVariable),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable add(SDVariable sameDiffVariable) {
        return add(sameDiff.generateVariableName("add",false,this,sameDiffVariable),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable sub(SDVariable sameDiffVariable) {
        return sub(sameDiff.generateVariableName("sub",false,this,sameDiffVariable),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable div(SDVariable sameDiffVariable) {
        return div(sameDiff.generateVariableName("div",false,this,sameDiffVariable),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable mul(SDVariable sameDiffVariable) {
        return mul(sameDiff.generateVariableName("mul",false,this,sameDiffVariable),sameDiffVariable);

    }


    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rsubi(SDVariable sameDiffVariable) {
        return rsubi(sameDiff.generateVariableName("rsubi",false,this,sameDiffVariable),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rdivi(SDVariable sameDiffVariable) {
        return rdivi(sameDiff.generateVariableName("rdivi",false,this,sameDiffVariable),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable addi(SDVariable sameDiffVariable) {
        return addi(sameDiff.generateVariableName("addi",false,this,sameDiffVariable),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable subi(SDVariable sameDiffVariable) {
        return subi(sameDiff.generateVariableName("subi",false,this,sameDiffVariable),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable divi(SDVariable sameDiffVariable) {
        return divi(sameDiff.generateVariableName("divi",false,this,sameDiffVariable),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable muli(SDVariable sameDiffVariable) {
        return muli(sameDiff.generateVariableName("muli",false,this,sameDiffVariable),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rsub(String varName, double sameDiffVariable) {
        val function = sameDiff.f().rsub(this,sameDiffVariable);
        return sameDiff.updateVariableNameAndReference(function,varName);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rdiv(String varName, double sameDiffVariable) {
        val function = sameDiff.f().rdiv(this,sameDiffVariable);
        return sameDiff.updateVariableNameAndReference(function,varName);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable add(String varName, double sameDiffVariable) {
        val function = sameDiff.f().add(this,sameDiffVariable);
        return sameDiff.updateVariableNameAndReference(function,varName);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable sub(String varName, double sameDiffVariable) {
        SDVariable right = this;
        val result = sameDiff.f().sub(right,sameDiffVariable);
        return sameDiff.updateVariableNameAndReference(result,varName);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable div(String varName, double sameDiffVariable) {
        val function = sameDiff.f().div(this,sameDiffVariable);
        return sameDiff.updateVariableNameAndReference(function,varName);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable mul(String varName, double sameDiffVariable) {
        val function = sameDiff.f().mul(this
                , sameDiffVariable);
        return sameDiff.updateVariableNameAndReference(function,varName);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rsubi(String varName, double sameDiffVariable) {
        val function = sameDiff.f().rsubi(this,sameDiffVariable);
        return sameDiff.updateVariableNameAndReference(function,varName);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rdivi(String varName, double sameDiffVariable) {
        SDVariable function = sameDiff.f().rdivi(this
                ,sameDiffVariable);
        return sameDiff.updateVariableNameAndReference(function,varName);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable addi(String varName, double sameDiffVariable) {
        val function = sameDiff.f().addi(this,sameDiffVariable);
        return sameDiff.updateVariableNameAndReference(function,varName);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable subi(String varName, double sameDiffVariable) {
        val function = sameDiff.f().subi(this,sameDiffVariable);
        return sameDiff.updateVariableNameAndReference(function,varName);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable divi(String varName, double sameDiffVariable) {
        val function = sameDiff.f().divi(this,sameDiffVariable);
        return sameDiff.updateVariableNameAndReference(function,varName);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable muli(String varName, double sameDiffVariable) {
        val function = sameDiff.f().muli(this,sameDiffVariable);
        return sameDiff.updateVariableNameAndReference(function,varName);

    }



    //end scalars


    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rsub(String varName, SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);
        val result = sameDiff.f().rsub(this,sameDiffVariable);
        return sameDiff.updateVariableNameAndReference(result,varName);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rdiv(String varName, SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);
        val result = sameDiff.f().rdiv(this,sameDiffVariable);
        return sameDiff.updateVariableNameAndReference(result,varName);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable add(String varName, SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);
        val result = sameDiff.f().add(this,sameDiffVariable);
        return sameDiff.updateVariableNameAndReference(result,varName);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable sub(String varName, SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);

        SDVariable left = this;
        SDVariable right = sameDiffVariable;
        val result = sameDiff.f().sub(left,right);
        return sameDiff.updateVariableNameAndReference(result,varName);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable div(String varName, SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);
        val result = sameDiff.f().div(this,sameDiffVariable);
        return sameDiff.updateVariableNameAndReference(result,varName);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable mul(String varName, SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);

        SDVariable left = this;
        SDVariable right = sameDiffVariable;
        Preconditions.checkState(left != null,"Left input is null!");
        Preconditions.checkState(right != null,"Right input is null!");

        val result = sameDiff.f().mul(left,right);
        return sameDiff.updateVariableNameAndReference(result,varName);

    }


    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rsubi(String varName, SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);
        val result = sameDiff.f().rsubi(this,sameDiffVariable);
        return sameDiff.updateVariableNameAndReference(result,varName);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rdivi(String varName, SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);
        val result = sameDiff.f().rdivi(this,sameDiffVariable);
        return sameDiff.updateVariableNameAndReference(result,varName);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable addi(String varName, SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);
        val result = sameDiff.f().addi(this,sameDiffVariable);
        return sameDiff.updateVariableNameAndReference(result,varName);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable subi(String varName, SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);
        SDVariable left = this;
        SDVariable right = sameDiffVariable;
        val result = sameDiff.f().subi(left,right);
        return sameDiff.updateVariableNameAndReference(result,varName);
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable divi(String varName, SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);
        val result = sameDiff.f().divi(this,sameDiffVariable);
        result.setVarName(varName);
        return result;
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable muli(String varName, SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);

        SDVariable left = this;
        SDVariable right = sameDiffVariable;
        SDVariable result = sameDiff.f().muli(left,right);
        result.setVarName(varName);
        return result;
    }



    /**
     * Evaluate the result of this variable
     * @return
     */
    public INDArray eval() {
        SameDiff exec = sameDiff.dup();
        exec.defineFunction("output", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                return new SDVariable[] { SDVariable.this};
            }
        });

        SDVariable output = exec.invokeFunctionOn("output",exec);
        return output.getSameDiff().execAndEndResult();
    }




    private void assertShapeEquals(SDVariable variable) {
        val shape = sameDiff.getShapeForVarName(getVarName());
        if(shape == null)
            throw new ND4JIllegalStateException("Shape not found for variable " + getVarName());

        if(!Arrays.equals(shape,variable.getShape()) && ArrayUtil.prod(variable.getShape()) != 1 && Shape.broadcastOutputShape(shape,variable.getShape()) == null) {
            throw new IllegalArgumentException("Input shape must be the same as this shape " + Arrays.toString(shape) + " and shape was " + Arrays.toString(variable.getShape()));
        }
    }



    @Override
    public String toString() {
        return varName;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        if (!super.equals(o)) return false;

        SDVariable that = (SDVariable) o;

        if (varName != null ? !varName.equals(that.varName) : that.varName != null) return false;
        return weightInitScheme != null ? weightInitScheme.equals(that.weightInitScheme) : that.weightInitScheme == null;
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + (varName != null ? varName.hashCode() : 0);
        result = 31 * result + (weightInitScheme != null ? weightInitScheme.hashCode() : 0);
        return result;
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }



}

package org.nd4j.autodiff.samediff;

import com.google.common.annotations.VisibleForTesting;
import lombok.*;
import onnx.OnnxProto3;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.builder.Diff;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.*;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.weightinit.WeightInitScheme;
import org.nd4j.weightinit.impl.ZeroInitScheme;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

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




    @Builder
    private SDVariable(String varName,
                       SameDiff sameDiff,
                       long[] shape,
                       WeightInitScheme weightInitScheme) {
        super(sameDiff,new Object[]{});
        this.varName = varName;
        this.weightInitScheme = weightInitScheme;

        if(weightInitScheme == null) {
            // we want C order as default in ALL cases
            this.weightInitScheme = new ZeroInitScheme('c');
        }

        if(shape == null) {
            sameDiff.addAsPlaceHolder(varName);
        }

        else {
            boolean foundPlaceHolder = false;
            for(int i = 0; i < shape.length; i++) {
                if(shape[i] < 0) {
                    sameDiff.addAsPlaceHolder(varName);
                    sameDiff.setOriginalPlaceHolderShape(varName, shape);
                    foundPlaceHolder = true;
                    break;
                }
            }

            if(!foundPlaceHolder && shape != null)
                sameDiff.putShapeForVarName(varName,shape);
        }

        this.sameDiff = sameDiff;


    }

    /**
     * Returns true if this variable is a place holder
     * @return
     */
    public boolean isPlaceHolder() {
        return sameDiff.isPlaceHolder(varName);
    }


    @Override
    public String opName() {
        return "variable";
    }

    @Override
    public SDVariable[] outputVariables() {
        return new SDVariable[] {this};
    }

    @Override
    public SDVariable arg() {
        return this;
    }

    @Override
    public SDVariable[] args() {
        return new SDVariable[] {this};
    }

    @Override
    public SDVariable[] outputVariables(String baseName) {
        return new SDVariable[] {this};
    }




    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {

    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {

    }



    /**
     * Allocate and return a  new array
     * based on the vertex id and weight initialization.
     * @return the allocated array
     */
    public INDArray storeAndAllocateNewArray() {
        val shape = sameDiff.getShapeForVarName(getVarName());
        if(getArr() != null && Arrays.equals(getArr().shape(),shape))
            return getArr();

        if(varName == null)
            throw new ND4JIllegalStateException("Unable to store array for null variable name!");

        if(shape == null) {
            throw new ND4JIllegalStateException("Unable to allocate new array. No shape found for variable " + varName);
        }

        val arr = getWeightInitScheme().create(shape);
        sameDiff.putArrayForVarName(getVarName(),arr);
        return arr;
    }

    /**
     * A getter for the allocated ndarray with this {@link SDVariable}.
     *
     * This getter will lazy initialize an array if one is not found based on the associated shape and
     * {@link WeightInitScheme} - if this is possible. If this is not possible (due to shapes being unknown, etc)
     * null is returned
     *
     * @return the {@link INDArray} associated with this variable.
     */
    public INDArray getArr() {
        return getArr(false);
    }

    /**
     * A getter for the allocated ndarray with this {@link SDVariable}.
     *
     * This getter will lazy initialize an array if one is not found based on the associated shape and
     * {@link WeightInitScheme} - if this is possible.<br>
     * If this is not possible (due to shapes being unknown, etc) either:<br>
     * (a) null is returned - if enforceExistence == false, or<br>
     * (b) an IllegalStateException is thrown, if enforceExistence == true
     *
     * @return the {@link INDArray} associated with this variable.
     */
    public INDArray getArr(boolean enforceExistence){
        if(sameDiff.arrayAlreadyExistsForVarName(getVarName()))
            return sameDiff.getArrForVarName(getVarName());

        //initialize value if it's actually a scalar constant (zero or 1 typically...)
        if(getScalarValue() != null && ArrayUtil.prod(getShape()) == 1) {
            INDArray arr = Nd4j.valueArrayOf(getShape(),
                    getScalarValue().doubleValue());
            sameDiff.associateArrayWithVariable(arr,this);
        }
        else if(sameDiff.getShapeForVarName(getVarName()) == null) {
            if (enforceExistence) {
                throw new IllegalStateException("Cannot get array for SDVariable \"" + getVarName() + "\": no array has" +
                        " been defined, and array shape cannot be calculated");
            }
            return null;
        } else {
            long[] shape = sameDiff.getShapeForVarName(getVarName());
            INDArray newAlloc = getWeightInitScheme().create(shape);
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
    public long[] getShape() {
        long[] initialShape =  sameDiff.getShapeForVarName(getVarName());
        if(initialShape == null) {
            val arr = getArr();
            if(arr != null)
                return arr.shape();
        }

        return initialShape;
    }



    /**
     *
     * @return
     */
    public SDVariable dup() {
        return sameDiff.var(this);
    }

    public SDVariable assign(Number value){
        return sameDiff.scalarSet(this, value);
    }

    /**
     * Negate op
     * @return Negated variable
     */
    public SDVariable neg(){
        return f().neg(this);
    }

    /**
     * Negate op
     * @return Negated variable
     */
    public SDVariable neg(String name){
        return sameDiff.neg(name, this);
    }

    public SDVariable lt(double value){
        return lt(null, value);
    }

    public SDVariable lt(String name, double value){
        return sameDiff.lt(name, this, value);
    }

    public SDVariable lte(double value){
        return lte(null, value);
    }

    public SDVariable lte(String name, double value){
        return sameDiff.lte(name, this, value);
    }

    public SDVariable gt(double value){
        return gt(null, value);
    }

    public SDVariable gt(String name, double value){
        return sameDiff.gt(name, this, value);
    }

    public SDVariable gte(double value){
        return gte(null, value);
    }

    public SDVariable gte(String name, double value){
        return sameDiff.gte(name, this, value);
    }


    public SDVariable eq(double value){
        return eq(null, value);
    }

    public SDVariable eq(String name, double value){
        return sameDiff.eq(name, this, value);
    }

    public SDVariable neq(double value){
        return neq(null, value);
    }

    public SDVariable neq(String name, double value){
        return sameDiff.neq(name, this, value);
    }





    public SDVariable lt(SDVariable other){
        return lt(null, other);
    }

    public SDVariable lt(String name, SDVariable other){
        return sameDiff.lt(name, this, other);
    }

    public SDVariable lte(SDVariable other){
        return lte(null, other);
    }

    public SDVariable lte(String name, SDVariable other){
        return sameDiff.lte(name, this, other);
    }

    public SDVariable gt(SDVariable other){
        return gt(null, other);
    }

    public SDVariable gt(String name, SDVariable other){
        return sameDiff.gt(name, this, other);
    }

    public SDVariable gte(SDVariable other){
        return gte(null, other);
    }

    public SDVariable gte(String name, SDVariable other){
        return sameDiff.gte(name, this, other);
    }


    public SDVariable eq(SDVariable other){
        return eq(null, other);
    }

    public SDVariable eq(String name, SDVariable other){
        return sameDiff.eq(name, this, other);
    }

    public SDVariable neq(SDVariable other){
        return neq(null, other);
    }

    public SDVariable neq(String name, SDVariable other){
        return sameDiff.neq(name, this, other);
    }


    //scalars

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rsub(double sameDiffVariable) {
        return rsub(sameDiff.generateNewVarName(RSubOp.OP_NAME,0),sameDiffVariable);
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rdiv(double sameDiffVariable) {
        return rdiv(sameDiff.generateNewVarName(RDivOp.OP_NAME,0),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable add(double sameDiffVariable) {
        return add(sameDiff.generateNewVarName(AddOp.OP_NAME,0),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable sub(double sameDiffVariable) {
        return sub(sameDiff.generateNewVarName(SubOp.OP_NAME,0),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable squaredDifference(SDVariable sameDiffVariable) {
        return squaredDifference(sameDiff.generateNewVarName(SquaredDifferenceOp.OP_NAME,0),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable div(double sameDiffVariable) {
        return div(sameDiff.generateNewVarName(DivOp.OP_NAME,0),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable mul(double sameDiffVariable) {
        return mul(sameDiff.generateNewVarName(MulOp.OP_NAME,0),sameDiffVariable);

    }


    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rsubi(double sameDiffVariable) {
        return rsubi(sameDiff.generateNewVarName(RSubOp.OP_NAME,0),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rdivi(double sameDiffVariable) {
        return rdivi(sameDiff.generateNewVarName(RDivOp.OP_NAME,0),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable addi(double sameDiffVariable) {
        return addi(sameDiff.generateNewVarName(AddOp.OP_NAME,0),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable subi(double sameDiffVariable) {
        return subi(sameDiff.generateNewVarName(SubOp.OP_NAME,0),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable divi(double sameDiffVariable) {
        return divi(sameDiff.generateNewVarName(DivOp.OP_NAME,0),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable muli(double sameDiffVariable) {
        return muli(sameDiff.generateNewVarName(MulOp.OP_NAME,0),sameDiffVariable);

    }



    //end scalars


    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rsub(SDVariable sameDiffVariable) {
        return rsub(sameDiff.generateNewVarName(RSubOp.OP_NAME,0),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rdiv(SDVariable sameDiffVariable) {
        return rdiv(sameDiff.generateNewVarName(RDivOp.OP_NAME,0),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable truncatedDiv(SDVariable sameDiffVariable) {
        return truncatedDiv(sameDiff.generateNewVarName(TruncateDivOp.OP_NAME,0),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable add(SDVariable sameDiffVariable) {
        return add(sameDiff.generateNewVarName(AddOp.OP_NAME,0),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable sub(SDVariable sameDiffVariable) {
        return sub(sameDiff.generateNewVarName(SubOp.OP_NAME,0),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable div(SDVariable sameDiffVariable) {
        return div(sameDiff.generateNewVarName(DivOp.OP_NAME,0),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable mul(SDVariable sameDiffVariable) {
        return mul(sameDiff.generateNewVarName(MulOp.OP_NAME,0),sameDiffVariable);

    }


    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rsubi(SDVariable sameDiffVariable) {
        return rsubi(sameDiff.generateNewVarName(RSubOp.OP_NAME,0),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rdivi(SDVariable sameDiffVariable) {
        return rdivi(sameDiff.generateNewVarName(RDivOp.OP_NAME,0),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable addi(SDVariable sameDiffVariable) {
        return addi(sameDiff.generateNewVarName(AddOp.OP_NAME,0),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable subi(SDVariable sameDiffVariable) {
        return subi(sameDiff.generateNewVarName(SubOp.OP_NAME,0),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable divi(SDVariable sameDiffVariable) {
        return divi(sameDiff.generateNewVarName(DivOp.OP_NAME,0),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable muli(SDVariable sameDiffVariable) {
        return muli(sameDiff.generateNewVarName(MulOp.OP_NAME,0),sameDiffVariable);

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
    public SDVariable truncatedDiv(String varName, SDVariable sameDiffVariable) {
        val function = sameDiff.f().truncatedDiv(this, sameDiffVariable);
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
     * @param value
     * @return
     */
    public SDVariable mul(String varName, double value) {
        val function = sameDiff.f().mul(this
                , value);
        return sameDiff.updateVariableNameAndReference(function,varName);
    }

    public SDVariable pow(double value){
        return pow(null, value);
    }

    public SDVariable pow(String varName, double value){
        SDVariable ret = f().pow(this, value);
        return sameDiff.updateVariableNameAndReference(ret, varName);
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
     * @return squared difference between variables
     */
    public SDVariable squaredDifference(String varName, SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);

        SDVariable left = this;
        SDVariable right = sameDiffVariable;
        val result = sameDiff.f().squaredDifference(left, right);
        return sameDiff.updateVariableNameAndReference(result, varName);
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
        Preconditions.checkNotNull(left,"Left input is null!");
        Preconditions.checkNotNull(right,"Right input is null!");

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

    @Override
    public Op.Type opType() {
        return Op.Type.RETURN;
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

    public SDVariable sum(int... dimensions){
        return sum(null, dimensions);
    }

    public SDVariable sum(boolean keepDims, int... dimensions){
        return sum(null, dimensions);
    }

    public SDVariable sum(String name, int... dimensions){
        return sum(name, false, dimensions);
    }

    public SDVariable sum(String name, boolean keepDims, int... dimensions){
        return sameDiff.sum(name, this, dimensions);
    }

    public SDVariable mean(boolean keepDims, int... dimensions){
        return mean(null, keepDims, dimensions);
    }

    public SDVariable mean(String name, int... dimensions){
        return mean(name, false, dimensions);
    }

    public SDVariable mean(int... dimensions){
        return mean(null, false, dimensions);
    }

    public SDVariable mean(String name, boolean keepDims, int... dimensions){
        return sameDiff.mean(name, this, keepDims, dimensions);
    }

    public SDVariable std(boolean biasCorrected, int... dimensions){
        return std(null, biasCorrected, dimensions);
    }

    public SDVariable std(String name, boolean biasCorrected, int... dimensions){
        return sameDiff.standardDeviation(name, this, biasCorrected, dimensions);
    }

    public SDVariable std(String name, boolean biasCorrected, boolean keepDims, int... dimensions){
        return sameDiff.standardDeviation(name, this, biasCorrected, keepDims, dimensions);
    }

    public SDVariable prod(int... dimensions){
        return prod(null, dimensions);
    }

    public SDVariable prod(boolean keepDims, int... dimensions){
        return prod(null, keepDims, dimensions);
    }

    public SDVariable prod(String name, int... dimensions){
        return sameDiff.prod(name, this, dimensions);
    }

    public SDVariable prod(String name, boolean keepDims, int... dimensions){
        return sameDiff.prod(name, this, keepDims, dimensions);
    }

    public SDVariable min(int... dimensions){
        return min(null, dimensions);
    }

    public SDVariable min(boolean keepDims, int... dimensions){
        return min(null, keepDims, dimensions);
    }

    public SDVariable min(String name, int... dimensions){
        return min(name, false, dimensions);
    }

    public SDVariable min(String name, boolean keepDims, int... dimensions){
        return sameDiff.min(name, this, keepDims, dimensions);
    }

    public SDVariable max(int... dimensions){
        return max(null, dimensions);
    }

    public SDVariable max(boolean keepDims, int... dimensions){
        return max(null, keepDims, dimensions);
    }

    public SDVariable max(String name, int... dimensions){
        return max(name, false, dimensions);
    }

    public SDVariable max(String name, boolean keepDims, int... dimensions){
        return sameDiff.max(name, this, keepDims, dimensions);
    }

    public SDVariable norm1(int... dimensions){
        return norm1(null, dimensions);
    }

    public SDVariable norm1(boolean keepDims, int... dimensions){
        return norm1(null, keepDims, dimensions);
    }

    public SDVariable norm1(String name, int... dimensions){
        return norm1(name, false, dimensions);
    }

    public SDVariable norm1(String name, boolean keepDims, int... dimensions){
        return sameDiff.norm1(name, this, keepDims, dimensions);
    }

    public SDVariable norm2(int... dimensions){
        return norm2(null, dimensions);
    }

    public SDVariable norm2(boolean keepDims, int... dimensions){
        return norm2(null, keepDims, dimensions);
    }

    public SDVariable norm2(String name, int... dimensions){
        return norm2(name, false, dimensions);
    }

    public SDVariable norm2(String name, boolean keepDims, int... dimensions){
        return sameDiff.norm2(name, this, keepDims, dimensions);
    }

    public SDVariable normmax(int... dimensions){
        return normmax(null, dimensions);
    }

    public SDVariable normmax(boolean keepDims, int... dimensions){
        return normmax(null, keepDims, dimensions);
    }

    public SDVariable normmax(String name, int... dimensions){
        return normmax(name, false, dimensions);
    }

    public SDVariable normmax(String name, boolean keepDims, int... dimensions){
        return sameDiff.normmax(name, this, keepDims, dimensions);
    }

    public SDVariable argmax(int... dimensions){
        return argmax(null, dimensions);
    }

    public SDVariable argmax(String name, int... dimensions){
        return sameDiff.argmax(name, this, dimensions);
    }

    public SDVariable argmin(int... dimensions){
        return argmin(null, dimensions);
    }

    public SDVariable argmin(String name, int... dimensions){
        return sameDiff.argmin(name, this, dimensions);
    }


    public SDVariable setArray(INDArray array){
        sameDiff.associateArrayWithVariable(array, this);
        return this;
    }


    /**
     * Evaluate the result of this variable
     * @return
     */
    public INDArray eval() {
        sameDiff.exec();
        return getArr();
    }

  private int outputIndex = 0;

  private DifferentialFunction creator;


    private void assertShapeEquals(SDVariable variable) {
       /* val shape = sameDiff.getShapeForVarName(getVarName());
        if(shape == null && !variable.isPlaceHolder())
            throw new ND4JIllegalStateException("Shape not found for variable " + getVarName());

        if(!Arrays.equals(shape,variable.getShape()) && ArrayUtil.prod(variable.getShape()) != 1 && Shape.broadcastOutputShape(shape,variable.getShape()) == null) {
            throw new IllegalArgumentException("Input shape must be the same as this shape " + Arrays.toString(shape) + " and shape was " + Arrays.toString(variable.getShape()));
        }*/
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

   public SDVariable get(SDIndex... indices){
       int ndims = indices.length;
       long[] begin = new long[ndims];
       long[] end = new long[ndims];
       long[] strides = new long[ndims];
       int[] begin_mask_arr = new int[ndims];
       int[] end_mask_arr = new int[ndims];
       int[] shrink_axis_mask_arr = new int[ndims];
       for(int i=0; i<ndims; i++){
           strides[i] = 1;
           SDIndex index = indices[i];
           SDIndex.IndexType indexType = index.getIndexType();
           if(indexType == SDIndex.IndexType.ALL){
               begin_mask_arr[i] = 1;
               end_mask_arr[i] = 1;
           }
           else if(indexType == SDIndex.IndexType.POINT){
                   long pointIndex = index.getPointIndex();
                   begin[i] = pointIndex;
                   end[i] = pointIndex + 1;
                   shrink_axis_mask_arr[i] = 1;
           }
           else if(indexType == SDIndex.IndexType.INTERVAL){
               if(index.getIntervalBegin() == null){
                   begin_mask_arr[i] = 1;
               }
               else{
                   begin[i] = index.getIntervalBegin();
               }
               if(index.getIntervalEnd() == null){
                   end_mask_arr[i] = 1;
               }
               else{
                   end[i] = index.getIntervalEnd();
               }
               if(index.getIntervalStrides() == null){
                   strides[i] = 1;
               }
               else{
                   strides[i] = index.getIntervalStrides();
               }
           }
       }

       // convert binary int[] to int
        int begin_mask = binArrToInt(begin_mask_arr);
       int end_mask = binArrToInt(end_mask_arr);
       int shrink_axis = binArrToInt(shrink_axis_mask_arr);

       return this.sameDiff.stridedSlice(this, begin, end, strides,
               begin_mask, end_mask, 0, 0, shrink_axis);
   }


   private static int binArrToInt(int[] arr){
        int x = 0;
        int m = 1;
        for(int i = 0; i < arr.length; i++){
            if(arr[i] == 1){
                x += m;
            }
            m *= 2;
        }
        return x;
   }
   
}

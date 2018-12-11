/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.autodiff.samediff;

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.*;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
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
@Slf4j
public class SDVariable extends DifferentialFunction implements Serializable {


    @Getter
    @Setter
    private String varName;
    @Getter
    @Setter
    private VariableType variableType;

    @Getter
    @Setter
    protected WeightInitScheme weightInitScheme;

    @Getter (AccessLevel.NONE)
    @Setter
    protected DataType dataType;

    private int outputIndex = 0;

    private DifferentialFunction creator;

    // autogen_tag::sdvars::start


    public SDVariable(@NonNull String varName, @NonNull VariableType varType, @NonNull SameDiff sameDiff, long[] shape, DataType dataType, WeightInitScheme weightInitScheme){
        super(sameDiff, new Object[0]);
        Preconditions.checkState(weightInitScheme == null || varType == VariableType.VARIABLE, "Weight initalization schemes can only be applied to VARIABLE type" +
                " SDVariables - variable \"%s\" is of type %s but was provided a weight initialization scheme %s", varName, varType, weightInitScheme);

        this.varName = varName;
        this.variableType = varType;
        this.dataType = dataType;
        this.weightInitScheme = weightInitScheme;

        if(varType == VariableType.PLACEHOLDER){
            sameDiff.setOriginalPlaceHolderShape(varName, shape);
            sameDiff.putShapeForVarName(varName, shape);
        }
    }

    /**
     * Returns true if this variable is a place holder
     * @return
     */
    public boolean isPlaceHolder() {
        return variableType == VariableType.PLACEHOLDER;
    }

    public boolean isConstant(){
        return variableType == VariableType.CONSTANT;
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
        Preconditions.checkState(variableType == VariableType.VARIABLE, "Unable to allocate and store array for variable of type %s: only" +
                " VARIABLE type variables can be initialized using this method", variableType);
        val shape = sameDiff.getShapeForVarName(getVarName());
        INDArray currArr = getArr();
        if(currArr != null && Arrays.equals(currArr.shape(),shape))
            return getArr();

        if(varName == null)
            throw new ND4JIllegalStateException("Unable to store array for null variable name!");

        if(shape == null) {
            throw new ND4JIllegalStateException("Unable to allocate new array. No shape found for variable " + varName);
        }

        val arr = getWeightInitScheme().create(dataType(), shape);
        sameDiff.associateArrayWithVariable(arr, this);
        if(log.isTraceEnabled()){
            log.trace("Generated and stored new array for variable \"{}\": old shape: {}, new shape {}", getVarName(),
                    (currArr == null ? "null" : Arrays.toString(currArr.shape())), Arrays.toString(arr.shape()));
        }
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


    // autogen_tag::sdvars::end
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
            INDArray arr = Nd4j.valueArrayOf(getShape(),getScalarValue().getDouble(0));
            sameDiff.associateArrayWithVariable(arr,this);
            if(log.isTraceEnabled()){
                log.trace("getArr() for variable \"{}\" allocated new scalar array: shape {}", getVarName(), Arrays.toString(getShape()));
            }
        }
        else if(sameDiff.getShapeForVarName(getVarName()) == null) {
            if (enforceExistence) {
                throw new IllegalStateException("Cannot get array for SDVariable \"" + getVarName() + "\": no array has" +
                        " been defined, and array shape cannot be calculated");
            }
            if(log.isTraceEnabled()){
                log.trace("SDVariable.getArr(): could not get array for variable {}: shape is null", getVarName());
            }
            return null;
        } else {
            long[] shape = sameDiff.getShapeForVarName(getVarName());
            INDArray newAlloc = getWeightInitScheme().create(dataType(), shape);
            sameDiff.associateArrayWithVariable(newAlloc,this);
            if(log.isTraceEnabled()){
                log.trace("getArr() for variable \"{}\" allocated new array with shape {}", getVarName(), Arrays.toString(getShape()));
            }
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


/*
    public DataType dataType() {
        throw new UnsupportedOperationException();
    }

*/
    /**
     * Returns the shape of this variable
     * @return Shape of the variable
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

    public DataType dataType() {
        if(this.dataType == null){
            //Try to infer datatype instead of returning null
            if(getArr() != null){
                this.dataType = getArr().dataType();
            }
        }

        return this.dataType;
    }

    public LongShapeDescriptor getShapeDescriptor() {
        return LongShapeDescriptor.fromShape(getShape(), this.dataType());
    }


    /**
     * Create a new SDVariable, the contents of which is copied from this current variable
     * @return The new variable
     */
    public SDVariable dup() {
        return sameDiff.var(this);
    }

    /**
     * Return a variable with equal shape to the input, but all elements set to the specified value
     *
     * @param value Value for returned variable
     * @return new variable
     */
    public SDVariable assign(Number value){
        return sameDiff.scalarSet(this, value);
    }

    /**
     * Negate op - returns a new variable with the values of the current variable negated
     * @return Negated variable
     */
    public SDVariable neg(){
        return f().neg(this);
    }

    /**
     * Negate op - returns a new variable with the values of the current variable negated
     * @param name Name of the new variable
     * @return Negated variable
     */
    public SDVariable neg(String name){
        return sameDiff.neg(name, this);
    }

    /**
     * See {@link #lt(String, double)}
     */
    public SDVariable lt(double value){
        return lt(null, value);
    }

    /**
     * Less than operation: elementwise {@code this < value}<br>
     * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
     * value 0 otherwise
     *
     * @param name  Name of the output variable
     * @param value value argument to use in operation
     * @return Output SDVariable with values 0 (not satisfied) and 1 (where the condition is satisfied)
     */
    public SDVariable lt(String name, double value){
        return sameDiff.lt(name, this, value);
    }

    /**
     * See {@link #lte(String, double)}
     */
    public SDVariable lte(double value){
        return lte(null, value);
    }

    /**
     * Less than or equals operation: elementwise {@code this <= value}<br>
     * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
     * value 0 otherwise
     *
     * @param name  Name of the output variable
     * @param value value argument to use in operation
     * @return Output SDVariable with values 0 (not satisfied) and 1 (where the condition is satisfied)
     */
    public SDVariable lte(String name, double value){
        return sameDiff.lte(name, this, value);
    }

    /**
     * See {@link #gt(String, double)}
     */
    public SDVariable gt(double value){
        return gt(null, value);
    }

    /**
     * Greater than operation: elementwise {@code this > value}<br>
     * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
     * value 0 otherwise
     *
     * @param name  Name of the output variable
     * @param value value argument to use in operation
     * @return Output SDVariable with values 0 (not satisfied) and 1 (where the condition is satisfied)
     */
    public SDVariable gt(String name, double value){
        return sameDiff.gt(name, this, value);
    }

    /**
     * See {@link #gte(String, double)}
     */
    public SDVariable gte(double value){
        return gte(null, value);
    }

    /**
     * Greater than or equals operation: elementwise {@code this >= value}<br>
     * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
     * value 0 otherwise
     *
     * @param name  Name of the output variable
     * @param value value argument to use in operation
     * @return Output SDVariable with values 0 (not satisfied) and 1 (where the condition is satisfied)
     */
    public SDVariable gte(String name, double value){
        return sameDiff.gte(name, this, value);
    }

    /**
     * See {@link #eq(String, double)}
     */
    public SDVariable eq(double value){
        return eq(null, value);
    }

    /**
     * Equals operation: elementwise {@code this == value}<br>
     * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
     * value 0 otherwise
     *
     * @param name  Name of the output variable
     * @param value value argument to use in operation
     * @return Output SDVariable with values 0 (not satisfied) and 1 (where the condition is satisfied)
     */
    public SDVariable eq(String name, double value){
        return sameDiff.eq(name, this, value);
    }

    /**
     * See {@link #neq(SDVariable)}
     */
    public SDVariable neq(double value){
        return neq(null, value);
    }

    /**
     * Not equals operation: elementwise {@code this != value}<br>
     * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
     * value 0 otherwise
     *
     * @param name  Name of the output variable
     * @param value value argument to use in operation
     * @return Output SDVariable with values 0 (not satisfied) and 1 (where the condition is satisfied)
     */
    public SDVariable neq(String name, double value){
        return sameDiff.neq(name, this, value);
    }


    /**
     * See {@link #lt(String, SDVariable)}
     */
    public SDVariable lt(SDVariable other){
        return lt(null, other);
    }

    /**
     * Less than operation: elementwise {@code this < y}<br>
     * If x and y arrays have equal shape, the output shape is the same as the inputs.<br>
     * Supports broadcasting: if x and y have different shapes and are broadcastable, the output shape is broadcast.<br>
     * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
     *
     * @param name  Name of the output variable
     * @param other Variable to compare values against
     * @return Output SDVariable with values 0 (not satisfied) and 1 (where the condition is satisfied)
     */
    public SDVariable lt(String name, SDVariable other){
        return sameDiff.lt(name, this, other);
    }

    /**
     * See {@link #lte(String, SDVariable)}
     */
    public SDVariable lte(SDVariable other){
        return lte(null, other);
    }

    /**
     * Less than or equal to operation: elementwise {@code this <= y}<br>
     * If x and y arrays have equal shape, the output shape is the same as the inputs.<br>
     * Supports broadcasting: if x and y have different shapes and are broadcastable, the output shape is broadcast.<br>
     * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
     *
     * @param name  Name of the output variable
     * @param other Variable to compare values against
     * @return Output SDVariable with values 0 (not satisfied) and 1 (where the condition is satisfied)
     */
    public SDVariable lte(String name, SDVariable other){
        return sameDiff.lte(name, this, other);
    }

    /**
     * See {@link #gt(String, SDVariable)}
     */
    public SDVariable gt(SDVariable other){
        return gt(null, other);
    }

    /**
     * Greater than operation: elementwise {@code this > y}<br>
     * If x and y arrays have equal shape, the output shape is the same as the inputs.<br>
     * Supports broadcasting: if x and y have different shapes and are broadcastable, the output shape is broadcast.<br>
     * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
     *
     * @param name  Name of the output variable
     * @param other Variable to compare values against
     * @return Output SDVariable with values 0 (not satisfied) and 1 (where the condition is satisfied)
     */
    public SDVariable gt(String name, SDVariable other){
        return sameDiff.gt(name, this, other);
    }

    /**
     * See {@link #gte(String, SDVariable)}
     */
    public SDVariable gte(SDVariable other){
        return gte(null, other);
    }

    /**
     * Greater than or equal to operation: elementwise {@code this >= y}<br>
     * If x and y arrays have equal shape, the output shape is the same as the inputs.<br>
     * Supports broadcasting: if x and y have different shapes and are broadcastable, the output shape is broadcast.<br>
     * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
     *
     * @param name  Name of the output variable
     * @param other Variable to compare values against
     * @return Output SDVariable with values 0 (not satisfied) and 1 (where the condition is satisfied)
     */
    public SDVariable gte(String name, SDVariable other){
        return sameDiff.gte(name, this, other);
    }

    /**
     * See {@link #eq(String, SDVariable)}
     */
    public SDVariable eq(SDVariable other){
        return eq(null, other);
    }

    /**
     * Equal to operation: elementwise {@code this == y}<br>
     * If x and y arrays have equal shape, the output shape is the same as the inputs.<br>
     * Supports broadcasting: if x and y have different shapes and are broadcastable, the output shape is broadcast.<br>
     * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
     *
     * @param name  Name of the output variable
     * @param other Variable to compare values against
     * @return Output SDVariable with values 0 (not satisfied) and 1 (where the condition is satisfied)
     */
    public SDVariable eq(String name, SDVariable other){
        return sameDiff.eq(name, this, other);
    }

    /**
     * See {@link #neq(String, SDVariable)}
     */
    public SDVariable neq(SDVariable other){
        return neq(null, other);
    }

    /**
     * Not equal to operation: elementwise {@code this != y}<br>
     * If x and y arrays have equal shape, the output shape is the same as the inputs.<br>
     * Supports broadcasting: if x and y have different shapes and are broadcastable, the output shape is broadcast.<br>
     * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
     *
     * @param name  Name of the output variable
     * @param other Variable to compare values against
     * @return Output SDVariable with values 0 (not satisfied) and 1 (where the condition is satisfied)
     */
    public SDVariable neq(String name, SDVariable other){
        return sameDiff.neq(name, this, other);
    }

    /**
     * See {@link #mmul(String, SDVariable)}
     */
    public SDVariable mmul(SDVariable other){
        return mmul(null, other);
    }

    /**
     * Matrix multiplication: out = mmul(this,other)
     *
     * @param name  Name of the output variable
     * @param other Other variable to perform matrix multiplication with
     * @return Output variable (result of mmul)
     */
    public SDVariable mmul(String name, SDVariable other){
        return sameDiff.mmul(name, this, other);
    }

    /**
     * Matrix multiplication: out = mmul(this,other)
     *
     * @param name          Name of the output variable
     * @param other         Other variable to perform matrix multiplication with
     * @param mMulTranspose Matrix transpose configuration
     * @return Output variable (result of mmul)
     */
    public SDVariable mmul(String name, SDVariable other, @NonNull MMulTranspose mMulTranspose) {
        return sameDiff.mmul(name, this, other, mMulTranspose);
    }


    /**
     * See {@link #add(String, double)}
     */
    public SDVariable add(double scalar) {
        return add(sameDiff.generateNewVarName(AddOp.OP_NAME,0),scalar);
    }

    /**
     * Scalar addition: {@code out = this + scalar}<br>
     * Output variable has the same shape as the input variable
     *
     * @param varName Output variable name
     * @param scalar  Scalar for operation
     * @return Output variable
     */
    public SDVariable add(String varName, double scalar) {
        val function = sameDiff.f().add(this,scalar);
        return sameDiff.updateVariableNameAndReference(function,varName);
    }

    /**
     * See {@link #add(String, SDVariable)}
     */
    public SDVariable add(SDVariable other) {
        return add(sameDiff.generateNewVarName(AddOp.OP_NAME,0),other);
    }

    /**
     * Addition operation: elementwise {@code this + x}<br>
     * If this and x variables have equal shape, the output shape is the same as the inputs.<br>
     * Supports broadcasting: if this and x have different shapes and are broadcastable, the output shape is broadcast.
     *
     * @param name Name of the output variable
     * @param x    Variable to perform operation with
     * @return Output (result) SDVariable
     */
    public SDVariable add(String name, SDVariable x) {
        val result = sameDiff.f().add(this, x);
        return sameDiff.updateVariableNameAndReference(result, name);
    }

    /**
     * See {@link #sub(String, double)}
     */
    public SDVariable sub(double scalar) {
        return sub(sameDiff.generateNewVarName(SubOp.OP_NAME,0),scalar);
    }

    /**
     * Scalar subtraction: {@code out = this - scalar}<br>
     * Output variable has the same shape as the input variable
     *
     * @param varName Output variable name
     * @param scalar  Scalar for operation
     * @return Output variable
     */
    public SDVariable sub(String varName, double scalar) {
        val result = sameDiff.f().sub(this, scalar);
        return sameDiff.updateVariableNameAndReference(result, varName);
    }

    /**
     * See {@link #sub(String, SDVariable)}
     */
    public SDVariable sub(SDVariable x) {
        return sub(sameDiff.generateNewVarName(SubOp.OP_NAME,0),x);
    }

    /**
     * Subtraction operation: elementwise {@code this - x}<br>
     * If this and x variables have equal shape, the output shape is the same as the inputs.<br>
     * Supports broadcasting: if this and x have different shapes and are broadcastable, the output shape is broadcast.
     *
     * @param name Name of the output variable
     * @param x    Variable to perform operation with
     * @return Output (result) SDVariable
     */
    public SDVariable sub(String name, SDVariable x) {
        val result = sameDiff.f().sub(this,x);
        return sameDiff.updateVariableNameAndReference(result,name);
    }

    /**
     * See {@link #div(String,double)}
     */
    public SDVariable div(double scalar) {
        return div(sameDiff.generateNewVarName(DivOp.OP_NAME,0),scalar);
    }

    /**
     * Scalar division: {@code out = this / scalar}<br>
     * Output variable has the same shape as the input variable
     *
     * @param varName Output variable name
     * @param scalar  Scalar for operation
     * @return Output variable
     */
    public SDVariable div(String varName, double scalar) {
        val function = sameDiff.f().div(this,scalar);
        return sameDiff.updateVariableNameAndReference(function,varName);
    }

    /**
     * See {@link #div(String, SDVariable)}
     */
    public SDVariable div(SDVariable x) {
        return div(sameDiff.generateNewVarName(DivOp.OP_NAME,0),x);
    }

    /**
     * Division operation: elementwise {@code this / x}<br>
     * If this and x variables have equal shape, the output shape is the same as the inputs.<br>
     * Supports broadcasting: if this and x have different shapes and are broadcastable, the output shape is broadcast.
     *
     * @param name Name of the output variable
     * @param x    Variable to perform operation with
     * @return Output (result) SDVariable
     */
    public SDVariable div(String name, SDVariable x) {
        val result = sameDiff.f().div(this, x);
        return sameDiff.updateVariableNameAndReference(result, name);
    }

    /**
     * See {@link #mul(String, double)}
     */
    public SDVariable mul(double scalar) {
        return mul(sameDiff.generateNewVarName(MulOp.OP_NAME,0),scalar);
    }

    /**
     * Scalar multiplication: {@code out = this * scalar}<br>
     * Output variable has the same shape as the input variable
     *
     * @param varName Output variable name
     * @param scalar  Scalar for operation
     * @return Output variable
     */
    public SDVariable mul(String varName, double scalar) {
        val function = sameDiff.f().mul(this, scalar);
        return sameDiff.updateVariableNameAndReference(function,varName);
    }


    /**
     * See {@link #mul(String, SDVariable)}
     */
    public SDVariable mul(SDVariable x) {
        return mul(sameDiff.generateNewVarName(MulOp.OP_NAME,0),x);
    }

    /**
     * Multiplication operation: elementwise {@code this * x}<br>
     * If this and x variables have equal shape, the output shape is the same as the inputs.<br>
     * Supports broadcasting: if this and x have different shapes and are broadcastable, the output shape is broadcast.
     *
     * @param name Name of the output variable
     * @param x    Variable to perform operation with
     * @return Output (result) SDVariable
     */
    public SDVariable mul(String name, SDVariable x) {
        val result = sameDiff.f().mul(this, x);
        return sameDiff.updateVariableNameAndReference(result,name);
    }

    /**
     * See {@link #pow(String, double)}
     */
    public SDVariable pow(double scalar) {
        return pow(null, scalar);
    }

    /**
     * Scalar power operation: {@code out = this ^ scalar}<br>
     * Output variable has the same shape as the input variable
     *
     * @param varName Output variable name
     * @param scalar  Scalar for operation
     * @return Output variable
     */
    public SDVariable pow(String varName, double scalar) {
        SDVariable ret = f().pow(this, scalar);
        return sameDiff.updateVariableNameAndReference(ret, varName);
    }

    /**
     * See {@link #rsub(String, double)}
     */
    public SDVariable rsub(double scalar) {
        return rsub(sameDiff.generateNewVarName(RSubOp.OP_NAME,0),scalar);
    }

    /**
     * Scalar reverse subtraction: {@code out = scalar - this}<br>
     * Output variable has the same shape as the input variable
     *
     * @param varName Output variable name
     * @param scalar  Scalar for operation
     * @return Output variable
     */
    public SDVariable rsub(String varName, double scalar) {
        val function = sameDiff.f().rsub(this,scalar);
        return sameDiff.updateVariableNameAndReference(function,varName);
    }

    /**
     * See {@link #rsub(String, SDVariable)}
     */
    public SDVariable rsub(SDVariable x) {
        return rsub(sameDiff.generateNewVarName(RSubOp.OP_NAME,0),x);
    }

    /**
     * Reverse subtraction operation: elementwise {@code x - this}<br>
     * If this and x variables have equal shape, the output shape is the same as the inputs.<br>
     * Supports broadcasting: if this and x have different shapes and are broadcastable, the output shape is broadcast.
     *
     * @param name Name of the output variable
     * @param x    Variable to perform operation with
     * @return Output (result) SDVariable
     */
    public SDVariable rsub(String name, SDVariable x) {
        val result = sameDiff.f().rsub(this,x);
        return sameDiff.updateVariableNameAndReference(result,name);
    }

    /**
     * See {@link #rdiv(String, double)}
     */
    public SDVariable rdiv(double scalar) {
        return rdiv(sameDiff.generateNewVarName(RDivOp.OP_NAME,0),scalar);
    }

    /**
     * Scalar reverse division: {@code out = scalar / this}<br>
     * Output variable has the same shape as the input variable
     *
     * @param varName Output variable name
     * @param scalar  Scalar for operation
     * @return Output variable
     */
    public SDVariable rdiv(String varName, double scalar) {
        val function = sameDiff.f().rdiv(this, scalar);
        return sameDiff.updateVariableNameAndReference(function, varName);
    }

    /**
     * See {@link #rdiv(String, SDVariable)}
     */
    public SDVariable rdiv(SDVariable sameDiffVariable) {
        return rdiv(sameDiff.generateNewVarName(RDivOp.OP_NAME,0),sameDiffVariable);
    }

    /**
     * Reverse division operation: elementwise {@code x / this}<br>
     * If this and x variables have equal shape, the output shape is the same as the inputs.<br>
     * Supports broadcasting: if this and x have different shapes and are broadcastable, the output shape is broadcast.
     *
     * @param name Name of the output variable
     * @param x    Variable to perform operation with
     * @return Output (result) SDVariable
     */
    public SDVariable rdiv(String name, SDVariable x) {
        val result = sameDiff.f().rdiv(this,x);
        return sameDiff.updateVariableNameAndReference(result,name);

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
    public SDVariable truncatedDiv(String varName, SDVariable sameDiffVariable) {
        val function = sameDiff.f().truncatedDiv(this, sameDiffVariable);
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




    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rsubi(String varName, SDVariable sameDiffVariable) {
        val result = sameDiff.f().rsubi(this,sameDiffVariable);
        return sameDiff.updateVariableNameAndReference(result,varName);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rdivi(String varName, SDVariable sameDiffVariable) {
        val result = sameDiff.f().rdivi(this,sameDiffVariable);
        return sameDiff.updateVariableNameAndReference(result,varName);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable addi(String varName, SDVariable sameDiffVariable) {
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
        SDVariable left = this;
        SDVariable right = sameDiffVariable;
        SDVariable result = sameDiff.f().muli(left,right);
        result.setVarName(varName);
        return result;
    }

    /**
     * See {@link #squaredDifference(String, SDVariable)}
     */
    public SDVariable squaredDifference(SDVariable x) {
        return squaredDifference(sameDiff.generateNewVarName(SquaredDifferenceOp.OP_NAME,0),x);
    }

    /**
     * Squared difference operation: {@code (this - x)^2}
     * @param x Other input variable
     * @return squared difference between variables
     */
    public SDVariable squaredDifference(String name, SDVariable x) {
        val result = sameDiff.f().squaredDifference(this, x);
        return sameDiff.updateVariableNameAndReference(result, name);
    }

    /**
     * See {@link #sum(String, boolean, int...)}
     */
    public SDVariable sum(int... dimensions){
        return sum(null, dimensions);
    }

    /**
     * See {@link #sum(String, boolean, int...)}
     */
    public SDVariable sum(boolean keepDims, int... dimensions){
        return sum(null, keepDims, dimensions);
    }

    /**
     * See {@link #sum(String, boolean, int...)}
     */
    public SDVariable sum(String name, int... dimensions){
        return sum(name, false, dimensions);
    }

    /**
     * Sum array reduction operation, optionally along specified dimensions.<br>
     * Note that if keepDims = true, the output variable has the same rank as the input variable,
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]
     *
     * @param name       Output variable name
     * @param keepDims   If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Output variable: reduced array of rank (input rank - num dimensions) if keepDims = false, or
     * of rank (input rank) if keepdims = true
     */
    public SDVariable sum(String name, boolean keepDims, int... dimensions){
        return sameDiff.sum(name, this, keepDims, dimensions);
    }

    /**
     * See {@link #mean(String, boolean, int...)}
     */
    public SDVariable mean(boolean keepDims, int... dimensions){
        return mean(null, keepDims, dimensions);
    }

    /**
     * See {@link #mean(String, boolean, int...)}
     */
    public SDVariable mean(String name, int... dimensions){
        return mean(name, false, dimensions);
    }

    /**
     * See {@link #mean(String, boolean, int...)}
     */
    public SDVariable mean(int... dimensions){
        return mean(null, false, dimensions);
    }


    /**
     * Mean (average) array reduction operation, optionally along specified dimensions<br>
     * Note that if keepDims = true, the output variable has the same rank as the input variable,
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]
     *
     * @param name       Output variable name
     * @param keepDims   If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Reduced array of rank (input rank - num dimensions)
     */
    public SDVariable mean(String name, boolean keepDims, int... dimensions){
        return sameDiff.mean(name, this, keepDims, dimensions);
    }

    /**
     * See {@link #std(String, boolean, boolean, int...)}
     */
    public SDVariable std(boolean biasCorrected, int... dimensions){
        return std(null, biasCorrected, dimensions);
    }

    /**
     * See {@link #std(String, boolean, boolean, int...)}
     */
    public SDVariable std(String name, boolean biasCorrected, int... dimensions){
        return sameDiff.standardDeviation(name, this, biasCorrected, dimensions);
    }

    /**
     * Stardard deviation array reduction operation, optionally along specified dimensions<br>
     * Note that if keepDims = true, the output variable has the same rank as the input variable,
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]
     *
     * @param biasCorrected If true: divide by (N-1) (i.e., sample stdev). If false: divide by N (population stdev)
     * @param keepDims      If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions    Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Output variable: reduced array of rank (input rank - num dimensions)
     */
    public SDVariable std(String name, boolean biasCorrected, boolean keepDims, int... dimensions){
        return sameDiff.standardDeviation(name, this, biasCorrected, keepDims, dimensions);
    }

    /**
     * See {@link #prod(String, boolean, int...)}
     */
    public SDVariable prod(int... dimensions){
        return prod(null, dimensions);
    }

    /**
     * See {@link #prod(String, boolean, int...)}
     */
    public SDVariable prod(boolean keepDims, int... dimensions){
        return prod(null, keepDims, dimensions);
    }

    /**
     * See {@link #prod(String, boolean, int...)}
     */
    public SDVariable prod(String name, int... dimensions){
        return sameDiff.prod(name, this, dimensions);
    }

    /**
     * Product array reduction operation, optionally along specified dimensions<br>
     * Note that if keepDims = true, the output variable has the same rank as the input variable,
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]
     *
     * @param name       Output variable name
     * @param keepDims   If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Output variable: reduced array of rank (input rank - num dimensions)
     */
    public SDVariable prod(String name, boolean keepDims, int... dimensions){
        return sameDiff.prod(name, this, keepDims, dimensions);
    }

    /**
     * See {@link #min(String, boolean, int...)}
     */
    public SDVariable min(int... dimensions){
        return min(null, dimensions);
    }

    /**
     * See {@link #min(String, boolean, int...)}
     */
    public SDVariable min(boolean keepDims, int... dimensions){
        return min(null, keepDims, dimensions);
    }

    /**
     * See {@link #min(String, boolean, int...)}
     */
    public SDVariable min(String name, int... dimensions){
        return min(name, false, dimensions);
    }

    /**
     * Minimum array reduction operation, optionally along specified dimensions. out = min(in)<br>
     * Note that if keepDims = true, the output variable has the same rank as the input variable,
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]
     *
     * @param name       Output variable name
     * @param keepDims   If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Reduced array of rank (input rank - num dimensions)
     */
    public SDVariable min(String name, boolean keepDims, int... dimensions){
        return sameDiff.min(name, this, keepDims, dimensions);
    }

    /**
     * See {@link #max(String, boolean, int...)}
     */
    public SDVariable max(int... dimensions) {
        return max(null, dimensions);
    }

    /**
     * See {@link #max(String, boolean, int...)}
     */
    public SDVariable max(String name, int... dimensions) {
        return max(name, false, dimensions);
    }

    /**
     * See {@link #max(String, boolean, int...)}
     */
    public SDVariable max(boolean keepDims, int... dimensions) {
        return max(null, keepDims, dimensions);
    }

    /**
     * Maximum array reduction operation, optionally along specified dimensions<br>
     * Note that if keepDims = true, the output variable has the same rank as the input variable,
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]
     *
     * @param name       Output variable name
     * @param keepDims   If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Reduced array of rank (input rank - num dimensions)
     */
    public SDVariable max(String name, boolean keepDims, int... dimensions) {
        return sameDiff.max(name, this, keepDims, dimensions);
    }

    /**
     * See {@link #norm1(String, boolean, int...)}
     */
    public SDVariable norm1(int... dimensions){
        return norm1(null, dimensions);
    }

    /**
     * See {@link #norm1(String, boolean, int...)}
     */
    public SDVariable norm1(boolean keepDims, int... dimensions){
        return norm1(null, keepDims, dimensions);
    }

    /**
     * See {@link #norm1(String, boolean, int...)}
     */
    public SDVariable norm1(String name, int... dimensions){
        return norm1(name, false, dimensions);
    }

    /**
     * Norm1 (L1 norm) reduction operation: The output contains the L1 norm for each tensor/subset along the specified dimensions:<br>
     * {@code out = sum_i abs(x[i])}<br>
     * Note that if keepDims = true, the output variable has the same rank as the input variable,
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]
     *
     * @param name       Output variable name
     * @param keepDims   If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions dimensions to reduce over
     * @return Output variable
     */
    public SDVariable norm1(String name, boolean keepDims, int... dimensions) {
        return sameDiff.norm1(name, this, keepDims, dimensions);
    }

    /**
     * See {@link #norm2(String, boolean, int...)}
     */
    public SDVariable norm2(int... dimensions){
        return norm2(null, dimensions);
    }

    /**
     * See {@link #norm2(String, boolean, int...)}
     */
    public SDVariable norm2(boolean keepDims, int... dimensions){
        return norm2(null, keepDims, dimensions);
    }

    /**
     * See {@link #norm2(String, boolean, int...)}
     */
    public SDVariable norm2(String name, int... dimensions){
        return norm2(name, false, dimensions);
    }

    /**
     * Norm2 (L2 norm) reduction operation: The output contains the L2 norm for each tensor/subset along the specified dimensions:<br>
     * {@code out = sqrt(sum_i x[i]^2)}<br>
     * Note that if keepDims = true, the output variable has the same rank as the input variable,
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]
     *
     * @param name       Output variable name
     * @param keepDims   If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions dimensions to reduce over
     * @return Output variable
     */
    public SDVariable norm2(String name, boolean keepDims, int... dimensions){
        return sameDiff.norm2(name, this, keepDims, dimensions);
    }

    /**
     * See {@link #normmax(String, boolean, int...)}
     */
    public SDVariable normmax(int... dimensions){
        return normmax(null, dimensions);
    }

    /**
     * See {@link #normmax(String, boolean, int...)}
     */
    public SDVariable normmax(boolean keepDims, int... dimensions){
        return normmax(null, keepDims, dimensions);
    }

    /**
     * See {@link #normmax(String, boolean, int...)}
     */
    public SDVariable normmax(String name, int... dimensions){
        return normmax(name, false, dimensions);
    }

    /**
     * Max norm (infinity norm) reduction operation: The output contains the max norm for each tensor/subset along the
     * specified dimensions:<br>
     * {@code out = max(abs(x[i]))}<br>
     * Note that if keepDims = true, the output variable has the same rank as the input variable,
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]
     *
     * @param name       Output variable name
     * @param keepDims   If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions dimensions to reduce over
     * @return Output variable
     */
    public SDVariable normmax(String name, boolean keepDims, int... dimensions){
        return sameDiff.normmax(name, this, keepDims, dimensions);
    }

    /**
     * See {@link #argmax(String, boolean, int...)}
     */
    public SDVariable argmax(int... dimensions){
        return argmax(null, dimensions);
    }

    /**
     * See {@link #argmax(String, boolean, int...)}
     */
    public SDVariable argmax(String name, int... dimensions){
        return sameDiff.argmax(name, this, dimensions);
    }

    /**
     * Argmax array reduction operation, optionally along specified dimensions.<br>
     * Output values are the index of the maximum value of each slice along the specified dimension.<br>
     * <br>
     * Note that if keepDims = true, the output variable has the same rank as the input variable,
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]
     *
     * @param name       Name of the output variable
     * @param keepDims   If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Output variable: reduced array of rank (input rank - num dimensions) if keepDims = false, or
     * of rank (input rank) if keepdims = true
     */
    public SDVariable argmax(String name, boolean keepDims, int... dimensions) {
        return sameDiff.argmax(name, this, keepDims, dimensions);
    }

    /**
     * See {@link #argmin(String, boolean, int...)}
     */
    public SDVariable argmin(int... dimensions){
        return argmin(null, dimensions);
    }

    /**
     * See {@link #argmin(String, boolean, int...)}
     */
    public SDVariable argmin(String name, int... dimensions){
        return sameDiff.argmin(name, this, dimensions);
    }

    /**
     * Argmin array reduction operation, optionally along specified dimensions.<br>
     * Output values are the index of the minimum value of each slice along the specified dimension.<br>
     * <br>
     * Note that if keepDims = true, the output variable has the same rank as the input variable,
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]
     *
     * @param name       Name of the output variable
     * @param keepDims   If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Output variable: reduced array of rank (input rank - num dimensions) if keepDims = false, or
     * of rank (input rank) if keepdims = true
     */
    public SDVariable argmin(String name, boolean keepDims, int... dimensions) {
        return sameDiff.argmax(name, this, keepDims, dimensions);
    }

    /**
     * Associate the specified array with this variable
     * @param array Array to associate with this variable
     * @return This variable
     */
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

    /**
     * Get a variable with content equal to a specified sub-array of this variable.<br>
     * Can be used (for example) to get rows, columns, sub-matrices, etc.
     * @param indices Indices to get
     * @return Sub-array variable
     */
    public SDVariable get(SDIndex... indices) {
        int ndims = indices.length;
        long[] begin = new long[ndims];
        long[] end = new long[ndims];
        long[] strides = new long[ndims];
        int[] begin_mask_arr = new int[ndims];
        int[] end_mask_arr = new int[ndims];
        int[] shrink_axis_mask_arr = new int[ndims];
        for (int i = 0; i < ndims; i++) {
            strides[i] = 1;
            SDIndex index = indices[i];
            SDIndex.IndexType indexType = index.getIndexType();
            if (indexType == SDIndex.IndexType.ALL) {
                begin_mask_arr[i] = 1;
                end_mask_arr[i] = 1;
            } else if (indexType == SDIndex.IndexType.POINT) {
                long pointIndex = index.getPointIndex();
                begin[i] = pointIndex;
                end[i] = pointIndex + 1;
                shrink_axis_mask_arr[i] = 1;
            } else if (indexType == SDIndex.IndexType.INTERVAL) {
                if (index.getIntervalBegin() == null) {
                    begin_mask_arr[i] = 1;
                } else {
                    begin[i] = index.getIntervalBegin();
                }
                if (index.getIntervalEnd() == null) {
                    end_mask_arr[i] = 1;
                } else {
                    end[i] = index.getIntervalEnd();
                }
                if (index.getIntervalStrides() == null) {
                    strides[i] = 1;
                } else {
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


    private static int binArrToInt(int[] arr) {
        int x = 0;
        int m = 1;
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == 1) {
                x += m;
            }
            m *= 2;
        }
        return x;
    }
   
}

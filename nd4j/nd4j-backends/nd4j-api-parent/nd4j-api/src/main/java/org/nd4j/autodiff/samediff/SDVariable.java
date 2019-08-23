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

import java.util.Objects;
import lombok.*;
import lombok.extern.slf4j.Slf4j;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.internal.Variable;
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
@NoArgsConstructor
@Slf4j
public class SDVariable implements Serializable {

    protected SameDiff sameDiff;

    @Getter
    @Setter
    protected String varName;
    @Getter
    @Setter
    protected VariableType variableType;

    @Getter
    @Setter
    protected WeightInitScheme weightInitScheme;
    protected long[] shape;

    @Getter (AccessLevel.NONE)
    @Setter
    protected DataType dataType;

    private DifferentialFunction creator;

    // autogen_tag::sdvars::start


    public SDVariable(@NonNull String varName, @NonNull VariableType varType, @NonNull SameDiff sameDiff, long[] shape, DataType dataType, WeightInitScheme weightInitScheme){
        Preconditions.checkState(weightInitScheme == null || varType == VariableType.VARIABLE, "Weight initalization schemes can only be applied to VARIABLE type" +
                " SDVariables - variable \"%s\" is of type %s but was provided a weight initialization scheme %s", varName, varType, weightInitScheme);
        Preconditions.checkState(dataType != DataType.UNKNOWN, "Unknown datatype is not allowed for SDVariables (variable name: %s)", varName);

        varName = sameDiff.generateNewVarName(varName, 0, true);

        this.sameDiff = sameDiff;
        this.varName = varName;
        this.variableType = varType;
        this.dataType = dataType;
        this.weightInitScheme = weightInitScheme;
        this.shape = shape;
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




    /**
     * Allocate and return a  new array
     * based on the vertex id and weight initialization.
     * @return the allocated array
     */
    public INDArray storeAndAllocateNewArray() {
        Preconditions.checkState(variableType == VariableType.VARIABLE, "Unable to allocate and store array for variable of type %s: only" +
                " VARIABLE type variables can be initialized using this method", variableType);

        if(!sameDiff.arrayAlreadyExistsForVarName(varName)){
            long[] shape = getShape();
            INDArray arr = getWeightInitScheme().create(dataType(), shape);
            sameDiff.associateArrayWithVariable(arr, this);
            if(log.isTraceEnabled()){
                log.trace("Generated and stored new array for variable \"{}\": shape {}", getVarName(), Arrays.toString(arr.shape()));
            }
            return arr;
        }

        //Variable type SDVariables: shape should never change (i.e., these are params in the net!)
        INDArray ret = getArr();
        return ret;
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
        if(variableType == VariableType.VARIABLE && weightInitScheme != null && shape != null){
            INDArray arr = weightInitScheme.create(dataType, shape);
            sameDiff.associateArrayWithVariable(arr, this);
            if(log.isTraceEnabled()){
                log.trace("getArr() for variable \"{}\" allocated new array: shape {}", getVarName(), Arrays.toString(getShape()));
            }
            return arr;
        } else if(sameDiff.getShapeForVarName(getVarName()) == null) {
            if (enforceExistence) {
                throw new IllegalStateException("Cannot get array for SDVariable \"" + getVarName() + "\": no array has" +
                        " been defined, and array shape cannot be calculated");
            }
            if(log.isTraceEnabled()){
                log.trace("SDVariable.getArr(): could not get array for variable {}: shape is null", getVarName());
            }
            return null;
        }
        return sameDiff.getArrForVarName(getVarName());
    }


    /**
     * Alias for the gradient variable - same as {@link #getGradient()}.
     * The gradient variable is the variable that represents the derivative of the loss function with respect
     * to the output of this variable. I.e., if this variable is X and loss function is L, then gradient() returns the
     * variable representing dL/dX.<br>
     * Note that only floating point variables can have gradients.
     */
    public SDVariable gradient() {
        return getGradient();
    }

    /**
     * The gradient variable is the variable that represents the derivative of the loss function with respect
     * to the output of this variable. I.e., if this variable is X and loss function is L, then gradient() returns the
     * variable representing dL/dX<br>
     * Note that only floating point variables can have gradients.<br>
     * Note also that a gradient may not yet be defined, and/or if no loss function variables have been set.<br>
     * You can set the loss function variables using {@link SameDiff#setLossVariables(String...)} and then create the
     * gradient functions using {@link SameDiff#createGradFunction()}. Alternatively, the gradient function will be
     * created automatically when training is performed.
     */
    public SDVariable getGradient() {
        Preconditions.checkState(dataType().isFPType(), "Cannot get gradient of %s variable \"%s\": only floating" +
                " point variables have gradients", getVarName(), dataType());
        return sameDiff.getGradForVariable(getVarName());
    }


    /**
     * Returns the shape of this variable
     * @return Shape of the variable
     */
    public long[] getShape() {
        if (variableType == VariableType.PLACEHOLDER && getArr() == null) {
            if (shape != null)
                return shape;
            else
                return new long[0];
        }

        long[] initialShape =  sameDiff.getShapeForVarName(getVarName());
        if(initialShape == null) {
            val arr = getArr();
            if(arr != null)
                return arr.shape();
        }

        return initialShape;
    }

    public long[] placeholderShape(){
        if(variableType != VariableType.PLACEHOLDER){
            throw new IllegalStateException("placeholderShape() can only be used for placeholder variables: variable \"" + getVarName()
                    + " is a variable of type " + variableType);
        }
        return shape;
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

    public SDVariable castTo(@NonNull DataType dataType){
        return castTo(null, dataType);
    }

    public SDVariable castTo(String name, @NonNull DataType dataType){
        return sameDiff.castTo(name, this, dataType);
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
        return sameDiff.f().neg(this);
    }

    /**
     * Negate op - returns a new variable with the values of the current variable negated
     * @param name Name of the new variable
     * @return Negated variable
     */
    public SDVariable neg(String name){
        return sameDiff.math().neg(name, this);
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
     * See {@link #dot(String, SDVariable, int...)}
     */
    public SDVariable dot(SDVariable other, int... dimensions){
        return dot(null, other, dimensions);
    }

    /**
     * Matrix dot product: out = dot(this,other, dimensions)
     *
     * @param name  Name of the output variable
     * @param other Other variable to perform matrix multiplication with
     * @return Output variable (result of mmul)
     */
    public SDVariable dot(String name, SDVariable other, int... dimensions){
        return sameDiff.dot(name, this, other, dimensions);
    }



    /**
     * See {@link #add(String, double)}
     */
    public SDVariable add(double scalar) {
        return add(null,scalar);
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
        return add(null,other);
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
     * For Kotlin operator interop
     * @see #add(String, SDVariable)
     */
    public SDVariable plus(SDVariable other){
        return add(other);
    }

    /**
     * For Kotlin operator interop
     * @see #add(String, double)
     */
    public SDVariable plus(double other){
        return add(other);
    }

    /**
     * See {@link #sub(String, double)}
     */
    public SDVariable sub(double scalar) {
        return sub(null,scalar);
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
        return sub(null,x);
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
     * For Kotlin operator interop
     * @see #sub(String, SDVariable)
     */
    public SDVariable minus(SDVariable other){
        return sub(other);
    }

    /**
     * For Kotlin operator interop
     * @see #sub(String, double)
     */
    public SDVariable minus(double other){
        return sub(other);
    }

    /**
     * See {@link #div(String,double)}
     */
    public SDVariable div(double scalar) {
        return div(null,scalar);
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
        return div(null,x);
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
     * Floor division operation: elementwise {@code this // x}<br>
     * If this and x variables have equal shape, the output shape is the same as the inputs.<br>
     * Supports broadcasting: if this and x have different shapes and are broadcastable, the output shape is broadcast.
     *
     * @param name Name of the output variable
     * @param x    Variable to perform operation with
     * @return Output (result) SDVariable
     */
    public SDVariable fdiv(String name, SDVariable x) {
        val result = sameDiff.f().floorDiv(this, x);
        return sameDiff.updateVariableNameAndReference(result, name);
    }

    /**
     * Modulo operation: elementwise {@code this / x}<br>
     * If this and x variables have equal shape, the output shape is the same as the inputs.<br>
     * Supports broadcasting: if this and x have different shapes and are broadcastable, the output shape is broadcast.
     *
     * @param name Name of the output variable
     * @param x    Variable to perform operation with
     * @return Output (result) SDVariable
     */
    public SDVariable mod(String name, SDVariable x) {
        val result = sameDiff.f().mod(this, x);
        return sameDiff.updateVariableNameAndReference(result, name);
    }

    /**
     * See {@link #mul(String, double)}
     */
    public SDVariable mul(double scalar) {
        return mul(null,scalar);
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
        return mul(null,x);
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
     * For Kotlin operator interop
     * @see #mul(String, SDVariable)
     */
    public SDVariable times(SDVariable other){
        return mul(other);
    }

    /**
     * For Kotlin operator interop
     * @see #mul(String, double)
     */
    public SDVariable times(double other){
        return mul(other);
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
        SDVariable ret = sameDiff.f().pow(this, scalar);
        return sameDiff.updateVariableNameAndReference(ret, varName);
    }

    /**
     * See {@link #rsub(String, double)}
     */
    public SDVariable rsub(double scalar) {
        return rsub(null,scalar);
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
        return rsub(null,x);
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
        return rdiv(null,scalar);
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
        return rdiv(null,sameDiffVariable);
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
    public SDVariable truncatedDiv(SDVariable sameDiffVariable) {
        return truncatedDiv(null,sameDiffVariable);

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
     * See {@link #squaredDifference(String, SDVariable)}
     */
    public SDVariable squaredDifference(SDVariable x) {
        return squaredDifference(null,x);
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
     * Get the shape of the array as a dynamic SDVariable
     * @return Shape SDVariable
     */
    public SDVariable shape(){
        return sameDiff.shape(this);
    }

    /**
     * Get the rank of this variable as a dynamic SDVariable
     * @return Rank SDVariable
     */
    public SDVariable rank(){
        return sameDiff.rank(this);
    }

    /**
     * Reshape the current variable to the specified (dynamic) shape. The output variable will have the same values as the
     * input, but with the specified shape.<br>
     * Note that prod(shape) must match length(input) == prod(input.shape)
     *
     * @param newShape New shape for variable
     * @return Output variable
     */
    public SDVariable reshape(SDVariable newShape){
        return sameDiff.reshape(this, newShape);
    }

    /**
     * Reshape the current variable to the specified shape. The output variable will have the same values as the
     * input, but with the specified shape.<br>
     * Note that prod(shape) must match length(input) == prod(input.shape)
     *
     * @param newShape New shape for variable
     * @return Output variable
     */
    public SDVariable reshape(int... newShape){
        return sameDiff.reshape(this, newShape);
    }

    /**
     * Reshape the current variable to the specified shape. The output variable will have the same values as the
     * input, but with the specified shape.<br>
     * Note that prod(shape) must match length(input) == prod(input.shape)
     *
     * @param newShape New shape for variable
     * @return Output variable
     */
    public SDVariable reshape(long... newShape){
        return sameDiff.reshape(this, newShape);
    }

    /**
     * Permute the dimensions of the current variable according to the specified permutation indices.<br>
     * Example: if the current variable has shape [a,b,c] and dimensions = [2,0,1] the output has shape [c,a,b]
     *
     * @param dimensions The new dimension order
     * @return Output variable (permuted input)
     */
    public SDVariable permute(int... dimensions){
        return sameDiff.permute(this, dimensions);
    }

    public SDVariable permute(SDVariable dimensions){
        return sameDiff.permute(null, this, dimensions);
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
        sameDiff.exec(null, getVarName());
        return getArr();
    }


    /**
     * Evaluate the result of this variable
     * @return
     */
    public INDArray eval(Map<String, INDArray> placeholders) {
        sameDiff.exec(placeholders, getVarName());
        return getArr();
    }


    @Override
    public String toString() {
        return "SDVariable(name=\"" + varName + "\",variableType=" + variableType + ",dtype=" + dataType +
                (variableType == VariableType.PLACEHOLDER && shape != null ? ",shape=" + Arrays.toString(shape): "") + ")";
    }

    /**
     * Add a control dependency for this variable on the specified variable.<br>
     * Control depnedencies can be used to enforce the execution order.
     * For example, if a control dependency X->Y exists, then Y will only be executed after X is executed - even
     * if Y wouldn't normally depend on the result/values of X.
     *
     * @param controlDependency Control dependency to add for this variable
     */
    public void addControlDependency(SDVariable controlDependency){
        String cdN = controlDependency.getVarName();
        String n = this.getVarName();
        Variable v = sameDiff.getVariables().get(n);
        if(v.getControlDeps() == null)
            v.setControlDeps(new ArrayList<String>());
        if(!v.getControlDeps().contains(cdN))
            v.getControlDeps().add(cdN);

        Variable v2 = sameDiff.getVariables().get(cdN);
        if(v2.getControlDepsForVar() == null)
            v2.setControlDepsForVar(new ArrayList<String>());
        if(!v2.getControlDepsForVar().contains(n))
            v2.getControlDepsForVar().add(n);
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
                if(!index.isPointKeepDim()) {
                    shrink_axis_mask_arr[i] = 1;
                }
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

    /**
     * Convert this variable to a constant. This is equivalent to "freezing" a variable so that it's value
     * won't be changed by further training.<br>
     * This can only be done for variables and placeholders, not ARRAY type variables (which are usually network activations).
     * As a constant, this variable will no longer be modified by any subsequent training.
     *
     * @return This variable (now a constant)
     */
    public SDVariable convertToConstant(){
        return sameDiff.convertToConstant(this);
    }

    /**
     * Convert this variable to a VARIABLE type SDVariable.<br>
     * This can only be done for constants and placeholders, not ARRAY type variables (which are usually network activations).
     * As a variable, this variable will modified during any subsequent training.
     *
     * @return This variable (now a variable type SDVariable)
     */
    public SDVariable convertToVariable(){
        return sameDiff.convertToVariable(this);
    }

    /**
     * Rename this variable to a new name. Equivalent to {@link SameDiff#renameVariable(String, String)}
     *
     * @param newName The new name for the variable - no variable with this name must already exist
     * @return The current variable (same object)
     */
    public SDVariable rename(String newName) {
        sameDiff.renameVariable(getVarName(), newName);
        return this;
    }

    /**
     * Mark this variable as a loss function variable. This means that this variable will be minimized via backprop during training.<br>
     * This will add the variable as a loss to any others - i.e., if multiple variables are marked as losses, their values will be summed
     * to give the total network loss.<br>
     * Note that only floating point (Float16/32/64) variables may be marked as a loss.<br>
     * Note also that only ARRAY type SDVariables can be marked as losses to be minimized. That is, we cannot mark the value
     * of a constant, variable or placeholder to be minimized as doing so would not make sense.<br>
     * This is equivalent to {@link SameDiff#addLossVariable(String)}
     */
    public void markAsLoss(){
        sameDiff.addLossVariable(getVarName());
    }

    /**
     * Determine if this variable has a gradient with respect to the current loss. Note that:
     * (a) Non-floating-point variables (integer, string, etc) will never have gradients<br>
     * (b) This method will return false if no gradient function has been created yet. See {@link SameDiff#createGradFunction()}
     * and {@link SameDiff#setLossVariables(String...)}<br>
     * (c) Floating point variables may not have any gradient if the current loss does not depend on the variable at all<br>
     * @return True if a gradient variable exists for the specified variable, for the current loss
     */
    public boolean hasGradient(){
        return sameDiff.variableHasGradient(getVarName());
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

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof SDVariable)) {
            return false;
        }

        SDVariable that = (SDVariable) o;

        if (!Objects.equals(varName, that.varName)) {
            return false;
        }
        if (variableType != that.variableType) {
            return false;
        }
        if(sameDiff != that.sameDiff){
            return false;
        }
        return dataType == that.dataType;
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + (varName != null ? varName.hashCode() : 0);
        result = 31 * result + (variableType != null ? variableType.hashCode() : 0);
        result = 31 * result + (dataType != null ? dataType.hashCode() : 0);
        return result;
    }

    public SDVariable clone(SameDiff sd){
        SDVariable v = new SDVariable();
        v.varName = varName;
        v.variableType = variableType;
        v.weightInitScheme = weightInitScheme;
        v.shape = shape == null ? null : shape.clone();
        v.dataType = dataType;
        v.sameDiff = sd;
        return v;
    }
}

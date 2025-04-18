/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.autodiff.samediff;

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.shape.CreateView;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.weightinit.WeightInitScheme;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;

@Data
@NoArgsConstructor
@Slf4j
public class SDVariable implements Serializable {

    protected SameDiff sameDiff;

    @Getter
    protected String varName;
    @Getter
    @Setter
    protected VariableType variableType;

    @Setter(AccessLevel.NONE)
    protected long[] shape;

    @Getter (AccessLevel.NONE)
    @Setter
    protected DataType dataType;

    private DifferentialFunction creator;



    public SDVariable(@NonNull String varName, @NonNull VariableType varType, @NonNull SameDiff sameDiff, long[] shape, DataType dataType){
        if(varType != VariableType.PLACEHOLDER)
            Preconditions.checkState(dataType != DataType.UNKNOWN, "Unknown datatype is not allowed for SDVariables (variable name: %s)", varName);
        if(varName == null)
            varName = sameDiff.generateNewVarName(varName, 0, true);

        this.sameDiff = sameDiff;
        this.varName = varName;
        this.variableType = varType;
        this.dataType = dataType;
        this.shape = shape;
    }

    /**
     * Get the name of the SDVariable
     * @return Name of the variable
     */
    public String name(){
        return varName;
    }

    public void setVarName(String varName) {
        this.varName = varName;
    }

    /**
     * @deprecated Use {@link #name()}
     */
    @Deprecated
    public String getVarName(){
        return name();
    }

    /**
     * Returns true if this variable is a placeholder
     * @return
     */
    public boolean isPlaceHolder() {
        return variableType == VariableType.PLACEHOLDER;
    }

    public boolean isConstant(){
        return variableType == VariableType.CONSTANT;
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
    public INDArray getArr(boolean enforceExistence) {
        if(sameDiff.arrayAlreadyExistsForVarName(getVarName()))
            return sameDiff.getArrForVarName(getVarName());
        if(variableType == VariableType.ARRAY && enforceExistence) {
            throw new UnsupportedOperationException("Cannot get array for ARRAY type SDVariable - use SDVariable.exec or SameDiff.output instead");
        } else if(variableType == VariableType.ARRAY) {
            if(sameDiff.isEagerMode()) {
                return sameDiff.getEagerArrForVarName(name());
            }
            return null;
        }

        INDArray ret = sameDiff.getArrForVarName(getVarName());
        if(enforceExistence && ret == null) {
            throw new IllegalStateException("No array exists for variable \"" + name() + "\"");
        }
        return ret;
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
        return sameDiff.getGradForVariable(getVarName());
    }


    /**
     * Returns the shape of this variable
     * @return Shape of the variable
     */
    public long[] getShape() {
        if (variableType == VariableType.PLACEHOLDER  || shape != null) {
            return shape;
        } else if(variableType == VariableType.VARIABLE || variableType == VariableType.CONSTANT) {
            if(getArr() != null)
                return getArr().shape();
        }

        return null;
    }

    public void setShape(long... shape) {
        this.shape = shape;
    }

    public long[] placeholderShape(){
        if(variableType != VariableType.PLACEHOLDER){
            throw new IllegalStateException("placeholderShape() can only be used for placeholder variables: variable \"" + getVarName()
                    + " is a variable of type " + variableType);
        }
        return shape;
    }

    public DataType dataType() {
        if(this.dataType == null) {
            //Try to infer datatype instead of returning null
            if(variableType != VariableType.ARRAY && getArr() != null) {
                this.dataType = getArr().dataType();
            }  else {
                this.dataType = DataType.UNKNOWN;
            }
        }

        return this.dataType;
    }


    // Add this helper method inside the SDVariable class
    private SDVariable handleRename(SDVariable resultVariable, String requestedName) {
        // No rename needed if name is null or already matches
        if (requestedName == null || requestedName.equals(resultVariable.name())) {
            return resultVariable;
        }

        // Ensure the result variable and its metadata exist in the associated SameDiff instance
        Variable resultVarMeta = this.sameDiff.getVariables().get(resultVariable.name());
        if (resultVarMeta == null) {
            // This might happen if the variable wasn't properly added to the map initially
            log.warn("Internal metadata for result variable '{}' not found during potential rename to '{}'. " +
                            "Returning variable with default name '{}'. Graph state might be inconsistent.",
                    resultVariable.name(), requestedName, resultVariable.name());
            return resultVariable;
        }

        // Find the operation that produced this result variable
        String producingOpName = resultVarMeta.getOutputOfOp();
        if (producingOpName == null) {
            // Variables like placeholders or constants might not have a producing op
            log.warn("Result variable '{}' does not have a producing op recorded. Cannot perform graph-aware rename to '{}'. " +
                            "Returning variable with default name '{}'.",
                    resultVariable.name(), requestedName, resultVariable.name());
            // Attempting a basic rename without full context might be possible but risky.
            // For safety, we return without renaming if the producing op isn't known.
            return resultVariable;
        }

        // Get the SameDiffOp instance for the producer
        SameDiffOp producingOp = this.sameDiff.getOps().get(producingOpName);
        if (producingOp == null) {
            // The op that supposedly produced the variable isn't in the ops map - inconsistency!
            log.warn("Could not find the producing op instance '{}' (referenced by variable '{}') during potential rename to '{}'. " +
                            "Returning variable with default name '{}'. Graph state might be inconsistent.",
                    producingOpName, resultVariable.name(), requestedName, resultVariable.name());
            return resultVariable;
        }

        // Perform the rename using the specific overload that takes the producing op
        // The 'false' argument means 'exactName=false', allowing generation of unique names if 'requestedName' clashes
        log.debug("Renaming variable '{}' produced by op '{}' to requested name '{}'", resultVariable.name(), producingOpName, requestedName);
        return this.sameDiff.updateVariableNameAndReference(producingOp, resultVariable, requestedName, false);
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
    public SDVariable assign(Number value) {
        return sameDiff.scalarSet(this, value.doubleValue());
    }

    /**
     * Negate op - returns a new variable with the values of the current variable negated
     * @return Negated variable
     */
    public SDVariable neg(){
        return sameDiff.math.neg(this);
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
        return sameDiff.mmul(name, this, other, mMulTranspose.isTransposeA(), mMulTranspose.isTransposeB(), mMulTranspose.isTransposeResult());
    }


    /**
     * See {@link #dot(String, SDVariable, long...)}
     */
    public SDVariable dot(SDVariable other, long... dimensions){
        return dot(null, other, dimensions);
    }

    /**
     * Matrix dot product: out = dot(this,other, dimensions)
     *
     * @param name  Name of the output variable
     * @param other Other variable to perform matrix multiplication with
     * @return Output variable (result of mmul)
     */
    public SDVariable dot(String name, SDVariable other, long... dimensions){
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
        val function = sameDiff.math.add(this,scalar);
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
        val result = sameDiff.math.add(this, x);
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
        val result = sameDiff.math.sub(this, scalar);
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
        val result = sameDiff.math.sub(this,x);
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
        val function = sameDiff.math.div(this,scalar);
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
        val result = sameDiff.math.div(this, x);
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
        val result = sameDiff.math.floorDiv(this, x);
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
        val result = sameDiff.math.mod(this, x);
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
        val function = sameDiff.math.mul(this, scalar);
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
        val result = sameDiff.math.mul(this, x);
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
        SDVariable ret = sameDiff.math.pow(this, scalar);
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
        val function = sameDiff.math.rsub(this,scalar);
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
        val result = sameDiff.math.rsub(this,x);
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
        val function = sameDiff.math.rdiv(this, scalar);
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
        val result = sameDiff.math.rdiv(this,x);
        return sameDiff.updateVariableNameAndReference(result,name);

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
        val result = sameDiff.math().squaredDifference(this, x);
        return sameDiff.updateVariableNameAndReference(result, name);
    }

    /**
     * See {@link #sum(String, boolean, long...)}
     */
    public SDVariable sum(long... dimensions){
        return sum(null, dimensions);
    }

    /**
     * See {@link #sum(String, boolean, long...)}
     */
    public SDVariable sum(boolean keepDims, long... dimensions){
        return sum(null, keepDims, dimensions);
    }

    /**
     * See {@link #sum(String, boolean, long...)}
     */
    public SDVariable sum(String name, long... dimensions){
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
    public SDVariable sum(String name, boolean keepDims, long... dimensions){
        return sameDiff.sum(name, this, keepDims, dimensions);
    }

    /**
     * See {@link #mean(String, boolean, long...)}
     */
    public SDVariable mean(boolean keepDims, long... dimensions){
        return mean(null, keepDims, dimensions);
    }

    /**
     * See {@link #mean(String, boolean, long...)}
     */
    public SDVariable mean(String name, long... dimensions){
        return mean(name, false, dimensions);
    }

    /**
     * See {@link #mean(String, boolean, long...)}
     */
    public SDVariable mean(long... dimensions){
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
    public SDVariable mean(String name, boolean keepDims, long... dimensions){
        return sameDiff.mean(name, this, keepDims, dimensions);
    }

    /**
     * See {@link #std(String, boolean, boolean, long...)}
     */
    public SDVariable std(boolean biasCorrected, long... dimensions){
        return std(null, biasCorrected, dimensions);
    }

    /**
     * See {@link #std(String, boolean, boolean, long...)}
     */
    public SDVariable std(String name, boolean biasCorrected, long... dimensions){
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
    public SDVariable std(String name, boolean biasCorrected, boolean keepDims, long... dimensions){
        return sameDiff.standardDeviation(name, this, biasCorrected, keepDims, dimensions);
    }

    /**
     * See {@link #prod(String, boolean, long...)}
     */
    public SDVariable prod(long... dimensions){
        return prod(null, dimensions);
    }

    /**
     * See {@link #prod(String, boolean, long...)}
     */
    public SDVariable prod(boolean keepDims, long... dimensions){
        return prod(null, keepDims, dimensions);
    }

    /**
     * See {@link #prod(String, boolean, long...)}
     */
    public SDVariable prod(String name, long... dimensions){
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
    public SDVariable prod(String name, boolean keepDims, long... dimensions){
        return sameDiff.prod(name, this, keepDims, dimensions);
    }

    /**
     * See {@link #min(String, boolean, long...)}
     */
    public SDVariable min(long... dimensions){
        return min(null, dimensions);
    }

    /**
     * See {@link #min(String, boolean, long...)}
     */
    public SDVariable min(boolean keepDims, long... dimensions){
        return min(null, keepDims, dimensions);
    }

    /**
     * See {@link #min(String, boolean, long...)}
     */
    public SDVariable min(String name, long... dimensions){
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
    public SDVariable min(String name, boolean keepDims, long... dimensions){
        return sameDiff.min(name, this, keepDims, dimensions);
    }

    /**
     * See {@link #max(String, boolean, long...)}
     */
    public SDVariable max(long... dimensions) {
        return max(null, dimensions);
    }

    /**
     * See {@link #max(String, boolean, long...)}
     */
    public SDVariable max(String name, long... dimensions) {
        return max(name, false, dimensions);
    }

    /**
     * See {@link #max(String, boolean, long...)}
     */
    public SDVariable max(boolean keepDims, long... dimensions) {
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
    public SDVariable max(String name, boolean keepDims, long... dimensions) {
        return sameDiff.max(name, this, keepDims, dimensions);
    }

    /**
     * See {@link #norm1(String, boolean, long...)}
     */
    public SDVariable norm1(long... dimensions){
        return norm1(null, dimensions);
    }

    /**
     * See {@link #norm1(String, boolean, long...)}
     */
    public SDVariable norm1(boolean keepDims, long... dimensions){
        return norm1(null, keepDims, dimensions);
    }

    /**
     * See {@link #norm1(String, boolean, long...)}
     */
    public SDVariable norm1(String name, long... dimensions){
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
    public SDVariable norm1(String name, boolean keepDims, long... dimensions) {
        return sameDiff.norm1(name, this, keepDims, dimensions);
    }

    /**
     * See {@link #norm2(String, boolean, long...)}
     */
    public SDVariable norm2(long... dimensions){
        return norm2(null, dimensions);
    }

    /**
     * See {@link #norm2(String, boolean, long...)}
     */
    public SDVariable norm2(boolean keepDims, long... dimensions){
        return norm2(null, keepDims, dimensions);
    }

    /**
     * See {@link #norm2(String, boolean, long...)}
     */
    public SDVariable norm2(String name, long... dimensions){
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
    public SDVariable norm2(String name, boolean keepDims, long... dimensions){
        return sameDiff.norm2(name, this, keepDims, dimensions);
    }

    /**
     * See {@link #normmax(String, boolean, long...)}
     */
    public SDVariable normmax(long... dimensions){
        return normmax(null, dimensions);
    }

    /**
     * See {@link #normmax(String, boolean, long...)}
     */
    public SDVariable normmax(boolean keepDims, long... dimensions){
        return normmax(null, keepDims, dimensions);
    }

    /**
     * See {@link #normmax(String, boolean, long...)}
     */
    public SDVariable normmax(String name, long... dimensions){
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
    public SDVariable normmax(String name, boolean keepDims, long... dimensions){
        return sameDiff.normmax(name, this, keepDims, dimensions);
    }

    /**
     * See {@link #argmax(String, boolean, long...)}
     */
    public SDVariable argmax(long... dimensions){
        return argmax(null, dimensions);
    }

    /**
     * See {@link #argmax(String, boolean, long...)}
     */
    public SDVariable argmax(String name, long... dimensions){
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
    public SDVariable argmax(String name, boolean keepDims, long... dimensions) {
        return sameDiff.argmax(name, this, keepDims, dimensions);
    }

    /**
     * See {@link #argmin(String, boolean, long...)}
     */
    public SDVariable argmin(long... dimensions){
        return argmin(null, dimensions);
    }

    /**
     * See {@link #argmin(String, boolean, long...)}
     */
    public SDVariable argmin(String name, long... dimensions){
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
    public SDVariable argmin(String name, boolean keepDims, long... dimensions) {
        return sameDiff.argmax(name, this, keepDims, dimensions);
    }


    /**
     * Return the total number of elements in this array
     * @return
     */
    public SDVariable length() {
        return sameDiff.prod(shape());
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
    public SDVariable reshape(SDVariable newShape) {
        return sameDiff.reshape(this, newShape);
    }

    /**
     * Reshape the current variable to the specified (dynamic) shape. The output variable will have the same values as the
     * input, but with the specified shape.<br>
     * Note that prod(shape) must match length(input) == prod(input.shape)
     *
     * @param newShape New shape for variable
     * @return Output variable
     */
    public SDVariable reshape(String name,SDVariable newShape) {
        return sameDiff.reshape(name,this, newShape);
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
        return sameDiff.reshape(this, ArrayUtil.toLongArray(newShape));
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
    public SDVariable permute(long... dimensions){
        return sameDiff.permute(this, dimensions);
    }

    public SDVariable permute(SDVariable dimensions){
        return sameDiff.permute( this, dimensions);
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
        Map<String,INDArray> m = sameDiff.output((Map<String,INDArray>)null, name());
        return m.get(name());
    }


    /**
     * Evaluate the result of this variable
     * @return
     */
    public INDArray eval(Map<String, INDArray> placeholders) {
        Map<String,INDArray> m = sameDiff.output(placeholders, name());
        return m.get(name());
    }


    @Override
    public String toString() {
        return "SDVariable(name=\"" + varName + "\",variableType=" + variableType + ",dtype=" + dataType +
                (variableType == VariableType.PLACEHOLDER && shape != null ? ",shape=" + Arrays.toString(shape): "") + ")";
    }

    /**
     * Add a control dependency for this variable on the specified variable.<br>
     * Control dependencies can be used to enforce the execution order.
     * For example, if a control dependency X->Y exists, then Y will only be executed after X is executed - even
     * if Y wouldn't normally depend on the result/values of X.
     *
     * @param controlDependency Control dependency to add for this variable
     */
    public void addControlDependency(SDVariable controlDependency){
        Variable vThis = sameDiff.getVariables().get(getVarName());
        Variable vCD = sameDiff.getVariables().get(controlDependency.name());

        //If possible: add control dependency on ops
        if(vThis.getOutputOfOp() != null && vCD.getOutputOfOp() != null ){
            //Op -> Op case
            SameDiffOp oThis = sameDiff.getOps().get(vThis.getOutputOfOp());
            SameDiffOp oCD = sameDiff.getOps().get(vCD.getOutputOfOp());

            if(oThis.getControlDeps() == null)
                oThis.setControlDeps(new ArrayList<>());
            if(!oThis.getControlDeps().contains(oCD.getName()))
                oThis.getControlDeps().add(oCD.getName());

            if(oCD.getControlDepFor() == null)
                oCD.setControlDepFor(new ArrayList<>());
            if(!oCD.getControlDepFor().contains(oThis.getName()))
                oCD.getControlDepFor().add(oThis.getName());
        } else {
            if(vThis.getOutputOfOp() != null){
                //const/ph -> op case
                SameDiffOp oThis = sameDiff.getOps().get(vThis.getOutputOfOp());

                if(oThis.getVarControlDeps() == null)
                    oThis.setVarControlDeps(new ArrayList<>());

                if(!oThis.getVarControlDeps().contains(vCD.getName()))
                    oThis.getVarControlDeps().add(vCD.getName());

                if(vCD.getControlDepsForOp() == null)
                    vCD.setControlDepsForOp(new ArrayList<>());
                if(!vCD.getControlDepsForOp().contains(oThis.getName()))
                    vCD.getControlDepsForOp().add(oThis.getName());
            } else {
                //const/ph -> const/ph case
                if(vThis.getControlDeps() == null)
                    vThis.setControlDeps(new ArrayList<>());
                if(!vThis.getControlDeps().contains(vCD.getName()))
                    vThis.getControlDeps().add(vCD.getName());

                if(vCD.getControlDepsForVar() == null)
                    vCD.setControlDepsForVar(new ArrayList<>());
                if(!vCD.getControlDepsForVar().contains(vThis.getName()))
                    vCD.getControlDepsForVar().add(vThis.getName());
            }
        }
    }



    /**
     * Get a variable with content equal to a specified sub-array of this variable.<br>
     * Can be used (for example) to get rows, columns, sub-matrices, etc.
     * @param indices Indices to get
     * @return Sub-array variable
     */
    public SDVariable getView(SDIndex... indices) {
        //copy because we can mutate this internally
        SDVariable[] indicesVars = new SDVariable[indices.length];
        for(int i = 0; i < indices.length; i++) {
            //convert indices to SDVariable based indices
            switch(indices[i].getIndexType()) {
                case INTERVAL:
                    indicesVars[i] = CreateView.createInterval(sameDiff,indices[i].getIntervalBegin(),indices[i].getIntervalEnd(),indices[i].getIntervalStrides(),indices[i].isInclusive()  ? 1 : 0);
                    break;
                case POINT:
                    indicesVars[i] = CreateView.createPoint(sameDiff,indices[i].getPointIndex());
                    break;
                case POINT_INPUT:
                    indicesVars[i] = CreateView.createPoint(sameDiff,indices[i].getPointVar());
                    break;
                case INTERVAL_INPUT:
                    indicesVars[i] = CreateView.createInterval(sameDiff,indices[i].getIntervalInputBegin(),indices[i].getIntervalInputEnd(),indices[i].getIntervalStrideInput(), indices[i].getInclusiveInput());
                    break;
                case ALL:
                    indicesVars[i] = CreateView.createAll(sameDiff);
                    break;

                default:
                    throw new IllegalArgumentException("Illegal type " + indices[i].getIndexType());

            }

        }

        return sameDiff.createView(this,indicesVars);

    }


    /**
     * Get a variable with content equal to a specified sub-array of this variable.<br>
     * Can be used (for example) to get rows, columns, sub-matrices, etc.
     * @param indices Indices to get
     * @return Sub-array variable
     */
    public SDVariable get(SDIndex... indices) {
        int ndims = indices.length;
        boolean variableIndices = false;
        //copy because we can mutate this internally
        SDIndex[] inputIndices = Arrays.copyOf(indices,indices.length);
        indices = inputIndices;
        for(int i = 0; i < indices.length; i++) {
            if(indices[i].getIndexType() == SDIndex.IndexType.POINT_INPUT || indices[i].getIndexType() == SDIndex.IndexType.INTERVAL_INPUT) {
                variableIndices = true;
            }

            //convert indices to SDVariable based indices
            if(variableIndices && (indices[i].getIndexType() == SDIndex.IndexType.INTERVAL || indices[i].getIndexType() == SDIndex.IndexType.POINT)) {
                switch(indices[i].getIndexType()) {
                    case INTERVAL:
                        indices[i] = SDIndex.interval(sameDiff.constant(indices[i].getIntervalBegin()),sameDiff.constant(indices[i].getIntervalEnd()),sameDiff.constant(indices[i].getIntervalEnd()));
                        break;
                    case POINT:
                        indices[i] = SDIndex.point(sameDiff.constant(indices[i].getPointIndex()),indices[i].isPointKeepDim());
                        break;
                }
            }

        }

        long[] begin = new long[ndims];
        long[] end = new long[ndims];
        long[] strides = new long[ndims];
        int[] begin_mask_arr = new int[ndims];
        int[] end_mask_arr = new int[ndims];
        int[] shrink_axis_mask_arr = new int[ndims];

        SDVariable beginVar = null;
        SDVariable endVar = null;
        SDVariable stridesVar = null;

        for (int i = 0; i < ndims; i++) {
            strides[i] = 1;
            SDIndex index = indices[i];
            SDIndex.IndexType indexType = index.getIndexType();
            if (indexType == SDIndex.IndexType.ALL) {
                begin_mask_arr[i] = 1;
                end_mask_arr[i] = 1;
            } else if (indexType == SDIndex.IndexType.POINT || indexType == SDIndex.IndexType.POINT_INPUT) {
                if(indexType == SDIndex.IndexType.POINT) {
                    long pointIndex = index.getPointIndex();
                    begin[i] = pointIndex;
                    end[i] = pointIndex + 1;
                } else if(indexType == SDIndex.IndexType.POINT_INPUT) {
                    if(beginVar == null && endVar == null) {
                        beginVar = index.getPointVar();
                        endVar = index.getPointVar().add(1.0);
                    }  else {
                        beginVar = sameDiff.concat(0,beginVar,index.getPointVar());
                        endVar = sameDiff.concat(0,endVar,index.getPointVar().add(1.0));
                    }
                }

                if(!index.isPointKeepDim()) {
                    shrink_axis_mask_arr[i] = 1;
                }
            } else if (indexType == SDIndex.IndexType.INTERVAL || indexType == SDIndex.IndexType.INTERVAL_INPUT) {
                if (index.getIntervalBegin() == null && indexType != SDIndex.IndexType.INTERVAL_INPUT) {
                    begin_mask_arr[i] = 1;
                } else if(indexType == SDIndex.IndexType.INTERVAL_INPUT) {
                    if(beginVar == null) {
                        beginVar = index.getIntervalInputBegin();
                    } else {
                        beginVar = sameDiff.concat(0,beginVar,index.getIntervalInputBegin());
                    }
                } else {
                    begin[i] = index.getIntervalBegin();
                }
                if (index.getIntervalEnd() == null && indexType != SDIndex.IndexType.INTERVAL_INPUT) {
                    end_mask_arr[i] = 1;
                } else if(indexType == SDIndex.IndexType.INTERVAL_INPUT) {
                    if(endVar == null) {
                        endVar = index.getIntervalInputEnd();
                    } else {
                        endVar = sameDiff.concat(0,endVar,index.getIntervalInputEnd());
                    }
                } else {
                    end[i] = index.getIntervalEnd();
                }
                if (index.getIntervalStrides() == null) {
                    strides[i] = 1;
                    if(stridesVar != null) {
                        stridesVar = sameDiff.concat(0,stridesVar,sameDiff.constant(1).reshape(1));
                    } else {
                        stridesVar = sameDiff.constant(1).reshape(1);
                    }
                } else {
                    strides[i] = index.getIntervalStrides();
                    if(stridesVar != null) {
                        stridesVar = sameDiff.concat(0,stridesVar,index.getIntervalStrideInput());
                    } else {
                        stridesVar = index.getIntervalStrideInput();
                    }
                }
            }
        }

        // convert binary int[] to int
        int begin_mask = binArrToInt(begin_mask_arr);
        int end_mask = binArrToInt(end_mask_arr);
        int shrink_axis = binArrToInt(shrink_axis_mask_arr);
        if(variableIndices) {
            if(stridesVar == null) {
                stridesVar = sameDiff.onesLike(beginVar);
            }

            return this.sameDiff.stridedSlice(this, beginVar, endVar, stridesVar,
                    begin_mask, end_mask, 0, 0, shrink_axis);
        } else  {
            return this.sameDiff.stridedSlice(this, begin, end, strides,
                    begin_mask, end_mask, 0, 0, shrink_axis);
        }
    }



    public static  SDVariable sliceEnd(SDVariable input,SDVariable sliceIndexInput) {
        SameDiff sameDiff = input.getSameDiff();
        SDVariable range = sameDiff.range(sameDiff.constant(0), input.rank(), sameDiff.constant(1), DataType.INT64);
        //0 1 1
        SDVariable mask = range.gt(0.0).castTo(DataType.INT64);

        SDVariable sliceMask = range.eq(0).castTo(DataType.INT64);


        SDVariable sliceIndex = sliceMask.mul(sliceIndexInput);

        SDVariable outputShape = input.shape().mul(mask).add(sliceIndex);
        return outputShape;
    }


    /**
     * Get a variable with content equal to a specified sub-array of this variable.<br>
     * Can be used (for example) to get rows, columns, sub-matrices, etc.
     *
     * This will loop over the indices (think of it as a list) and concatenate
     * each slice of the input array to the final result.
     *
     * Expected input for indices would be a vector with indices such as 0,1,2,3,4.
     * For each element in the index we then concatenate the result to the previous iteration.
     *
     * Note that this is slow and should only be used in very specific circumstances.
     * Otherwise {@link org.nd4j.linalg.api.ops.impl.shape.StridedSlice} will be more performant
     * for creating views. Many times {@link org.nd4j.linalg.api.ops.impl.shape.StridedSlice} avoids
     * this slower approach by directly calculating the strides of a view.
     *
     * @param indices Indices to get
     * @return Sub-array variable
     */
    public SDVariable get(SDVariable indices) {
        SDVariable initialSize = sameDiff.zerosLike(shape()).castTo(DataType.INT64);
        //pull from the first slice as the starting point and concatenate each result together
        SDVariable startResult = sameDiff.slice(this, initialSize.castTo(DataType.INT64), sliceEnd(this,
                sameDiff.onesLike(shape()).castTo(DataType.INT64)));
        //start at 1 because we start with the initial output (basically the item at the first element in the indices)
        SDVariable currIteration = sameDiff.var(Nd4j.ones(1).castTo(DataType.INT32));
        //this condition is normally used when you want to toss in an extra condition to terminate early
        SDVariable cond = sameDiff.constant("curr_cond",true);
        //the total length of the indices to loop till
        SDVariable indicesLength = indices.length();
        //sub graph that uses invoke
        SameDiff loop = createLoopConcat(this,indices);
        //collect slices along the first dimension concatenating the result along the way
        return this.sameDiff.loopWithConditions(ControlFlow.LoopParams.builder()
                .functionBody(loop)
                .loopVars(new SDVariable[] {
                        currIteration,
                        indicesLength,
                        cond,
                        startResult,
                        this,
                        indices
                }).functionBodyInputs(new String[] {
                        //note here all inputs are the same as the outputs, and we return the original
                        //input concatenated with the starting input (the first slice at index 0)
                        //and then loop over each index in the list till we get the specific result
                        "index",
                        "max",
                        "cond",
                        "input",
                        "pullFrom",
                        "indices"
                })
                .functionBodyOutputs(new String[]{
                        "index",
                        "max",
                        "cond",
                        "output",
                        "pullFrom",
                        "indices"})
                .functionName("slices")
                .loopName("outputs")
                //note the ordering here is important. Output is the accumulated output of each iteration appending
                //a result to the previous iteration. We start with the initial input and add more overtime.
                .build())[3];

    }



    /**
     * Get a variable with content equal to a specified sub-array of this variable.<br>
     * Can be used (for example) to get rows, columns, sub-matrices, etc.
     *
     * This will loop over the indices (think of it as a list) and add each slice
     * specified by the indices from the source to the new array.
     *
     * The end result will be this variable but with the new updated results.
     *
     * Note that this is slow and should only be used in very specific circumstances.
     * Otherwise {@link org.nd4j.linalg.api.ops.impl.shape.StridedSlice} will be more performant
     * for creating views. Many times {@link org.nd4j.linalg.api.ops.impl.shape.StridedSlice} avoids
     * this slower approach by directly calculating the strides of a view.
     *
     * @param indices Indices to get
     * @param toPut  the source array to pull results from to put in to this array
     * @param putIndices  the equivalent indices for the other array
     * @return the updated array with the elements from the toPut array put in to this new array
     */
    public SDVariable put(SDVariable indices,SDVariable toPut,SDVariable putIndices) {
        //start at 1 because we start with the initial output (basically the item at the first element in the indices)
        SDVariable currIteration = sameDiff.var(Nd4j.zeros(1).castTo(DataType.INT32));
        //this condition is normally used when you want to toss in an extra condition to terminate early
        SDVariable cond = sameDiff.constant(true);
        //the total length of the indices to loop till
        SDVariable indicesLength = indices.length();
        //sub graph that uses invoke
        SameDiff loop = createLoopPut(this,indices);
        loop.setEnableCache(false);
        //collect slices along the first dimension concatenating the result along the way
        return this.sameDiff.loopWithConditions(ControlFlow.LoopParams.builder()
                .functionBody(loop)
                .loopVars(new SDVariable[] {
                        currIteration,
                        indicesLength,
                        cond,
                        this,
                        toPut,
                        indices,
                        putIndices
                }).functionBodyInputs(new String[] {
                        //note here all inputs are the same as the outputs, and we return the original
                        //the default  3 values (current iteration, max index to loop to and optional condition)
                        //index,max,cond,assignTo,putIn,indices,indicesPut
                        "index",
                        "max",
                        "cond",
                        "assignTo",
                        "toPut",
                        "indices",
                        "indicesPut"
                })
                .functionBodyOutputs(new String[]{
                        "index",
                        "max",
                        "cond",
                        "assignOutput",
                        "toPut",
                        "indices",
                        "indicesPut"})
                .functionName("sliceputs")
                .loopName("outputs")
                //note the ordering here is important. Output is the original array where we assigned values.
                .build())[3];



    }



    /**
     * Create a graph that takes in the indices as a placeholder, loops over each element in the index vector
     * and appends the slice to the end result. This graph is equivalent to something like:
     * INDArray input = ....;
     * INDArray indices = ...;
     * INDArray result = input.get(NDArrayIndex.point(indices.getInt(0));
     * for(int i = i; i < maxIndex && customInputResult; i++) {
     * result = Nd4j.concat(0,input.get(NDArrayIndex.point(i)));
     * }
     * return result
     * <p>
     * Note this is similar to {@link INDArray#get(INDArray)}
     *
     * @param relative the expected target input variable. We use this to pull expected
     *                 return data type for the result
     * @param indices  the indices to get
     * @return the graph for dynamically creating a result graph
     */
    public static SameDiff createLoopPut(SDVariable relative,SDVariable indices) {
        //standard loop body for loopWithConditions
        SameDiff loop = SameDiff.create();
        //curr index
        SDVariable index = loop.placeHolder("index",DataType.INT32);
        //loop until
        SDVariable maxIndex = loop.placeHolder("max",DataType.INT32);
        //constant condition of true for custom,  just loop till max iterations hit
        SDVariable currCondition = loop.placeHolder("cond",DataType.BOOL);
        //the actual variable to pull from
        SDVariable assignTo = loop.placeHolder("assignTo",relative.dataType());

        SDVariable toPut = loop.placeHolder("toPut",relative.dataType());

        //the indices to loop over (the input variable
        SDVariable indicesLoop = loop.placeHolder("indices",indices.dataType());
        //standardize indices to length 1
        indicesLoop = indicesLoop.reshape("indicesReshape",indicesLoop.length());

        SDVariable indicesPut = loop.placeHolder("indicesPut",indices.dataType());
        indicesPut =  indicesPut.reshape("indicesPutReshape",indicesPut.length());

        //the current index to retrieve
        SDVariable indexToRetrieve = indicesLoop.getView(SDIndex.point(index)).reshape(1).castTo("indexToReceive",DataType.INT64);
        SDVariable indexToPut = indicesPut.getView(SDIndex.point(index)).reshape(1).castTo("indexToPut",DataType.INT64);
        SDVariable toAssign = toPut.getView(SDIndex.point(indexToPut));

        SDVariable sliceOutput = assignTo.getView(SDIndex.point(indexToRetrieve));
        SDVariable assignOutput = loop.assign(sliceOutput,toAssign);
        SDVariable outputIdentity = loop.identity("assignOutput",assignTo);
        //ensure the output depends on the final assign so it gets executed, return the final output as a view

        outputIdentity.addControlDependency(assignOutput);
        return loop;

    }


    /**
     * Create a graph that takes in the indices as a placeholder, loops over each element in the index vector
     * and appends the slice to the end result. This graph is equivalent to something like:
     * INDArray input = ....;
     * INDArray indices = ...;
     * INDArray result = input.get(NDArrayIndex.point(indices.getInt(0));
     * for(int i = i; i < maxIndex && customInputResult; i++) {
     *      result = Nd4j.concat(0,input.get(NDArrayIndex.point(i)));
     * }
     * return result
     *
     * Note this is similar to {@link INDArray#get(INDArray)}
     * @param relative the expected target input variable. We use this to pull expected
     *                 return data type for the result
     * @param indices the indices to get
     * @return the graph for dynamically creating a result graph
     */
    public static SameDiff createLoopConcat(SDVariable relative,SDVariable indices) {
        //standard loop body for loopWithConditions
        SameDiff loop = SameDiff.create();
        //curr index
        SDVariable index = loop.placeHolder("index",DataType.INT32);
        //loop until
        SDVariable maxIndex = loop.placeHolder("max",DataType.INT32);
        //constant condition of true for custom,  just loop till max iterations hit
        SDVariable currCondition = loop.placeHolder("cond",DataType.BOOL);
        //the input to pull from (in this case this)
        SDVariable input = loop.placeHolder("input", relative.dataType());
        //the actual variable to pull from
        SDVariable pullFrom = loop.placeHolder("pullFrom",relative.dataType());
        //the indices to loop over (the input variable
        SDVariable indicesLoop = loop.placeHolder("indices",indices.dataType());
        //standardize indices to length 1
        indicesLoop = indicesLoop.reshape("indicesReshape",indicesLoop.length());
        //the current index to retrieve
        SDVariable indexToRetrieve = indicesLoop.get(SDIndex.point(index)).reshape(1).castTo("indexToReceive",DataType.INT64);

        //the final concatenated output
        SDVariable sliceOutput = loop.expandDims("outputSlice",pullFrom.get(SDIndex.point(indexToRetrieve)),0);
        SDVariable output = loop.concat("output",0,input,sliceOutput);
        return loop;

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
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + (varName != null ? varName.hashCode() : 0);
        result = 31 * result + (variableType != null ? variableType.hashCode() : 0);
        result = 31 * result + (dataType != null ? dataType.hashCode() : 0);
        return result;
    }


    public SDVariable clone(String name,SameDiff sd) {
        SDVariable v = new SDVariable();
        v.varName = name;
        v.variableType = variableType;
        v.shape = shape == null ? null : shape.clone();
        v.dataType = dataType;
        v.sameDiff = sd;
        return v;
    }

    public SDVariable clone(SameDiff sd) {
        SDVariable v = new SDVariable();
        v.varName = varName;
        v.variableType = variableType;
        v.shape = shape == null ? null : shape.clone();
        v.dataType = dataType;
        v.sameDiff = sd;
        return v;
    }

    @Override
    public boolean equals(Object o){
        if(o == this) return true;
        if(!(o instanceof SDVariable))
            return false;

        SDVariable s = (SDVariable)o;
        if(!varName.equals(s.varName))
            return false;
        if(variableType != s.variableType)
            return false;
        if(dataType != s.dataType)
            return false;

        if(variableType == VariableType.VARIABLE || variableType == VariableType.CONSTANT){
            INDArray a1 = getArr();
            INDArray a2 = s.getArr();
            return a1.equals(a2);
        }
        return true;
    }


}

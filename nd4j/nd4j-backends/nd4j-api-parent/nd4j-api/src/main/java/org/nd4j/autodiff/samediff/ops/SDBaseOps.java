/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.nd4j.autodiff.samediff.ops;

import lombok.NonNull;
import org.nd4j.autodiff.functions.DifferentialFunctionFactory;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.impl.shape.OneHot;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.GradientBackwardsMarker;
import org.nd4j.linalg.indexing.conditions.Condition;

import static org.nd4j.autodiff.samediff.ops.SDValidation.*;

/**
 * Core op creator methods available via SameDiff class directly
 *
 * @author Alex Black
 * @see SDMath SDMath for Math operations
 * @see SDRandom SDRandom for random number generator operations
 * @see SDNN SDNN for general neural network operations
 * @see SDCNN SDCNN for Convolutional Neural Network operations
 * @see SDRNN SDRNN for Recurrent Neural Network operations
 * @see SDLoss SDLoss for loss function operations
 */
public abstract class SDBaseOps {

    /**
     * Intended for internal/developer use
     */
    protected SDVariable gradientBackwardsMarker(SDVariable x) {
        return gradientBackwardsMarker(generateNewVarName(new GradientBackwardsMarker().opName(), 0), x);
    }

    /**
     * Intended for internal/developer use
     */
    protected SDVariable gradientBackwardsMarker(String name, SDVariable x) {
        SDVariable result = f().gradientBackwardsMarker(x);
        return updateVariableNameAndReference(result, name);
    }

    protected abstract String generateNewVarName(String baseName, int argIndex);

    protected abstract DifferentialFunctionFactory f();

    protected abstract SDVariable updateVariableNameAndReference(SDVariable varToUpdate, String newVarName);

    protected abstract SameDiff sd();

    /**
     * Argmax array reduction operation, optionally along specified dimensions.<br>
     * Output values are the index of the maximum value of each slice along the specified dimension
     *
     * @param in         Input variable
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Reduced array of rank (input rank - num dimensions)
     */
    public SDVariable argmax(SDVariable in, int... dimensions) {
        return argmax(null, in, false, dimensions);
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
     * @param in         Input variable
     * @param keepDims   If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Output variable: reduced array of rank (input rank - num dimensions) if keepDims = false, or
     * of rank (input rank) if keepdims = true
     */
    public SDVariable argmax(String name, SDVariable in, boolean keepDims, int... dimensions) {
        validateNumerical("argmax", in);
        SDVariable ret = f().argmax(in, keepDims, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #argmax(String, SDVariable, boolean, int...)
     */
    public SDVariable argmax(SDVariable in, boolean keepDims, int... dimensions) {
        return argmax(null, in, keepDims, dimensions);
    }

    /**
     * Argmax array reduction operation, optionally along specified dimensions.<br>
     * Output values are the index of the maximum value of each slice along the specified dimension
     *
     * @param in         Input variable
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Reduced array of rank (input rank - num dimensions)
     */
    public SDVariable argmax(String name, SDVariable in, int... dimensions) {
        return argmax(name, in, false, dimensions);
    }

    /**
     * Argmin array reduction operation, optionally along specified dimensions.<br>
     * Output values are the index of the minimum value of each slice along the specified dimension
     *
     * @param in         Input variable
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Reduced array of rank (input rank - num dimensions)
     */
    public SDVariable argmin(SDVariable in, int... dimensions) {
        return argmin(null, in, dimensions);
    }

    /**
     * Argmin array reduction operation, optionally along specified dimensions.<br>
     * Output values are the index of the minimum value of each slice along the specified dimension
     *
     * @param in         Input variable
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Reduced array of rank (input rank - num dimensions)
     */
    public SDVariable argmin(String name, SDVariable in, int... dimensions) {
        return argmin(name, in, false, dimensions);
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
     * @param in         Input variable
     * @param keepDims   If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Output variable: reduced array of rank (input rank - num dimensions) if keepDims = false, or
     * of rank (input rank) if keepdims = true
     */
    public SDVariable argmin(String name, SDVariable in, boolean keepDims, int... dimensions) {
        validateNumerical("argmin", in);
        SDVariable ret = f().argmin(in, keepDims, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #argmin(String, SDVariable, boolean, int...)
     */
    public SDVariable argmin(SDVariable in, boolean keepDims, int... dimensions) {
        return argmin(null, in, keepDims, dimensions);
    }

    /**
     * Assign/copy op: out = x.assign(y). Supports broadcasting
     *
     * @param x Input variable x
     * @param y Input variable y
     * @return Output variable
     */
    public SDVariable assign(SDVariable x, SDVariable y) {
        return assign(null, x, y);
    }

    /**
     * Assign/copy op: out = x.assign(y). Supports broadcasting
     *
     * @param name Name of the output variable
     * @param x    Input variable x
     * @param y    Input variable y
     * @return Output variable
     */
    public SDVariable assign(String name, SDVariable x, SDVariable y) {
        SDVariable ret = f().assign(x, y);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Return an array with equal shape to the input, but all elements set to 'value'
     *
     * @param in    Input variable
     * @param value Value to set
     * @return Output variable
     */
    public SDVariable assign(SDVariable in, Number value) {
        return assign(null, in, value);
    }

    /**
     * Return an array with equal shape to the input, but all elements set to 'value'
     *
     * @param name  Name of the output variable
     * @param in    Input variable
     * @param value Value to set
     * @return Output variable
     */
    public SDVariable assign(String name, SDVariable in, Number value) {
        SDVariable ret = f().assign(in, value);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Matrix multiply a batch of matrices. matricesA and matricesB have to be arrays of same
     * length and each pair taken from these sets has to have dimensions (M, N) and (N, K),
     * respectively. If transposeA is true, matrices from matricesA will have shape (N, M) instead.
     * Likewise, if transposeB is true, matrices from matricesB will have shape (K, N).
     * <p>
     * <p>
     * The result of this operation will be a batch of multiplied matrices. The
     * result has the same length as both input batches and each output matrix is of shape (M, K).
     *
     * @param matricesA  First array of input matrices, all of shape (M, N) or (N, M)
     * @param matricesB  Second array of input matrices, all of shape (N, K) or (K, N)
     * @param transposeA whether first batch of matrices is transposed.
     * @param transposeB whether second batch of matrices is transposed.
     * @return Array of multiplied SDVariables of shape (M, K)
     */
    public SDVariable[] batchMmul(SDVariable[] matricesA, SDVariable[] matricesB,
                                  boolean transposeA, boolean transposeB) {
        return batchMmul(null, matricesA, matricesB, transposeA, transposeB);
    }

    /**
     * Matrix multiply a batch of matrices. matricesA and matricesB have to be arrays of same
     * length and each pair taken from these sets has to have dimensions (M, N) and (N, K),
     * respectively. If transposeA is true, matrices from matricesA will have shape (N, M) instead.
     * Likewise, if transposeB is true, matrices from matricesB will have shape (K, N).
     * <p>
     * <p>
     * The result of this operation will be a batch of multiplied matrices. The
     * result has the same length as both input batches and each output matrix is of shape (M, K).
     *
     * @param matricesA  First array of input matrices, all of shape (M, N) or (N, M)
     * @param matricesB  Second array of input matrices, all of shape (N, K) or (K, N)
     * @param transposeA whether first batch of matrices is transposed.
     * @param transposeB whether second batch of matrices is transposed.
     * @param names      names for all provided SDVariables
     * @return Array of multiplied SDVariables of shape (M, K)
     */
    public SDVariable[] batchMmul(String[] names, SDVariable[] matricesA, SDVariable[] matricesB,
                                  boolean transposeA, boolean transposeB) {
        validateSameType("batchMmul", true, matricesA);
        validateSameType("batchMmul", true, matricesB);
        SDVariable[] result = f().batchMmul(matricesA, matricesB, transposeA, transposeB);
        return updateVariableNamesAndReferences(result, names);
    }

    protected abstract SDVariable[] updateVariableNamesAndReferences(SDVariable[] variablesToUpdate, String[] newVariableNames);

    /**
     * Matrix multiply a batch of matrices. matricesA and matricesB have to be arrays of same
     * length and each pair taken from these sets has to have dimensions (M, N) and (N, K),
     * respectively. The result of this operation will be a batch of multiplied matrices. The
     * result has the same length as both input batches and each output matrix is of shape (M, K).
     *
     * @param matricesA First array of input matrices, all of shape (M, N)
     * @param matricesB Second array of input matrices, all of shape (N, K)
     * @return Array of multiplied SDVariables of shape (M, K)
     */
    public SDVariable[] batchMmul(SDVariable[] matricesA, SDVariable[] matricesB) {
        return batchMmul(null, matricesA, matricesB, false, false);
    }

    public SDVariable castTo(SDVariable toCast, org.nd4j.linalg.api.buffer.DataType toType) {
        return castTo(null, toCast, toType);
    }

    public SDVariable castTo(String name, SDVariable toCast, org.nd4j.linalg.api.buffer.DataType toType) {
        SDVariable ret = f().cast(toCast, toType);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #concat(String, int, SDVariable...)
     */
    public SDVariable concat(int dimension, SDVariable... inputs) {
        return concat(null, dimension, inputs);
    }

    /**
     * Concatenate a set of inputs along the specified dimension.<br>
     * Note that inputs must have identical rank and identical dimensions, other than the dimension to stack on.<br>
     * For example, if 2 inputs have shape [a, x, c] and [a, y, c] and dimension = 1, then the output has shape [a, x+y, c]
     *
     * @param name      Name of the output variable
     * @param dimension Dimension to concatenate on
     * @param inputs    Input variables
     * @return Output variable
     * @see #stack(String, int, SDVariable...)
     */
    public SDVariable concat(String name, int dimension, SDVariable... inputs) {
        validateSameType("concat", false, inputs);
        SDVariable result = f().concat(dimension, inputs);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @see #cumprod(String, SDVariable, boolean, boolean, int...)
     */
    public SDVariable cumprod(SDVariable in, boolean exclusive, boolean reverse, int... axis) {
        return cumprod(null, in, exclusive, reverse, axis);
    }

    /**
     * Cumulative product operation.<br>
     * For input: [ a, b, c], output is:<br>
     * exclusize=false, reverse=false: [a, a*b, a*b*c]<br>
     * exclusive=true, reverse=false, [0, a, a*b]<br>
     * exclusive=false, reverse=true: [a*b*c, b*c, c]<br>
     * exclusive=true, reverse=true: [b*c, c, 0]<br><br>
     *
     * @param name      Name of the output variable
     * @param in        Input variable
     * @param axis      Scalar axis argument for dimension to perform cumululative sum operations along
     * @param exclusive If true: exclude the first value
     * @param reverse   If true: reverse the direction of the accumulation
     * @return Output variable
     */
    public SDVariable cumprod(String name, SDVariable in, boolean exclusive, boolean reverse, int... axis) {
        validateNumerical("cumprod", in);
        SDVariable ret = f().cumprod(in, exclusive, reverse, axis);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #cumsum(String, SDVariable, boolean, boolean, int...)
     */
    public SDVariable cumsum(SDVariable in, boolean exclusive, boolean reverse, int... axis) {
        return cumsum(null, in, exclusive, reverse, axis);
    }

    /**
     * Cumulative sum operation.<br>
     * For input: [ a, b, c], output is:<br>
     * exclusize=false, reverse=false: [a, a+b, a+b+c]<br>
     * exclusive=true, reverse=false, [0, a, a+b]<br>
     * exclusive=false, reverse=true: [a+b+c, b+c, c]<br>
     * exclusive=true, reverse=true: [b+c, c, 0]<br><br>
     *
     * @param name      Name of the output variable
     * @param in        Input variable
     * @param axis      Scalar axis argument for dimension to perform cumululative sum operations along
     * @param exclusive If true: exclude the first value
     * @param reverse   If true: reverse the direction of the accumulation
     * @return Output variable
     */
    public SDVariable cumsum(String name, SDVariable in, boolean exclusive, boolean reverse, int... axis) {
        validateNumerical("cumsum", in);
        SDVariable ret = f().cumsum(in, exclusive, reverse, axis);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * TODO doc string
     *
     * @param x
     * @param y
     * @param dimensions
     * @return
     */
    public SDVariable dot(SDVariable x, SDVariable y, int... dimensions) {
        return dot(null, x, y, dimensions);
    }

    /**
     * TODO doc string
     *
     * @param name
     * @param x
     * @param y
     * @param dimensions
     * @return
     */
    public SDVariable dot(String name, SDVariable x, SDVariable y, int... dimensions) {
        SDValidation.validateNumerical("dot", x, y);
        SDVariable ret = f().dot(x, y, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #dynamicPartition(String[], SDVariable, SDVariable, int)
     */
    public SDVariable[] dynamicPartition(SDVariable x, SDVariable partitions, int numPartitions) {
        return dynamicPartition(null, x, partitions, numPartitions);
    }

    /**
     * Dynamically partition the input variable values into the specified number of paritions, using the indices.<br>
     * Example:<br>
     * <pre>
     * {@code input = [1,2,3,4,5]
     * numPartitions = 2
     * partitions = [1,0,0,1,0]
     * out[0] = [2,3,5]
     * out[1] = [1,4] }
     * </pre>
     *
     * @param name          Names for the output variables. Length must be equal to numPartitions
     * @param x             Input variable
     * @param partitions    1D input with values 0 to numPartitions-1
     * @param numPartitions Number of partitions, >= 1
     * @return Output variables (equal in number to numPartitions)
     */
    public SDVariable[] dynamicPartition(String[] name, SDVariable x, SDVariable partitions, int numPartitions) {
        SDVariable[] ret = f().dynamicPartition(x, partitions, numPartitions);
        return updateVariableNamesAndReferences(ret, name);
    }

    /**
     * @see #dynamicStitch(String, SDVariable[], SDVariable[])
     */
    public SDVariable dynamicStitch(SDVariable[] indices, SDVariable[] x) {
        return dynamicStitch(null, indices, x);
    }

    /**
     * Dynamically merge the specified input arrays into a single array, using the specified indices
     *
     * @param name    Name of the output variable
     * @param indices Indices to use when merging. Must be >= 1, same length as input variables
     * @param x       Input variables.
     * @return Merged output variable
     */
    public SDVariable dynamicStitch(String name, SDVariable[] indices, SDVariable[] x) {
        SDVariable ret = f().dynamicStitch(indices, x);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Equals operation: elementwise x == y<br>
     * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
     * value 0 otherwise
     *
     * @param x Input array
     * @param y Double value argument to use in operation
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable eq(SDVariable x, double y) {
        return eq(null, x, y);
    }

    /**
     * Equals operation: elementwise x == y<br>
     * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
     * value 0 otherwise
     *
     * @param name Name of the output variable
     * @param x    Input array
     * @param y    Double value argument to use in operation
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable eq(String name, SDVariable x, double y) {
        SDVariable result = f().eq(x, y);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Equal to operation: elementwise x == y<br>
     * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
     *
     * @param x Input 1
     * @param y Input 2
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable eq(SDVariable x, SDVariable y) {
        return eq(null, x, y);
    }

    /**
     * Equal to operation: elementwise x == y<br>
     * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
     *
     * @param name Name of the output variable
     * @param x    Input 1
     * @param y    Input 2
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable eq(String name, SDVariable x, SDVariable y) {
        SDVariable result = f().eq(x, y);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @see #expandDims(String, SDVariable, int)
     */
    public SDVariable expandDims(SDVariable x, int axis) {
        return expandDims(null, x, axis);
    }

    /**
     * Reshape the input by adding a 1 at the specified location.<br>
     * For example, if input has shape [a, b], then output shape is:<br>
     * axis = 0: [1, a, b]<br>
     * axis = 1: [a, 1, b]<br>
     * axis = 2: [a, b, 1]<br>
     *
     * @param name Name of the output variable
     * @param x    Input variable
     * @param axis Axis to expand
     * @return Output variable
     * @see #squeeze(String, SDVariable, int)
     */
    public SDVariable expandDims(String name, SDVariable x, int axis) {
        SDVariable result = f().expandDims(x, axis);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Generate an output variable with the specified (dynamic) shape with all elements set to the specified value
     *
     * @param shape Shape: must be a 1D array/variable
     * @param value Value to set all elements to
     * @return Output variable
     */
    public SDVariable fill(SDVariable shape, org.nd4j.linalg.api.buffer.DataType dataType, double value) {
        return fill(null, shape, dataType, value);
    }

    /**
     * Generate an output variable with the specified (dynamic) shape with all elements set to the specified value
     *
     * @param name  Name of the output variable
     * @param shape Shape: must be a 1D array/variable
     * @param value Value to set all elements to
     * @return Output variable
     */
    public SDVariable fill(String name, SDVariable shape, org.nd4j.linalg.api.buffer.DataType dataType, double value) {
        SDVariable result = f().fill(shape, dataType, value);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @see #gather(String, SDVariable, int[], int)
     */
    public SDVariable gather(SDVariable df, int[] indices, int axis) {
        return gather(null, df, indices, axis);
    }

    /**
     * Gather slices from the input variable where the indices are specified as fixed int[] values.<br>
     * Output shape is same as input shape, except for axis dimension, which has size equal to indices.length.
     *
     * @param name    name of the output variable
     * @param df      Input variable
     * @param indices Indices to get
     * @param axis    Axis that the indices refer to
     * @return Output variable with slices pulled from the specified axis
     */
    public SDVariable gather(String name, SDVariable df, int[] indices, int axis) {
        SDVariable ret = f().gather(df, indices, axis);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #gather(String, SDVariable, SDVariable, int)
     */
    public SDVariable gather(SDVariable df, SDVariable indices, int axis) {
        return gather(null, df, indices, axis);
    }

    /**
     * Gather slices from the input variable where the indices are specified as dynamic SDVariable values.<br>
     * Output shape is same as input shape, except for axis dimension, which has size equal to indices.length.
     *
     * @param name    name of the output variable
     * @param df      Input variable
     * @param indices Indices to get slices for. Rank 0 or 1 input
     * @param axis    Axis that the indices refer to
     * @return Output variable with slices pulled from the specified axis
     */
    public SDVariable gather(String name, SDVariable df, SDVariable indices, int axis) {
        SDVariable ret = f().gather(df, indices, axis);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * TODO doc string
     *
     * @param df
     * @param indices
     * @return
     */
    public SDVariable gatherNd(SDVariable df, SDVariable indices) {
        return gatherNd(null, df, indices);
    }

    /**
     * TODO doc string
     *
     * @param name
     * @param df
     * @param indices
     * @return
     */
    public SDVariable gatherNd(String name, SDVariable df, SDVariable indices) {
        SDVariable ret = f().gatherNd(df, indices);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Greater than operation: elementwise x > y<br>
     * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
     * value 0 otherwise
     *
     * @param x Input array
     * @param y Double value argument to use in operation
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable gt(SDVariable x, double y) {
        return gt(null, x, y);
    }

    /**
     * Greater than operation: elementwise x > y<br>
     * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
     * value 0 otherwise
     *
     * @param name Name of the output variable
     * @param x    Input array
     * @param y    Double value argument to use in operation
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable gt(String name, SDVariable x, double y) {
        validateNumerical("greater than (gt)", x);
        SDVariable result = f().gt(x, y);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Greater than operation: elementwise x > y<br>
     * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
     *
     * @param x Input 1
     * @param y Input 2
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable gt(SDVariable x, SDVariable y) {
        return gt(null, x, y);
    }

    /**
     * Greater than operation: elementwise x > y<br>
     * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
     *
     * @param name Name of the output variable
     * @param x    Input 1
     * @param y    Input 2
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable gt(String name, SDVariable x, SDVariable y) {
        SDValidation.validateNumerical("greater than (gt)", x, y);
        SDVariable result = f().gt(x, y);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Greater than or equals operation: elementwise x >= y<br>
     * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
     * value 0 otherwise
     *
     * @param x Input array
     * @param y Double value argument to use in operation
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable gte(SDVariable x, double y) {
        return gte(null, x, y);
    }

    /**
     * Greater than or equals operation: elementwise x >= y<br>
     * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
     * value 0 otherwise
     *
     * @param name Name of the output variable
     * @param x    Input array
     * @param y    Double value argument to use in operation
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable gte(String name, SDVariable x, double y) {
        validateNumerical("greater than or equal (gte)", x);
        SDVariable result = f().gte(x, y);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Greater than or equal to operation: elementwise x >= y<br>
     * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
     *
     * @param x Input 1
     * @param y Input 2
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable gte(SDVariable x, SDVariable y) {
        return gte(null, x, y);
    }

    /**
     * Greater than or equal to operation: elementwise x >= y<br>
     * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
     *
     * @param name Name of the output variable
     * @param x    Input 1
     * @param y    Input 2
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable gte(String name, SDVariable x, SDVariable y) {
        validateNumerical("greater than or equal (gte)", x, y);
        SDVariable result = f().gte(x, y);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Elementwise identity operation: out = x
     *
     * @param input Input variable
     * @return Output variable
     */
    public SDVariable identity(SDVariable input) {
        return identity(null, input);
    }

    /**
     * Elementwise identity operation: out = x
     *
     * @param name  name of the output variable
     * @param input Input variable
     * @return Output variable
     */
    public SDVariable identity(String name, SDVariable input) {
        SDVariable s = f().identity(input);
        return updateVariableNameAndReference(s, name);
    }

    /**
     * Compute the inverse permutation indices for a permutation operation<br>
     * Example: if input is [2, 0, 1] then output is [1, 2, 0]<br>
     * The idea is that x.permute(input).permute(invertPermutation(input)) == x
     *
     * @param input 1D indices for permutation
     * @return 1D inverted permutation
     */
    public SDVariable invertPermutation(SDVariable input) {
        return invertPermutation(null, input);
    }

    /**
     * Compute the inverse permutation indices for a permutation operation<br>
     * Example: if input is [2, 0, 1] then output is [1, 2, 0]<br>
     * The idea is that x.permute(input).permute(invertPermutation(input)) == x
     *
     * @param name  name of the output variable
     * @param input 1D indices for permutation
     * @return 1D inverted permutation
     */
    public SDVariable invertPermutation(String name, SDVariable input) {
        validateInteger("invert permutation", input);
        SDVariable ret = f().invertPermutation(input, false);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Is the director a numeric tensor? In the current version of ND4J/SameDiff, this always returns true/1
     *
     * @param x Input variable
     * @return Scalar variable with value 1
     */
    public SDVariable isNumericTensor(SDVariable x) {
        return isNumericTensor(null, x);
    }

    /**
     * Is the director a numeric tensor? In the current version of ND4J/SameDiff, this always returns true/1
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Scalar variable with value 1
     */
    public SDVariable isNumericTensor(String name, SDVariable x) {
        SDVariable result = f().isNumericTensor(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Create a new 1d array with values evenly spaced between values 'start' and 'stop'
     * For example, linspace(start=3.0, stop=4.0, number=3) will generate [3.0, 3.5, 4.0]
     *
     * @param start  Start value
     * @param stop   Stop value
     * @param number Number of values to generate
     * @return SDVariable with linearly spaced elements
     */
    public SDVariable linspace(DataType dataType, double start, double stop, long number) {
        return linspace(dataType, start, stop, number);
    }

    /**
     * Create a new 1d array with values evenly spaced between values 'start' and 'stop'
     * For example, linspace(start=3.0, stop=4.0, number=3) will generate [3.0, 3.5, 4.0]
     *
     * @param name     Name of the new variable
     * @param dataType Data type of the output array
     * @param start    Start value
     * @param stop     Stop value
     * @param number   Number of values to generate
     * @return SDVariable with linearly spaced elements
     */
    public SDVariable linspace(String name, DataType dataType, double start, double stop, long number) {
        SDVariable ret = f().linspace(sd().constant(start), sd().constant(stop), sd().constant(number), dataType);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Create a new 1d array with values evenly spaced between values 'start' and 'stop'
     * For example, linspace(start=3.0, stop=4.0, number=3) will generate [3.0, 3.5, 4.0]
     *
     * @param name   Name of the new variable
     * @param from   Start value
     * @param to     Stop value
     * @param length Number of values to generate
     * @param dt     Data type of the output array
     * @return SDVariable with linearly spaced elements
     */
    public SDVariable linspace(String name, SDVariable from, SDVariable to, SDVariable length, DataType dt) {
        SDVariable ret = f().linspace(from, to, length, dt);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Less than operation: elementwise x < y<br>
     * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
     * value 0 otherwise
     *
     * @param x Input array
     * @param y Double value argument to use in operation
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable lt(SDVariable x, double y) {
        return lt(null, x, y);
    }

    /**
     * Less than operation: elementwise x < y<br>
     * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
     * value 0 otherwise
     *
     * @param name Name of the output variable
     * @param x    Input array
     * @param y    Double value argument to use in operation
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable lt(String name, SDVariable x, double y) {
        validateNumerical("less than (lt)", x);
        SDVariable result = f().lt(x, y);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Less than operation: elementwise x < y<br>
     * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
     *
     * @param x Input 1
     * @param y Input 2
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable lt(SDVariable x, SDVariable y) {
        return lt(null, x, y);
    }

    /**
     * Less than operation: elementwise x < y<br>
     * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
     *
     * @param name Name of the output variable
     * @param x    Input 1
     * @param y    Input 2
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable lt(String name, SDVariable x, SDVariable y) {
        validateNumerical("less than (lt)", x, y);
        SDVariable result = f().lt(x, y);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Less than or equals operation: elementwise x <= y<br>
     * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
     * value 0 otherwise
     *
     * @param x Input array
     * @param y Double value argument to use in operation
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable lte(SDVariable x, double y) {
        return lte(null, x, y);
    }

    /**
     * Less than or equals operation: elementwise x <= y<br>
     * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
     * value 0 otherwise
     *
     * @param name Name of the output variable
     * @param x    Input array
     * @param y    Double value argument to use in operation
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable lte(String name, SDVariable x, double y) {
        validateNumerical("less than or equal (lte)", x);
        SDVariable result = f().lte(x, y);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Less than or equal to operation: elementwise x <= y<br>
     * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
     *
     * @param x Input 1
     * @param y Input 2
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable lte(SDVariable x, SDVariable y) {
        return lte(null, x, y);
    }

    /**
     * Less than or equal to operation: elementwise x <= y<br>
     * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
     *
     * @param name Name of the output variable
     * @param x    Input 1
     * @param y    Input 2
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable lte(String name, SDVariable x, SDVariable y) {
        validateNumerical("less than or equal (lte)", x, y);
        SDVariable result = f().lte(x, y);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Returns a boolean mask of equal shape to the input, where the condition is satisfied - value 1 where satisfied, 0 otherwise
     *
     * @param in        Input variable
     * @param condition Condition
     * @return Boolean mask mariable
     */
    public SDVariable matchCondition(SDVariable in, Condition condition) {
        return matchCondition(null, in, condition);
    }

    /**
     * Returns a boolean mask of equal shape to the input, where the condition is satisfied - value 1 where satisfied, 0 otherwise
     *
     * @param in        Input
     * @param condition Condition
     * @return Boolean mask
     */
    public SDVariable matchCondition(String name, SDVariable in, Condition condition) {
        SDVariable ret = f().matchCondition(in, condition);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Returns a count of the number of elements that satisfy the condition
     *
     * @param in        Input
     * @param condition Condition
     * @return Number of elements that the condition is satisfied for
     */
    public SDVariable matchConditionCount(SDVariable in, Condition condition) {
        return matchConditionCount(null, in, condition);
    }

    /**
     * Returns a count of the number of elements that satisfy the condition
     *
     * @param name      Name of the output variable
     * @param in        Input
     * @param condition Condition
     * @return Number of elements that the condition is satisfied for
     */
    public SDVariable matchConditionCount(String name, SDVariable in, Condition condition) {
        return matchConditionCount(name, in, condition, false);
    }

    /**
     * Returns a count of the number of elements that satisfy the condition (for each slice along the specified dimensions)<br>
     * Note that if keepDims = true, the output variable has the same rank as the input variable,
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]
     *
     * @param name       Name of the output variable
     * @param in         Input variable
     * @param condition  Condition
     * @param keepDim    If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Number of elements that the condition is satisfied for
     */
    public SDVariable matchConditionCount(String name, SDVariable in, Condition condition, boolean keepDim, int... dimensions) {
        SDVariable ret = f().matchConditionCount(in, condition, keepDim, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Max array reduction operation, optionally along specified dimensions
     *
     * @param x          Input variable
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Reduced array of rank (input rank - num dimensions)
     */
    public SDVariable max(SDVariable x, int... dimensions) {
        return max(null, x, dimensions);
    }

    /**
     * Max array reduction operation, optionally along specified dimensions
     *
     * @param name       Output variable name
     * @param x          Input variable
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Reduced array of rank (input rank - num dimensions)
     */
    public SDVariable max(String name, SDVariable x, int... dimensions) {
        return max(name, x, false, dimensions);
    }

    /**
     * Max array reduction operation, optionally along specified dimensions<br>
     * Note that if keepDims = true, the output variable has the same rank as the input variable,
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]
     *
     * @param name       Output variable name
     * @param x          Input variable
     * @param keepDims   If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Reduced array of rank (input rank - num dimensions)
     */
    public SDVariable max(String name, SDVariable x, boolean keepDims, int... dimensions) {
        validateNumerical("max reduction", x);
        SDVariable result = f().max(x, keepDims, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Element-wise maximum operation: out[i] = max(first[i], second[i])<br>
     * Supports broadcasting
     *
     * @param first  First input array
     * @param second Second input array
     * @return Output variable
     */
    public SDVariable max(SDVariable first, SDVariable second) {
        return max(null, first, second);
    }

    /**
     * Element-wise maximum operation: out[i] = max(first[i], second[i])<br>
     * Supports broadcasting
     *
     * @param name   Name of the output variable
     * @param first  First input array
     * @param second Second input array
     * @return Output variable
     */
    public SDVariable max(String name, SDVariable first, SDVariable second) {
        validateNumerical("pairwise maxiumum (max)", first, second);
        SDVariable result = f().max(first, second);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Full array mean reduction operation
     *
     * @param x Input variable
     * @return Output variable - scalar
     */
    public SDVariable mean(SDVariable x) {
        return mean(null, x);
    }

    /**
     * Mean (average) array reduction operation, optionally along specified dimensions
     *
     * @param name      Output variable name
     * @param x         Input variable
     * @param dimension Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Reduced array of rank (input rank - num dimensions)
     */
    public SDVariable mean(String name, SDVariable x, int... dimension) {
        return mean(name, x, false, dimension);
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
     * @param name      Output variable name
     * @param x         Input variable
     * @param keepDims  If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimension Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Reduced array of rank (input rank - num dimensions)
     */
    public SDVariable mean(String name, SDVariable x, boolean keepDims, int... dimension) {
        validateNumerical("mean reduction", x);
        SDVariable result = f().mean(x, keepDims, dimension);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Mean (average) array reduction operation, optionally along specified dimensions
     *
     * @param x         Input variable
     * @param dimension Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Reduced array of rank (input rank - num dimensions)
     */
    public SDVariable mean(SDVariable x, int... dimension) {
        return mean(null, x, dimension);
    }

    /**
     * Minimum array reduction operation, optionally along specified dimensions. out = min(in)
     *
     * @param x          Input variable
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Reduced array of rank (input rank - num dimensions)
     */
    public SDVariable min(SDVariable x, int... dimensions) {
        return min(null, x, dimensions);
    }

    /**
     * Minimum array reduction operation, optionally along specified dimensions. out = min(in)
     *
     * @param name       Output variable name
     * @param x          Input variable
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Reduced array of rank (input rank - num dimensions)
     */
    public SDVariable min(String name, SDVariable x, int... dimensions) {
        return min(name, x, false, dimensions);
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
     * @param x          Input variable
     * @param keepDims   If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Reduced array of rank (input rank - num dimensions)
     */
    public SDVariable min(String name, SDVariable x, boolean keepDims, int... dimensions) {
        validateNumerical("min reduction", x);
        SDVariable result = f().min(x, keepDims, dimensions);
        return updateVariableNameAndReference(result, name);

    }

    /**
     * Element-wise minimum operation: out[i] = min(first[i], second[i])<br>
     * Supports broadcasting
     *
     * @param first  First input array
     * @param second Second input array
     * @return Output variable
     */
    public SDVariable min(SDVariable first, SDVariable second) {
        return min(null, first, second);
    }

    /**
     * Element-wise minimum operation: out[i] = min(first[i], second[i])<br>
     * Supports broadcasting
     *
     * @param name   Name of the output variable
     * @param first  First input array
     * @param second Second input array
     * @return Output variable
     */
    public SDVariable min(String name, SDVariable first, SDVariable second) {
        validateNumerical("mean (pairwise)", first, second);
        SDVariable result = f().min(first, second);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Matrix multiplication: out = mmul(x,y)<br>
     * Supports specifying a {@link MMulTranspose} argument to perform operation such as mmul(a^T, b), etc.
     *
     * @param x         First input variable
     * @param y         Second input variable
     * @param transpose Transpose arguments
     * @return Output variable
     */
    public SDVariable mmul(SDVariable x, SDVariable y, MMulTranspose transpose) {
        return mmul(null, x, y, transpose);

    }

    /**
     * Matrix multiplication: out = mmul(x,y)<br>
     * Supports specifying a {@link MMulTranspose} argument to perform operation such as mmul(a^T, b), etc.
     *
     * @param name      Output variable name
     * @param x         First input variable
     * @param y         Second input variable
     * @param transpose Transpose arguments
     * @return Output variable
     */
    public SDVariable mmul(String name, SDVariable x, SDVariable y, MMulTranspose transpose) {
        validateNumerical("matrix multiplication (mmul)", x, y);
        SDVariable result = f().mmul(x, y, transpose);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Matrix multiplication: out = mmul(x,y)
     *
     * @param x First input variable
     * @param y Second input variable
     * @return Output variable
     */
    public SDVariable mmul(SDVariable x, SDVariable y) {
        return mmul(null, x, y);
    }

    /**
     * Matrix multiplication: out = mmul(x,y)
     *
     * @param name Output variable name
     * @param x    First input variable
     * @param y    Second input variable
     * @return Output variable
     */
    public SDVariable mmul(String name, SDVariable x, SDVariable y) {
        return mmul(name, x, y, MMulTranspose.allFalse());
    }

    /**
     * Not equals operation: elementwise x != y<br>
     * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
     * value 0 otherwise
     *
     * @param x Input array
     * @param y Double value argument to use in operation
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable neq(SDVariable x, double y) {
        return neq(null, x, y);
    }

    /**
     * Not equals operation: elementwise x != y<br>
     * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
     * value 0 otherwise
     *
     * @param name Name of the output variable
     * @param x    Input array
     * @param y    Double value argument to use in operation
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable neq(String name, SDVariable x, double y) {
        validateNumerical("not equals (neq)", x);
        SDVariable result = f().neq(x, y);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Not equal to operation: elementwise x != y<br>
     * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
     *
     * @param x Input 1
     * @param y Input 2
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable neq(SDVariable x, SDVariable y) {
        return neq(null, x, y);
    }

    /**
     * Not equal to operation: elementwise x != y<br>
     * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
     *
     * @param name Name of the output variable
     * @param x    Input 1
     * @param y    Input 2
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable neq(String name, SDVariable x, SDVariable y) {
        validateNumerical("not equals (neq)", x, y);
        SDVariable result = f().neq(x, y);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Norm1 (L1 norm) reduction operation: The output contains the L1 norm for each tensor/subset along the specified dimensions:<br>
     * out = sum_i abs(x[i])
     *
     * @param name       Output variable name
     * @param x          Input variable
     * @param dimensions dimensions to reduce over
     * @return Output variable
     */
    public SDVariable norm1(String name, SDVariable x, int... dimensions) {
        return norm1(name, x, false, dimensions);
    }

    /**
     * Norm1 (L1 norm) reduction operation: The output contains the L1 norm for each tensor/subset along the specified dimensions:<br>
     * out = sum_i abs(x[i])<br>
     * Note that if keepDims = true, the output variable has the same rank as the input variable,
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]
     *
     * @param name       Output variable name
     * @param x          Input variable
     * @param keepDims   If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions dimensions to reduce over
     * @return Output variable
     */
    public SDVariable norm1(String name, SDVariable x, boolean keepDims, int... dimensions) {
        validateNumerical("norm1 reduction", x);
        SDVariable result = f().norm1(x, keepDims, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Norm2 (L2 norm) reduction operation: The output contains the L2 norm for each tensor/subset along the specified dimensions:<br>
     * out = sqrt(sum_i x[i]^2)
     *
     * @param name       Output variable name
     * @param x          Input variable
     * @param dimensions dimensions to reduce over
     * @return Output variable
     */
    public SDVariable norm2(String name, SDVariable x, int... dimensions) {
        return norm2(name, x, false, dimensions);
    }

    /**
     * Norm2 (L2 norm) reduction operation: The output contains the L2 norm for each tensor/subset along the specified dimensions:<br>
     * out = sqrt(sum_i x[i]^2)<br>
     * Note that if keepDims = true, the output variable has the same rank as the input variable,
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]
     *
     * @param name       Output variable name
     * @param x          Input variable
     * @param keepDims   If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions dimensions to reduce over
     * @return Output variable
     */
    public SDVariable norm2(String name, SDVariable x, boolean keepDims, int... dimensions) {
        validateNumerical("norm2 reduction", x);
        SDVariable result = f().norm2(x, keepDims, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Max norm (infinity norm) reduction operation: The output contains the max norm for each tensor/subset along the
     * specified dimensions
     *
     * @param name       Output variable name
     * @param x          Input variable
     * @param dimensions dimensions to reduce over
     * @return Output variable
     */
    public SDVariable normmax(String name, SDVariable x, int... dimensions) {
        return normmax(name, x, false, dimensions);
    }

    /**
     * Max norm (infinity norm) reduction operation: The output contains the max norm for each tensor/subset along the
     * specified dimensions:<br>
     * out = max(abs(x[i]))<br>
     * Note that if keepDims = true, the output variable has the same rank as the input variable,
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]
     *
     * @param name       Output variable name
     * @param x          Input variable
     * @param keepDims   If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions dimensions to reduce over
     * @return Output variable
     */
    public SDVariable normmax(String name, SDVariable x, boolean keepDims, int... dimensions) {
        validateNumerical("norm max reduction", x);
        SDVariable result = f().normmax(x, keepDims, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @see #oneHot(String, SDVariable, int)
     */
    public SDVariable oneHot(SDVariable indices, int depth) {
        return oneHot(null, indices, depth, -1, 1.00, 0.00);
    }

    /**
     * Convert the array to a one-hot array with walues {@code on} and {@code off} for each entry<br>
     * If input has shape [ a, ..., n] then output has shape [ a, ..., n, depth],
     * with {@code out[i, ..., j, in[i,...,j]] = on} with other values being set to {@code off}
     *
     * @param name    Output variable name
     * @param indices Indices - value 0 to depth-1
     * @param depth   Number of classes
     * @return Output variable
     */
    public SDVariable oneHot(String name, SDVariable indices, int depth, int axis, double on, double off) {
        return oneHot(name, indices, depth, axis, on, off, OneHot.DEFAULT_DTYPE);
    }

    /**
     * As per {@link #oneHot(String, SDVariable, int, int, double, double)} but allows configuring the output datatype
     */
    public SDVariable oneHot(String name, SDVariable indices, int depth, int axis, double on, double off, DataType dataType) {
        validateInteger("oneHot", "indices", indices);
        SDVariable ret = f().onehot(indices, depth, axis, on, off, dataType);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #oneHot(String, SDVariable, int, int, double, double)
     */
    public SDVariable oneHot(SDVariable indices, int depth, int axis, double on, double off) {
        return oneHot(null, indices, depth, axis, on, off, OneHot.DEFAULT_DTYPE);
    }

    /**
     * @see #oneHot(String, SDVariable, int, int, double, double, DataType)
     */
    public SDVariable oneHot(SDVariable indices, int depth, int axis, double on, double off, DataType dataType) {
        return oneHot(null, indices, depth, axis, on, off, dataType);
    }

    /**
     * Convert the array to a one-hot array with walues 0 and 1 for each entry<br>
     * If input has shape [ a, ..., n] then output has shape [ a, ..., n, depth],
     * with out[i, ..., j, in[i,...,j]] = 1 with other values being set to 0
     *
     * @param name    Output variable name
     * @param indices Indices - value 0 to depth-1
     * @param depth   Number of classes
     * @return Output variable
     * @see #oneHot(SDVariable, int, int, double, double)
     */
    public SDVariable oneHot(String name, SDVariable indices, int depth) {
        return oneHot(name, indices, depth, -1, 1.00, 0.00);
    }

    /**
     * Return a variable of all 1s, with the same shape as the input variable. Note that this is dynamic:
     * if the input shape changes in later execution, the returned variable's shape will also be updated
     *
     * @param input Input SDVariable
     * @return A new SDVariable with the same (dynamic) shape as the input
     */
    public SDVariable onesLike(SDVariable input) {
        return onesLike(null, input);
    }

    /**
     * Return a variable of all 1s, with the same shape as the input variable. Note that this is dynamic:
     * if the input shape changes in later execution, the returned variable's shape will also be updated
     *
     * @param name  Name of the new SDVariable
     * @param input Input SDVariable
     * @return A new SDVariable with the same (dynamic) shape as the input
     */
    public SDVariable onesLike(String name, SDVariable input) {
        return onesLike(name, input, input.dataType());
    }

    /**
     * As per {@link #onesLike(String, SDVariable)} but the output datatype may be specified
     */
    public SDVariable onesLike(String name, @NonNull SDVariable input, @NonNull DataType dataType) {
        SDVariable ret = f().onesLike(name, input, dataType);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #stack(String, int, SDVariable...)
     */
    public SDVariable parallel_stack(SDVariable[] values) {
        return parallel_stack(null, values);
    }

    /**
     * @see #stack(String, int, SDVariable...)
     */
    public SDVariable parallel_stack(String name, SDVariable[] values) {
        validateSameType("parallel_stack", false, values);
        SDVariable ret = f().parallel_stack(values);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Array permutation operation: permute the dimensions according to the specified permutation indices.<br>
     * Example: if input has shape [a,b,c] and dimensions = [2,0,1] the output has shape [c,a,b]
     *
     * @param x Input variable
     * @return Output variable (permuted input)
     */
    public SDVariable permute(SDVariable x, int... dimensions) {
        return permute(null, x, dimensions);
    }

    /**
     * Array permutation operation: permute the dimensions according to the specified permutation indices.<br>
     * Example: if input has shape [a,b,c] and dimensions = [2,0,1] the output has shape [c,a,b]
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable (permuted input)
     */
    public SDVariable permute(String name, SDVariable x, int... dimensions) {
        SDVariable result = f().permute(x, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Product array reduction operation, optionally along specified dimensions
     *
     * @param x          Input variable
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Output variable: reduced array of rank (input rank - num dimensions)
     */
    public SDVariable prod(SDVariable x, int... dimensions) {
        return prod(null, x, dimensions);
    }

    /**
     * Product array reduction operation, optionally along specified dimensions
     *
     * @param name       Output variable name
     * @param x          Input variable
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Output variable: reduced array of rank (input rank - num dimensions)
     */
    public SDVariable prod(String name, SDVariable x, int... dimensions) {
        return prod(name, x, false, dimensions);
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
     * @param x          Input variable
     * @param keepDims   If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Output variable: reduced array of rank (input rank - num dimensions)
     */
    public SDVariable prod(String name, SDVariable x, boolean keepDims, int... dimensions) {
        validateNumerical("product reduction (prod)", x);
        SDVariable result = f().prod(x, keepDims, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Create a new variable with a 1d array, where the values start at {@code from} and increment by {@code step}
     * up to (but not including) limit.<br>
     * For example, {@code range(1.0, 3.0, 0.5)} will return {@code [1.0, 1.5, 2.0, 2.5]}
     *
     * @param from     Initial/smallest value
     * @param to       Largest value (exclusive)
     * @param step     Step size
     * @param dataType The output variable datatype
     * @return 1D SDVariable with the specified values
     */
    public SDVariable range(double from, double to, double step, DataType dataType) {
        return range(null, from, to, step, dataType);
    }

    /**
     * Create a new variable with a 1d array, where the values start at {@code from} and increment by {@code step}
     * up to (but not including) limit.<br>
     * For example, {@code range(1.0, 3.0, 0.5)} will return {@code [1.0, 1.5, 2.0, 2.5]}
     *
     * @param name Name of the new variable
     * @param from Initial/smallest value
     * @param to   Largest value (exclusive)
     * @param step Step size
     * @return 1D SDVariable with the specified values
     */
    public SDVariable range(String name, double from, double to, double step, DataType dataType) {
        SDVariable ret = f().range(from, to, step, dataType);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Returns the rank (number of dimensions, i.e., length(shape)) of the specified SDVariable as a 0D scalar variable
     *
     * @param in Input variable
     * @return 0D (scalar) output variable with value equal to the rank of the input variable
     */
    public SDVariable rank(SDVariable in) {
        return rank(null, in);
    }

    /**
     * Returns the rank (number of dimensions, i.e., length(shape)) of the specified SDVariable as a 0D scalar variable
     *
     * @param name Name of the output variable
     * @param in   Input variable
     * @return 0D (scalar) output variable with value equal to the rank of the input variable
     */
    public SDVariable rank(String name, SDVariable in) {
        SDVariable ret = f().rank(in);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #repeat(String, SDVariable, int)
     */
    public SDVariable repeat(SDVariable df, int axis) {
        return repeat(null, df, axis);
    }

    /**
     * @see #repeat(String, SDVariable, int)
     */
    public SDVariable repeat(String name, SDVariable df, int axis) {
        SDVariable ret = f().repeat(df, axis);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Element-wise replace where condition:<br>
     * out[i] = from[i] if condition(update[i]) is satisfied, or<br>
     * out[i] = update[i] if condition(update[i]) is NOT satisfied
     *
     * @param update    Source array
     * @param from      Replacement values array (used conditionally). Must be same shape as 'update' array
     * @param condition Condition to check on update array elements
     * @return New array with values replaced where condition is satisfied
     */
    public SDVariable replaceWhere(SDVariable update, SDVariable from, Condition condition) {
        return replaceWhere(null, update, from, condition);
    }

    /**
     * Element-wise replace where condition:<br>
     * out[i] = from[i] if condition(update[i]) is satisfied, or<br>
     * out[i] = update[i] if condition(update[i]) is NOT satisfied
     *
     * @param name      Name of the output variable
     * @param update    Source array
     * @param from      Replacement values array (used conditionally). Must be same shape as 'update' array
     * @param condition Condition to check on update array elements
     * @return New array with values replaced where condition is satisfied
     */
    public SDVariable replaceWhere(String name, SDVariable update, SDVariable from, Condition condition) {
        SDVariable ret = f().replaceWhere(update, from, condition);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Element-wise replace where condition:<br>
     * out[i] = value if condition(update[i]) is satisfied, or<br>
     * out[i] = update[i] if condition(update[i]) is NOT satisfied
     *
     * @param update    Source array
     * @param value     Value to set at the output, if the condition is satisfied
     * @param condition Condition to check on update array elements
     * @return New array with values replaced where condition is satisfied
     */
    public SDVariable replaceWhere(SDVariable update, Number value, Condition condition) {
        return replaceWhere(null, update, value, condition);
    }

    /**
     * Element-wise replace where condition:<br>
     * out[i] = value if condition(update[i]) is satisfied, or<br>
     * out[i] = update[i] if condition(update[i]) is NOT satisfied
     *
     * @param name      Name of the output variable
     * @param update    Source array
     * @param value     Value to set at the output, if the condition is satisfied
     * @param condition Condition to check on update array elements
     * @return New array with values replaced where condition is satisfied
     */
    public SDVariable replaceWhere(String name, SDVariable update, Number value, Condition condition) {
        SDVariable ret = f().replaceWhere(update, value, condition);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Reshape the input variable to the specified (fixed) shape. The output variable will have the same values as the
     * input, but with the specified shape.<br>
     * Note that prod(shape) must match length(input) == prod(input.shape)
     *
     * @param x     Input variable
     * @param shape New shape for variable
     * @return Output variable
     * @see #reshape(SDVariable, SDVariable)
     */
    public SDVariable reshape(SDVariable x, long... shape) {
        return reshape(null, x, shape);
    }

    /**
     * Reshape the input variable to the specified (fixed) shape. The output variable will have the same values as the
     * input, but with the specified shape.<br>
     * Note that prod(shape) must match length(input) == prod(input.shape)
     *
     * @param name  Output variable name
     * @param x     Input variable
     * @param shape New shape for variable
     * @return Output variable
     * @see #reshape(SDVariable, SDVariable)
     */
    public SDVariable reshape(String name, SDVariable x, long... shape) {
        SDVariable result = f().reshape(x, shape);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Reshape the input variable to the specified (fixed) shape. The output variable will have the same values as the
     * input, but with the specified shape.<br>
     * Note that prod(shape) must match length(input) == prod(input.shape)
     *
     * @param x     Input variable
     * @param shape New shape for variable
     * @return Output variable
     * @see #reshape(SDVariable, SDVariable)
     */
    public SDVariable reshape(SDVariable x, int... shape) {
        return reshape(null, x, shape);
    }

    /**
     * Reshape the input variable to the specified (fixed) shape. The output variable will have the same values as the
     * input, but with the specified shape.<br>
     * Note that prod(shape) must match length(input) == prod(input.shape)
     *
     * @param name  Output variable name
     * @param x     Input variable
     * @param shape New shape for variable
     * @return Output variable
     * @see #reshape(SDVariable, SDVariable)
     */
    public SDVariable reshape(String name, SDVariable x, int... shape) {
        SDVariable result = f().reshape(x, shape);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Reshape the input variable to the specified (dynamic) shape. The output variable will have the same values as the
     * input, but with the specified shape.<br>
     * Note that prod(shape) must match length(input) == prod(input.shape)
     *
     * @param x     Input variable
     * @param shape New shape for variable
     * @return Output variable
     * @see #reshape(SDVariable, int[])
     */
    public SDVariable reshape(SDVariable x, SDVariable shape) {
        return reshape(null, x, shape);
    }

    /**
     * Reshape the input variable to the specified (dynamic) shape. The output variable will have the same values as the
     * input, but with the specified shape.<br>
     * Note that prod(shape) must match length(input) == prod(input.shape)
     *
     * @param name  Output variable name
     * @param x     Input variable
     * @param shape New shape for variable
     * @return Output variable
     * @see #reshape(SDVariable, int[])
     */
    public SDVariable reshape(String name, SDVariable x, SDVariable shape) {
        validateInteger("reshape", "shape", shape);
        SDVariable result = f().reshape(x, shape);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @see #reverse(String, SDVariable, int...)
     */
    public SDVariable reverse(SDVariable x, int... dimensions) {
        return reverse(null, x, dimensions);
    }

    /**
     * Reverse the values of an array for the specified dimensions<br>
     * If input is:<br>
     * [ 1, 2, 3]<br>
     * [ 4, 5, 6]<br>
     * then<br>
     * reverse(in, 0):<br>
     * [3, 2, 1]<br>
     * [6, 5, 4]<br>
     * reverse(in, 0):<br>
     * [4, 5, 6]<br>
     * [1, 2 3]<br>
     *
     * @param x          Input variable
     * @param dimensions Dimensions
     * @return Output variable
     */
    public SDVariable reverse(String name, SDVariable x, int... dimensions) {
        SDVariable ret = f().reverse(x, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #reverseSequence(String, SDVariable, SDVariable, int, int)
     */
    public SDVariable reverseSequence(SDVariable x, SDVariable seq_lengths, int seqDim, int batchDim) {
        return reverseSequence(null, x, seq_lengths, seqDim, batchDim);
    }

    /**
     * Reverse sequence op: for each slice along dimension seqDimension, the first seqLength values are reversed
     *
     * @param name        Name of the output variable
     * @param x           Input variable
     * @param seq_lengths Length of the sequences
     * @param seqDim      Sequence dimension
     * @param batchDim    Batch dimension
     * @return Reversed sequences
     */
    public SDVariable reverseSequence(String name, SDVariable x, SDVariable seq_lengths, int seqDim, int batchDim) {
        SDVariable ret = f().reverseSequence(x, seq_lengths, seqDim, batchDim);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #reverseSequence(String, SDVariable, SDVariable, int, int)
     */
    public SDVariable reverseSequence(SDVariable x, SDVariable seq_lengths) {
        return reverseSequence(null, x, seq_lengths);
    }

    /**
     * @see #reverseSequence(String, SDVariable, SDVariable, int, int)
     */
    public SDVariable reverseSequence(String name, SDVariable x, SDVariable seq_lengths) {
        SDVariable ret = f().reverseSequence(x, seq_lengths);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Element-wise scalar floor modulus operation: out = floorMod(in, value).
     * i.e., returns the remainder after division by 'value'
     *
     * @param in    Input variable
     * @param value Scalar value to compare
     * @return Output variable
     */
    public SDVariable scalarFloorMod(SDVariable in, Number value) {
        return scalarFloorMod(null, in, value);
    }

    /**
     * Element-wise scalar floor modulus operation: out = floorMod(in, value).
     * i.e., returns the remainder after division by 'value'
     *
     * @param name  Name of the output variable
     * @param in    Input variable
     * @param value Scalar value to compare
     * @return Output variable
     */
    public SDVariable scalarFloorMod(String name, SDVariable in, Number value) {
        validateNumerical("floorMod", in);
        SDVariable ret = f().scalarFloorMod(in, value);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Element-wise scalar maximum operation: out = max(in, value)
     *
     * @param in    Input variable
     * @param value Scalar value to compare
     * @return Output variable
     */
    public SDVariable scalarMax(SDVariable in, Number value) {
        return scalarMax(null, in, value);
    }

    /**
     * Element-wise scalar maximum operation: out = max(in, value)
     *
     * @param name  Name of the output variable
     * @param in    Input variable
     * @param value Scalar value to compare
     * @return Output variable
     */
    public SDVariable scalarMax(String name, SDVariable in, Number value) {
        validateNumerical("max", in);
        SDVariable ret = f().scalarMax(in, value);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Element-wise scalar minimum operation: out = min(in, value)
     *
     * @param in    Input variable
     * @param value Scalar value to compare
     * @return Output variable
     */
    public SDVariable scalarMin(SDVariable in, Number value) {
        return scalarMin(null, in, value);
    }

    /**
     * Element-wise scalar minimum operation: out = min(in, value)
     *
     * @param name  Name of the output variable
     * @param in    Input variable
     * @param value Scalar value to compare
     * @return Output variable
     */
    public SDVariable scalarMin(String name, SDVariable in, Number value) {
        validateNumerical("min", in);
        SDVariable ret = f().scalarMin(in, value);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Return an array with equal shape to the input, but all elements set to value 'set'
     *
     * @param in  Input variable
     * @param set Value to set
     * @return Output variable
     */
    public SDVariable scalarSet(SDVariable in, Number set) {
        return scalarSet(null, in, set);
    }

    /**
     * Return a variable with equal shape to the input, but all elements set to value 'set'
     *
     * @param name Name of the output variable
     * @param in   Input variable
     * @param set  Value to set
     * @return Output variable
     */
    public SDVariable scalarSet(String name, SDVariable in, Number set) {
        SDVariable ret = f().scalarSet(in, set);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #scatterAdd(String, SDVariable, SDVariable, SDVariable)
     */
    public SDVariable scatterAdd(SDVariable ref, SDVariable indices, SDVariable updates) {
        return scatterAdd(null, ref, indices, updates);
    }

    /**
     * Scatter addition operation.<br>
     * If indices is rank 0 (a scalar), then out[index, ...] += updates[...]<br>
     * If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] += updates[i, ...]<br>
     * If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] += updates[i, ..., k, ...]<br>
     * Note that if multiple indices refer to the same location, the contributions from each is handled correctly.
     *
     * @param name    Name of the output variable
     * @param ref     Initial/source variable
     * @param indices Indices array
     * @param updates Updates to add to the initial/source array
     * @return The updated variable
     */
    public SDVariable scatterAdd(String name, SDVariable ref, SDVariable indices, SDVariable updates) {
        validateInteger("scatterAdd", "indices", indices);
        SDVariable ret = f().scatterAdd(ref, indices, updates);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #scatterDiv(String, SDVariable, SDVariable, SDVariable)
     */
    public SDVariable scatterDiv(SDVariable ref, SDVariable indices, SDVariable updates) {
        return scatterDiv(null, ref, indices, updates);
    }

    /**
     * Scatter division operation.<br>
     * If indices is rank 0 (a scalar), then out[index, ...] /= updates[...]<br>
     * If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] /= updates[i, ...]<br>
     * If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] /= updates[i, ..., k, ...]<br>
     * Note that if multiple indices refer to the same location, the contributions from each is handled correctly.
     *
     * @param name    Name of the output variable
     * @param ref     Initial/source variable
     * @param indices Indices array
     * @param updates Updates to add to the initial/source array
     * @return The updated variable
     */
    public SDVariable scatterDiv(String name, SDVariable ref, SDVariable indices, SDVariable updates) {
        validateInteger("scatterDiv", "indices", indices);
        SDVariable ret = f().scatterDiv(ref, indices, updates);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #scatterMax(String, SDVariable, SDVariable, SDVariable)
     */
    public SDVariable scatterMax(SDVariable ref, SDVariable indices, SDVariable updates) {
        return scatterMax(null, ref, indices, updates);
    }

    /**
     * Scatter max operation.<br>
     * If indices is rank 0 (a scalar), then out[index, ...] = max(updates[...], in[index,...])<br>
     * If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] = max(updates[i,...], in[indices[i],...])<br>
     * If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] = max(updates[i, ..., k, ...], in[indices[i], ..., indices[k], ...]<br>
     * Note that if multiple indices refer to the same location, the contributions from each is handled correctly.
     *
     * @param name    Name of the output variable
     * @param ref     Initial/source variable
     * @param indices Indices array
     * @param updates Updates to add to the initial/source array
     * @return The updated variable
     */
    public SDVariable scatterMax(String name, SDVariable ref, SDVariable indices, SDVariable updates) {
        validateInteger("scatterMax", "indices", indices);
        SDVariable ret = f().scatterMax(ref, indices, updates);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #scatterMin(String, SDVariable, SDVariable, SDVariable)
     */
    public SDVariable scatterMin(SDVariable ref, SDVariable indices, SDVariable updates) {
        return scatterMin(null, ref, indices, updates);
    }

    /**
     * Scatter min operation.<br>
     * If indices is rank 0 (a scalar), then out[index, ...] = min(updates[...], in[index,...])<br>
     * If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] = min(updates[i,...], in[indices[i],...])<br>
     * If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] = min(updates[i, ..., k, ...], in[indices[i], ..., indices[k], ...]<br>
     * Note that if multiple indices refer to the same location, the contributions from each is handled correctly.
     *
     * @param name    Name of the output variable
     * @param ref     Initial/source variable
     * @param indices Indices array
     * @param updates Updates to add to the initial/source array
     * @return The updated variable
     */
    public SDVariable scatterMin(String name, SDVariable ref, SDVariable indices, SDVariable updates) {
        validateInteger("scatterMin", "indices", indices);
        SDVariable ret = f().scatterMin(ref, indices, updates);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #scatterMul(String, SDVariable, SDVariable, SDVariable)
     */
    public SDVariable scatterMul(SDVariable ref, SDVariable indices, SDVariable updates) {
        return scatterMul(null, ref, indices, updates);
    }

    /**
     * Scatter multiplication operation.<br>
     * If indices is rank 0 (a scalar), then out[index, ...] *= updates[...]<br>
     * If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] *= updates[i, ...]<br>
     * If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] *= updates[i, ..., k, ...]<br>
     * Note that if multiple indices refer to the same location, the contributions from each is handled correctly.
     *
     * @param name    Name of the output variable
     * @param ref     Initial/source variable
     * @param indices Indices array
     * @param updates Updates to add to the initial/source array
     * @return The updated variable
     */
    public SDVariable scatterMul(String name, SDVariable ref, SDVariable indices, SDVariable updates) {
        validateInteger("scatterMul", "indices", indices);
        SDVariable ret = f().scatterMul(ref, indices, updates);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #scatterSub(String, SDVariable, SDVariable, SDVariable)
     */
    public SDVariable scatterSub(SDVariable ref, SDVariable indices, SDVariable updates) {
        return scatterSub(null, ref, indices, updates);
    }

    /**
     * Scatter subtraction operation.<br>
     * If indices is rank 0 (a scalar), then out[index, ...] -= updates[...]<br>
     * If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] -= updates[i, ...]<br>
     * If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] -= updates[i, ..., k, ...]<br>
     * Note that if multiple indices refer to the same location, the contributions from each is handled correctly.
     *
     * @param name    Name of the output variable
     * @param ref     Initial/source variable
     * @param indices Indices array
     * @param updates Updates to add to the initial/source array
     * @return The updated variable
     */
    public SDVariable scatterSub(String name, SDVariable ref, SDVariable indices, SDVariable updates) {
        validateInteger("scatterSub", "indices", indices);
        SDVariable ret = f().scatterSub(ref, indices, updates);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #scatterUpdate(String, SDVariable, SDVariable, SDVariable)
     */
    public SDVariable scatterUpdate(SDVariable ref, SDVariable indices, SDVariable updates) {
        return scatterUpdate(null, ref, indices, updates);
    }

    /**
     * Scatter update operation.<br>
     * If indices is rank 0 (a scalar), then out[index, ...] = updates[...]<br>
     * If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] = updates[i, ...]<br>
     * If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] = updates[i, ..., k, ...]<br>
     * Note that if multiple indices refer to the same location, the output at those locations is undefined - different
     * updates may occur in different orders
     *
     * @param name    Name of the output variable
     * @param ref     Initial/source variable
     * @param indices Indices array
     * @param updates Updates to add to the initial/source array
     * @return The updated variable
     */
    public SDVariable scatterUpdate(String name, SDVariable ref, SDVariable indices, SDVariable updates) {
        validateInteger("scatterUpdate", "indices", indices);
        SDVariable ret = f().scatterUpdate(ref, indices, updates);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #segmentMax(String, SDVariable, SDVariable)
     */
    public SDVariable segmentMax(SDVariable data, SDVariable segmentIds) {
        return segmentMax(null, data, segmentIds);
    }

    /**
     * Segment max operation.<br>
     * If data =     [3, 6, 1, 4, 9, 2, 8]<br>
     * segmentIds =  [0, 0, 1, 1, 1, 2, 2]<br>
     * then output = [6, 9, 8] = [max(3,6), max(1,4,9), max(2,8)]<br>
     * Note that the segment IDs must be sorted from smallest to largest segment.
     * See {@link #unsortedSegmentMax(String, SDVariable, SDVariable, int)}
     * for the same op without this sorted requirement
     *
     * @param name       Name of the output variable. May be null
     * @param data       Data to perform segment max on
     * @param segmentIds Variable for the segment IDs
     * @return Segment max output
     */
    public SDVariable segmentMax(String name, SDVariable data, SDVariable segmentIds) {
        validateNumerical("segmentMax", "data", data);
        validateInteger("segmentMax", "segmentIds", segmentIds);
        SDVariable ret = f().segmentMax(data, segmentIds);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #segmentMean(String, SDVariable, SDVariable)
     */
    public SDVariable segmentMean(SDVariable data, SDVariable segmentIds) {
        return segmentMean(null, data, segmentIds);
    }

    /**
     * Segment mean operation.<br>
     * If data =     [3, 6, 1, 4, 9, 2, 8]<br>
     * segmentIds =  [0, 0, 1, 1, 1, 2, 2]<br>
     * then output = [4.5, 4.666, 5] = [mean(3,6), mean(1,4,9), mean(2,8)]<br>
     * Note that the segment IDs must be sorted from smallest to largest segment.
     * See {@link #unsortedSegmentMean(String, SDVariable, SDVariable, int)} for the same op without this sorted requirement
     *
     * @param name       Name of the output variable. May be null
     * @param data       Data to perform segment max on
     * @param segmentIds Variable for the segment IDs
     * @return Segment mean output
     */
    public SDVariable segmentMean(String name, SDVariable data, SDVariable segmentIds) {
        validateNumerical("segmentMean", "data", data);
        validateInteger("segmentMean", "segmentIds", segmentIds);
        SDVariable ret = f().segmentMean(data, segmentIds);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #segmentMin(String, SDVariable, SDVariable)
     */
    public SDVariable segmentMin(SDVariable data, SDVariable segmentIds) {
        return segmentMin(null, data, segmentIds);
    }

    /**
     * Segment min operation.<br>
     * If data =     [3, 6, 1, 4, 9, 2, 8]<br>
     * segmentIds =  [0, 0, 1, 1, 1, 2, 2]<br>
     * then output = [3, 1, 2] = [min(3,6), min(1,4,9), min(2,8)]<br>
     * Note that the segment IDs must be sorted from smallest to largest segment.
     * See {@link #unsortedSegmentMin(String, SDVariable, SDVariable, int)} for the same op without this sorted requirement
     *
     * @param name       Name of the output variable. May be null
     * @param data       Data to perform segment max on
     * @param segmentIds Variable for the segment IDs
     * @return Segment min output
     */
    public SDVariable segmentMin(String name, SDVariable data, SDVariable segmentIds) {
        validateNumerical("segmentMin", "data", data);
        validateInteger("segmentMin", "segmentIds", segmentIds);
        SDVariable ret = f().segmentMin(data, segmentIds);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #segmentProd(String, SDVariable, SDVariable)
     */
    public SDVariable segmentProd(SDVariable data, SDVariable segmentIds) {
        return segmentProd(null, data, segmentIds);
    }

    /**
     * Segment product operation.<br>
     * If data =     [3, 6, 1, 4, 9, 2, 8]<br>
     * segmentIds =  [0, 0, 1, 1, 1, 2, 2]<br>
     * then output = [18, 36, 16] = [prod(3,6), prod(1,4,9), prod(2,8)]<br>
     * Note that the segment IDs must be sorted from smallest to largest segment.
     * See {@link #unsortedSegmentProd(String, SDVariable, SDVariable, int)} for the same op without this sorted requirement
     *
     * @param name       Name of the output variable. May be null
     * @param data       Data to perform segment max on
     * @param segmentIds Variable for the segment IDs
     * @return Segment product output
     */
    public SDVariable segmentProd(String name, SDVariable data, SDVariable segmentIds) {
        validateNumerical("segmentProd", "data", data);
        validateInteger("segmentProd", "segmentIds", segmentIds);
        SDVariable ret = f().segmentProd(data, segmentIds);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #segmentSum(String, SDVariable, SDVariable)
     */
    public SDVariable segmentSum(SDVariable data, SDVariable segmentIds) {
        return segmentSum(null, data, segmentIds);
    }

    /**
     * Segment sum operation.<br>
     * If data =     [3, 6, 1, 4, 9, 2, 8]<br>
     * segmentIds =  [0, 0, 1, 1, 1, 2, 2]<br>
     * then output = [9, 14, 10] = [sum(3,6), sum(1,4,9), sum(2,8)]<br>
     * Note that the segment IDs must be sorted from smallest to largest segment.
     * See {@link #unsortedSegmentSum(String, SDVariable, SDVariable, int)} for the same op without this sorted requirement
     *
     * @param name       Name of the output variable. May be null
     * @param data       Data to perform segment max on
     * @param segmentIds Variable for the segment IDs
     * @return Segment sum output
     */
    public SDVariable segmentSum(String name, SDVariable data, SDVariable segmentIds) {
        validateNumerical("segmentSum", "data", data);
        validateInteger("segmentSum", "segmentIds", segmentIds);
        SDVariable ret = f().segmentSum(data, segmentIds);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #sequenceMask(String, SDVariable, SDVariable, DataType)
     */
    public SDVariable sequenceMask(SDVariable lengths, int maxLen, DataType dataType) {
        return sequenceMask(null, lengths, maxLen, dataType);
    }

    /**
     * @see #sequenceMask(String, SDVariable, SDVariable, DataType)
     */
    public SDVariable sequenceMask(String name, SDVariable lengths, int maxLen, DataType dataType) {
        validateInteger("sequenceMask", "lengths", lengths);
        SDVariable ret = f().sequenceMask(lengths, maxLen, dataType);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #sequenceMask(String, SDVariable, SDVariable, DataType)
     */
    public SDVariable sequenceMask(String name, SDVariable lengths, DataType dataType) {
        SDVariable ret = f().sequenceMask(lengths, dataType);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #sequenceMask(String, SDVariable, SDVariable, DataType)
     */
    public SDVariable sequenceMask(SDVariable lengths, DataType dataType) {
        return sequenceMask(lengths, null, dataType);
    }

    /**
     * @see #sequenceMask(String, SDVariable, SDVariable, DataType)
     */
    public SDVariable sequenceMask(SDVariable lengths, SDVariable maxLen, DataType dataType) {
        return sequenceMask(null, lengths, maxLen, dataType);
    }

    /**
     * Generate a sequence mask (with values 0 or 1) based on the specified lengths<br>
     * Specifically, out[i, ..., k, j] = (j < lengths[i, ..., k] ? 1.0 : 0.0)
     *
     * @param name    Name of the output variable
     * @param lengths Lengths of the sequences
     * @param maxLen  Maximum sequence length
     * @return Output variable
     */
    public SDVariable sequenceMask(String name, SDVariable lengths, SDVariable maxLen, DataType dataType) {
        validateInteger("sequenceMask", "lengths", lengths);
        SDVariable ret = f().sequenceMask(lengths, maxLen, dataType);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Returns the shape of the specified SDVariable as a 1D SDVariable
     *
     * @param input Input variable
     * @return 1D output variable with contents equal to the shape of the input
     */
    public SDVariable shape(SDVariable input) {
        return shape(null, input);
    }

    /**
     * Returns the shape of the specified SDVariable as a 1D SDVariable
     *
     * @param name  Name of the output variable
     * @param input Input variable
     * @return 1D output variable with contents equal to the shape of the input
     */
    public SDVariable shape(String name, SDVariable input) {
        SDVariable ret = f().shape(input);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Returns the size (number of elements, i.e., prod(shape)) of the specified SDVariable as a 0D scalar variable
     *
     * @param in Input variable
     * @return 0D (scalar) output variable with value equal to the number of elements in the specified array
     */
    public SDVariable size(SDVariable in) {
        return size(null, in);
    }

    /**
     * Returns the size (number of elements, i.e., prod(shape)) of the specified SDVariable as a 0D scalar variable
     *
     * @param name Name of the output variable
     * @param in   Input variable
     * @return 0D (scalar) output variable with value equal to the number of elements in the specified array
     */
    public SDVariable size(String name, SDVariable in) {
        SDVariable ret = f().size(in);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #sizeAt(String, SDVariable, int)
     */
    public SDVariable sizeAt(SDVariable in, int dimension) {
        return sizeAt(null, in, dimension);
    }

    /**
     * Returns a rank 0 (scalar) variable for the size of the specified dimension.
     * For example, if X has shape [10,20,30] then sizeAt(X,1)=20. Similarly, sizeAt(X,-1)=30
     *
     * @param name      Name of the output variable
     * @param in        Input variable
     * @param dimension Dimension to get size of
     * @return Scalar SDVariable for size at specified variable
     */
    public SDVariable sizeAt(String name, SDVariable in, int dimension) {
        SDVariable ret = f().sizeAt(in, dimension);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #slice(String, SDVariable, int[], int[])
     */
    public SDVariable slice(SDVariable input, int[] begin, int[] size) {
        return slice(null, input, begin, size);
    }

    public SDVariable slice(SDVariable input, SDVariable begin, SDVariable size) {
        return slice(null, input, begin, size);
    }

    /**
     * Get a subset of the specified input, by specifying the first element and the size of the array.<br>
     * For example, if input is:<br>
     * [a, b, c]<br>
     * [d, e, f]<br>
     * then slice(input, begin=[0,1], size=[2,1] will return:<br>
     * [b]<br>
     * [e]<br>
     * <br>
     * Note that for each dimension i, begin[i] + size[i] <= input.size(i)
     *
     * @param name  Output variable name
     * @param input Variable to get subset of
     * @param begin Beginning index. Must be same length as rank of input array
     * @param size  Size of the output array. Must be same length as rank of input array
     * @return Subset of the input
     */
    public SDVariable slice(String name, SDVariable input, int[] begin, int[] size) {
        SDVariable ret = f().slice(input, begin, size);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable slice(String name, SDVariable input, @NonNull SDVariable begin, @NonNull SDVariable size) {
        SDVariable ret = f().slice(input, begin, size);
        return updateVariableNameAndReference(ret, name);
    }



    /**
     * Squared L2 norm: see {@link #norm2(String, SDVariable, int...)}
     */
    public SDVariable squaredNorm(SDVariable x, int... dimensions) {
        return squaredNorm(null, x, false, dimensions);
    }

    /**
     * Squared L2 norm: see {@link #norm2(String, SDVariable, boolean, int...)}
     */
    public SDVariable squaredNorm(String name, SDVariable x, boolean keepDims, int... dimensions) {
        validateNumerical("squaredNorm", x);
        SDVariable result = f().squaredNorm(x, keepDims, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Squared L2 norm: see {@link #norm2(String, SDVariable, int...)}
     */
    public SDVariable squaredNorm(String name, SDVariable x, int... dimensions) {
        return squaredNorm(name, x, false, dimensions);
    }

    /**
     * Squared L2 norm: see {@link #norm2(String, SDVariable, boolean, int...)}
     */
    public SDVariable squaredNorm(SDVariable x, boolean keepDims, int... dimensions) {
        return squaredNorm(null, x, keepDims, dimensions);
    }

    /**
     * @see #squeeze(String, SDVariable, int)
     */
    public SDVariable squeeze(SDVariable x, int axis) {
        return squeeze(null, x, axis);
    }

    /**
     * Remove a single dimension of size 1.
     * For example, if input has shape [a,b,1,c] then squeeze(input, 2) returns an array of shape [a,b,c]
     *
     * @param name Name of the output variable
     * @param x    Input variable
     * @param axis Size 1 dimension to remove
     * @return Output variable
     */
    public SDVariable squeeze(String name, SDVariable x, int axis) {
        SDVariable result = f().squeeze(x, axis);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @see #stack(String, int, SDVariable...)
     */
    public SDVariable stack(int axis, SDVariable... values) {
        return stack(null, axis, values);
    }

    /**
     * Stack a set of N SDVariables of rank X into one rank X+1 variable.
     * If inputs have shape [a,b,c] then output has shape:<br>
     * axis = 0: [N,a,b,c]<br>
     * axis = 1: [a,N,b,c]<br>
     * axis = 2: [a,b,N,c]<br>
     * axis = 3: [a,b,c,N]<br>
     *
     * @param name   Name of the output variable
     * @param axis   Axis to stack on
     * @param values Input variables to stack. Must have the same shape for all inputs
     * @return Output variable
     * @see #unstack(String[], SDVariable, int, int)
     */
    public SDVariable stack(String name, int axis, SDVariable... values) {
        validateSameType("stack", false, values);
        SDVariable ret = f().stack(values, axis);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #standardDeviation(String, SDVariable, boolean, int...)
     */
    public SDVariable standardDeviation(SDVariable x, boolean biasCorrected, int... dimensions) {
        return standardDeviation(null, x, biasCorrected, dimensions);
    }

    /**
     * Stardard deviation array reduction operation, optionally along specified dimensions
     *
     * @param name          Output variable name
     * @param x             Input variable
     * @param biasCorrected If true: divide by (N-1) (i.e., sample stdev). If false: divide by N (population stdev)
     * @param dimensions    Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Output variable: reduced array of rank (input rank - num dimensions)
     */
    public SDVariable standardDeviation(String name, SDVariable x, boolean biasCorrected, int... dimensions) {
        return standardDeviation(name, x, biasCorrected, false, dimensions);
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
     * @param x             Input variable
     * @param biasCorrected If true: divide by (N-1) (i.e., sample stdev). If false: divide by N (population stdev)
     * @param keepDims      If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions    Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Output variable: reduced array of rank (input rank - num dimensions)
     */
    public SDVariable standardDeviation(String name, SDVariable x, boolean biasCorrected, boolean keepDims, int... dimensions) {
        validateNumerical("standard deviation", x);
        SDVariable result = f().std(x, biasCorrected, keepDims, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @see #stridedSlice(String, SDVariable, long[], long[], long[])
     */
    public SDVariable stridedSlice(SDVariable input, int[] begin, int[] end, int[] strides) {
        return stridedSlice(null, input, begin, end, strides);
    }

    /**
     * @see #stridedSlice(String, SDVariable, long[], long[], long[])
     */
    public SDVariable stridedSlice(String name, SDVariable input, int[] begin, int[] end, int[] strides) {
        return stridedSlice(name, input, begin, end, strides, 0, 0, 0, 0, 0);
    }

    /**
     * @see #stridedSlice(String, SDVariable, long[], long[], long[], int, int, int, int, int)
     */
    public SDVariable stridedSlice(String name, SDVariable in, int[] begin, int[] end, int[] strides, int beginMask,
                                   int endMask, int ellipsisMask, int newAxisMask, int shrinkAxisMask) {
        SDVariable ret = f().stridedSlice(in, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #stridedSlice(String, SDVariable, long[], long[], long[])
     */
    public SDVariable stridedSlice(SDVariable input, long[] begin, long[] end, long[] strides) {
        return stridedSlice(null, input, begin, end, strides);
    }

    /**
     * Get a subset of the specified input, by specifying the first element, last element, and the strides.<br>
     * For example, if input is:<br>
     * [a, b, c]<br>
     * [d, e, f]<br>
     * [g, h, i]<br>
     * then stridedSlice(input, begin=[0,1], end=[2,2], strides=[2,1]) will return:<br>
     * [b, c]<br>
     * [h, i]<br>
     * <br>
     *
     * @param name    Output variable name
     * @param input   Variable to get subset of
     * @param begin   Beginning index. Must be same length as rank of input array
     * @param end     End index. Must be same length as the rank of the array
     * @param strides Stride ("step size") for each dimension. Must be same length as the rank of the array. For example,
     *                stride of 2 means take every second element.
     * @return Subset of the input
     */
    public SDVariable stridedSlice(String name, SDVariable input, long[] begin, long[] end, long[] strides) {
        return stridedSlice(name, input, begin, end, strides, 0, 0, 0, 0, 0);
    }

    /**
     * Get a subset of the specified input, by specifying the first element, last element, and the strides.<br>
     * Operates as described in {@link #stridedSlice(SDVariable, long[], long[], long[])} with some extra mask arrays
     * as described below.
     *
     * @param name           Output variable name
     * @param in             Variable to get subset of
     * @param begin          Beginning index
     * @param end            End index
     * @param strides        Stride ("step size") for each dimension. For example,
     *                       stride of 2 means take every second element.
     * @param beginMask      Bit mask: If the ith bit is set to 1, then the value in the begin long[] is ignored,
     *                       and a value of 0 is used instead for the beginning index for that dimension
     * @param endMask        Bit mask: If the ith bit is set to 1, then the value in the end long[] is ignored,
     *                       and a value of size(i)-1 is used instead for the end index for that dimension
     * @param ellipsisMask   Bit mask: only one non-zero value is allowed here. If a non-zero value is set, then other
     *                       dimensions are inserted as required at the specified position
     * @param newAxisMask    Bit mask: if the ith bit is set to 1, then the begin/end/stride values are ignored, and
     *                       a size 1 dimension is inserted at this point
     * @param shrinkAxisMask Bit mask: if the ith bit is set to 1, then the begin/end/stride values are ignored, and
     *                       a size 1 dimension is removed at this point. Note that begin/end/stride values must
     *                       result in a size 1 output for these dimensions
     * @return A subset of the input array
     */
    public SDVariable stridedSlice(String name, SDVariable in, long[] begin, long[] end, long[] strides, int beginMask,
                                   int endMask, int ellipsisMask, int newAxisMask, int shrinkAxisMask) {
        SDVariable ret = f().stridedSlice(in, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #stridedSlice(String, SDVariable, long[], long[], long[], int, int, int, int, int)
     */
    public SDVariable stridedSlice(SDVariable in, int[] begin, int[] end, int[] strides, int beginMask,
                                   int endMask, int ellipsisMask, int newAxisMask, int shrinkAxisMask) {
        return stridedSlice(null, in, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask);
    }

    /**
     * @see #stridedSlice(String, SDVariable, long[], long[], long[], int, int, int, int, int)
     */
    public SDVariable stridedSlice(SDVariable in, long[] begin, long[] end, long[] strides, int beginMask,
                                   int endMask, int ellipsisMask, int newAxisMask, int shrinkAxisMask) {
        return stridedSlice(null, in, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask);
    }

    /**
     * Sum array reduction operation, optionally along specified dimensions
     *
     * @param x          Input variable
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Output variable: reduced array of rank (input rank - num dimensions)
     */
    public SDVariable sum(SDVariable x, int... dimensions) {
        return sum(null, x, dimensions);
    }

    /**
     * Sum array reduction operation, optionally along specified dimensions
     *
     * @param x          Input variable
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Output variable: reduced array of rank (input rank - num dimensions) if keepDims = false, or
     * of rank (input rank) if keepdims = true
     */
    public SDVariable sum(String name, SDVariable x, int... dimensions) {
        return sum(name, x, false, dimensions);
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
     * @param x          Input variable
     * @param keepDims   If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Output variable: reduced array of rank (input rank - num dimensions) if keepDims = false, or
     * of rank (input rank) if keepdims = true
     */
    public SDVariable sum(String name, SDVariable x, boolean keepDims, int... dimensions) {
        validateNumerical("sum reduction", x);
        SDVariable result = f().sum(x, keepDims, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @see #sum(String, SDVariable, boolean, int...)
     */
    public SDVariable sum(SDVariable x, boolean keepDims, int... dimensions) {
        return sum(null, x, keepDims, dimensions);
    }

    /**
     * @param x
     * @param y
     * @param dimensions
     * @return
     */
    public SDVariable tensorMmul(SDVariable x,
                                 SDVariable y,
                                 int[][] dimensions) {
        return tensorMmul(null, x, y, dimensions);
    }

    /**
     * @param x          Input variable x
     * @param y          Input variable y
     * @param dimensions dimensions
     * @return Output variable
     */
    public SDVariable tensorMmul(String name,
                                 SDVariable x,
                                 SDVariable y,
                                 int[][] dimensions) {
        validateNumerical("tensorMmul", x, y);
        SDVariable result = f().tensorMmul(x, y, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @see #tile(String, SDVariable, int...)
     */
    public SDVariable tile(SDVariable x, int... repeat) {
        return tile(null, x, repeat);
    }

    /**
     * Repeat (tile) the input tensor the specified number of times.<br>
     * For example, if input is<br>
     * [1, 2]<br>
     * [3, 4]<br>
     * and repeat is [2, 3]<br>
     * then output is<br>
     * [1, 2, 1, 2, 1, 2]<br>
     * [3, 4, 3, 4, 3, 4]<br>
     * [1, 2, 1, 2, 1, 2]<br>
     * [3, 4, 3, 4, 3, 4]<br>
     * <br>
     *
     * @param name   Output variable name
     * @param x      Input variable
     * @param repeat Number of times to repeat in each axis. Must have length equal to the rank of the input array
     * @return Output variable
     */
    public SDVariable tile(String name, SDVariable x, int... repeat) {
        SDVariable result = f().tile(x, repeat);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @see #tile(String, SDVariable, int...)
     */
    public SDVariable tile(SDVariable x, SDVariable repeat) {
        return tile(null, x, repeat);
    }

    /**
     * @see #tile(String, SDVariable, int...)
     */
    public SDVariable tile(String name, SDVariable x, SDVariable repeat) {
        SDVariable result = f().tile(x, repeat);
        return updateVariableNameAndReference(result, name);
    }
    /**
     * Matrix transpose operation: If input has shape [a,b] output has shape [b,a]
     *
     * @param x Input variable
     * @return Output variable (transposed input)
     */
    public SDVariable transpose(SDVariable x) {
        return transpose(null, x);
    }

    /**
     * Matrix transpose operation: If input has shape [a,b] output has shape [b,a]
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable (transposed input)
     */
    public SDVariable transpose(String name, SDVariable x) {
        SDVariable result = f().transpose(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * See {@link #unsortedSegmentMax(String, SDVariable, SDVariable, int)}
     */
    public SDVariable unsortedSegmentMax(SDVariable data, SDVariable segmentIds, int numSegments) {
        return unsortedSegmentMax(null, data, segmentIds, numSegments);
    }

    /**
     * Unsorted segment max operation. As per {@link #segmentMax(String, SDVariable, SDVariable)} but without
     * the requirement for the indices to be sorted.<br>
     * If data =     [1, 3, 2, 6, 4, 9, 8]<br>
     * segmentIds =  [1, 0, 2, 0, 1, 1, 2]<br>
     * then output = [6, 9, 8] = [max(3,6), max(1,4,9), max(2,8)]<br>
     *
     * @param name        Name of the output variable
     * @param data        Data (variable) to perform unsorted segment max on
     * @param segmentIds  Variable for the segment IDs
     * @param numSegments Number of segments
     * @return Unsorted segment max output
     */
    public SDVariable unsortedSegmentMax(String name, SDVariable data, SDVariable segmentIds, int numSegments) {
        validateNumerical("unsortedSegmentMax", "data", data);
        validateInteger("unsortedSegmentMax", "segmentIds", segmentIds);
        SDVariable ret = f().unsortedSegmentMax(data, segmentIds, numSegments);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * See {@link #unsortedSegmentMean(String, SDVariable, SDVariable, int)}
     */
    public SDVariable unsortedSegmentMean(SDVariable data, SDVariable segmentIds, int numSegments) {
        return unsortedSegmentMean(null, data, segmentIds, numSegments);
    }

    /**
     * Unsorted segment mean operation. As per {@link #segmentMean(String, SDVariable, SDVariable)} but without
     * the requirement for the indices to be sorted.<br>
     * If data =     [1, 3, 2, 6, 4, 9, 8]<br>
     * segmentIds =  [1, 0, 2, 0, 1, 1, 2]<br>
     * then output = [4.5, 4.666, 5] = [mean(3,6), mean(1,4,9), mean(2,8)]<br>
     *
     * @param name        Name of the output variable
     * @param data        Data (variable) to perform unsorted segment mean on
     * @param segmentIds  Variable for the segment IDs
     * @param numSegments Number of segments
     * @return Unsorted segment mean output
     */
    public SDVariable unsortedSegmentMean(String name, SDVariable data, SDVariable segmentIds, int numSegments) {
        validateNumerical("unsortedSegmentMean", "data", data);
        validateInteger("unsortedSegmentMean", "segmentIds", segmentIds);
        SDVariable ret = f().unsortedSegmentMean(data, segmentIds, numSegments);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * See {@link #unsortedSegmentMin(String, SDVariable, SDVariable, int)}
     */
    public SDVariable unsortedSegmentMin(SDVariable data, SDVariable segmentIds, int numSegments) {
        return unsortedSegmentMin(null, data, segmentIds, numSegments);
    }

    /**
     * Unsorted segment min operation. As per {@link #segmentMin(String, SDVariable, SDVariable)} but without
     * the requirement for the indices to be sorted.<br>
     * If data =     [1, 3, 2, 6, 4, 9, 8]<br>
     * segmentIds =  [1, 0, 2, 0, 1, 1, 2]<br>
     * then output = [3, 1, 2] = [min(3,6), min(1,4,9), min(2,8)]<br>
     *
     * @param name        Name of the output variable
     * @param data        Data (variable) to perform unsorted segment min on
     * @param segmentIds  Variable for the segment IDs
     * @param numSegments Number of segments
     * @return Unsorted segment min output
     */
    public SDVariable unsortedSegmentMin(String name, SDVariable data, SDVariable segmentIds, int numSegments) {
        validateNumerical("unsortedSegmentMin", "data", data);
        validateInteger("unsortedSegmentMin", "segmentIds", segmentIds);
        SDVariable ret = f().unsortedSegmentMin(data, segmentIds, numSegments);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * See {@link #unsortedSegmentProd(String, SDVariable, SDVariable, int)}
     */
    public SDVariable unsortedSegmentProd(SDVariable data, SDVariable segmentIds, int numSegments) {
        return unsortedSegmentProd(null, data, segmentIds, numSegments);
    }

    /**
     * Unsorted segment product operation. As per {@link #segmentProd(String, SDVariable, SDVariable)} but without
     * the requirement for the indices to be sorted.<br>
     * If data =     [1, 3, 2, 6, 4, 9, 8]<br>
     * segmentIds =  [1, 0, 2, 0, 1, 1, 2]<br>
     * then output = [4.5, 4.666, 5] = [mean(3,6), mean(1,4,9), mean(2,8)]<br>
     *
     * @param name       Name of the output variable
     * @param data       Data (variable) to perform unsorted segment product on
     * @param segmentIds Variable for the segment IDs
     * @return Unsorted segment product output
     */
    public SDVariable unsortedSegmentProd(String name, SDVariable data, SDVariable segmentIds, int numSegments) {
        validateNumerical("unsortedSegmentProd", "data", data);
        validateInteger("unsortedSegmentProd", "segmentIds", segmentIds);
        SDVariable ret = f().unsortedSegmentProd(data, segmentIds, numSegments);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * See {@link #unsortedSegmentSqrtN(String, SDVariable, SDVariable, int)}
     */
    public SDVariable unsortedSegmentSqrtN(SDVariable data, SDVariable segmentIds, int numSegments) {
        return unsortedSegmentSqrtN(null, data, segmentIds, numSegments);
    }

    /**
     * Unsorted segment sqrtN operation. Simply returns the sqrt of the count of the number of values in each segment<br>
     * If data =     [1, 3, 2, 6, 4, 9, 8]<br>
     * segmentIds =  [1, 0, 2, 0, 1, 1, 2]<br>
     * then output = [1.414, 1.732, 1.414] = [sqrt(2), sqrtN(3), sqrtN(2)]<br>
     *
     * @param name       Name of the output variable
     * @param data       Data (variable) to perform unsorted segment sqrtN on
     * @param segmentIds Variable for the segment IDs
     * @return Unsorted segment sqrtN output
     */
    public SDVariable unsortedSegmentSqrtN(String name, SDVariable data, SDVariable segmentIds, int numSegments) {
        SDVariable ret = f().unsortedSegmentSqrtN(data, segmentIds, numSegments);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * See {@link #unsortedSegmentSum(String, SDVariable, SDVariable, int)}
     */
    public SDVariable unsortedSegmentSum(@NonNull SDVariable data, @NonNull SDVariable segmentIds, int numSegments) {
        return unsortedSegmentSum(null, data, segmentIds, numSegments);
    }

    /**
     * Unsorted segment sum operation. As per {@link #segmentSum(String, SDVariable, SDVariable)} but without
     * the requirement for the indices to be sorted.<br>
     * If data =     [1, 3, 2, 6, 4, 9, 8]<br>
     * segmentIds =  [1, 0, 2, 0, 1, 1, 2]<br>
     * then output = [9, 14, 10] = [sum(3,6), sum(1,4,9), sum(2,8)]<br>
     *
     * @param name        Name of the output variable
     * @param data        Data (variable) to perform unsorted segment sum on
     * @param segmentIds  Variable for the segment IDs
     * @param numSegments Number of segments
     * @return Unsorted segment sum output
     */
    public SDVariable unsortedSegmentSum(String name, @NonNull SDVariable data, @NonNull SDVariable segmentIds, int numSegments) {
        validateNumerical("unsortedSegmentSum", "data", data);
        validateInteger("unsortedSegmentSum", "segmentIds", segmentIds);
        SDVariable ret = f().unsortedSegmentSum(data, segmentIds, numSegments);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #unstack(String[], SDVariable, int, int)
     */
    public SDVariable[] unstack(SDVariable value, int axis) {
        return unstack(null, value, axis);
    }

    /**
     * @see #unstack(String[], SDVariable, int, int)
     */
    public SDVariable[] unstack(String[] names, @NonNull SDVariable value, int axis) {
        SDVariable[] ret = f().unstack(value, axis);
        return updateVariableNamesAndReferences(ret, names);
    }

    /**
     * @see #unstack(String[], SDVariable, int, int)
     */
    public SDVariable[] unstack(@NonNull SDVariable value, int axis, int num) {
        return unstack(null, value, axis, num);
    }

    /**
     * Unstack a variable of rank X into N rank X-1 variables by taking slices along the specified axis.
     * If input has shape [a,b,c] then output has shape:
     * axis = 0: [b,c]<br>
     * axis = 1: [a,c]<br>
     * axis = 2: [a,b]<br>
     *
     * @param names Output variable names. May be null
     * @param value Input variable to unstack
     * @param axis  Axis to unstack on
     * @param num   Number of output variables
     * @return Output variables
     * @see #stack(String, int, SDVariable...)
     */
    public SDVariable[] unstack(String[] names, @NonNull SDVariable value, int axis, int num) {
        SDVariable[] ret = f().unstack(value, axis, num);
        return updateVariableNamesAndReferences(ret, names);
    }

    /**
     * @see #variance(String, SDVariable, boolean, int...)
     */
    public SDVariable variance(@NonNull SDVariable x, boolean biasCorrected, int... dimensions) {
        return variance(null, x, biasCorrected, dimensions);
    }

    /**
     * Variance array reduction operation, optionally along specified dimensions
     *
     * @param name          Output variable name
     * @param x             Input variable
     * @param biasCorrected If true: divide by (N-1) (i.e., sample variable). If false: divide by N (population variance)
     * @param dimensions    Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Output variable: reduced array of rank (input rank - num dimensions)
     */
    public SDVariable variance(String name, @NonNull SDVariable x, boolean biasCorrected, int... dimensions) {
        return variance(name, x, biasCorrected, false, dimensions);
    }

    /**
     * Variance array reduction operation, optionally along specified dimensions<br>
     * Note that if keepDims = true, the output variable has the same rank as the input variable,
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]
     *
     * @param name          Output variable name
     * @param x             Input variable
     * @param biasCorrected If true: divide by (N-1) (i.e., sample variable). If false: divide by N (population variance)
     * @param keepDims      If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
     * @param dimensions    Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Output variable: reduced array of rank (input rank - num dimensions)
     */
    public SDVariable variance(String name, @NonNull SDVariable x, boolean biasCorrected, boolean keepDims, int... dimensions) {
        validateNumerical("variance", x);
        SDVariable result = f().variance(x, biasCorrected, keepDims, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Return a variable of all 0s, with the same shape as the input variable. Note that this is dynamic:
     * if the input shape changes in later execution, the returned variable's shape will also be updated
     *
     * @param input Input SDVariable
     * @return A new SDVariable with the same (dynamic) shape as the input
     */
    public SDVariable zerosLike(@NonNull SDVariable input) {
        return zerosLike(null, input);
    }

    /**
     * Return a variable of all 0s, with the same shape as the input variable. Note that this is dynamic:
     * if the input shape changes in later execution, the returned variable's shape will also be updated
     *
     * @param name  Name of the new SDVariable
     * @param input Input SDVariable
     * @return A new SDVariable with the same (dynamic) shape as the input
     */
    public SDVariable zerosLike(String name, @NonNull SDVariable input) {
        SDVariable ret = f().zerosLike(name, input);
        return updateVariableNameAndReference(ret, name);
    }
}

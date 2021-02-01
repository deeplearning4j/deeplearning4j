/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//================== GENERATED CODE - DO NOT MODIFY THIS FILE ==================

package org.nd4j.linalg.factory.ops;

import static org.nd4j.linalg.factory.NDValidation.isSameType;

import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDValidation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Condition;

public class NDBase {
  public NDBase() {
  }

  /**
   * Boolean and array reduction operation, optionally along specified dimensions<br>
   *
   * @param x Input variable (NDARRAY type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output reduced array of rank (input rank - num dimensions) (BOOL type)
   */
  public INDArray all(INDArray x, int... dimensions) {
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.bool.All(x, dimensions));
  }

  /**
   * Boolean or array reduction operation, optionally along specified dimensions<br>
   *
   * @param x  Input variable (NDARRAY type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output reduced array of rank (input rank - num dimensions) (BOOL type)
   */
  public INDArray any(INDArray x, int... dimensions) {
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.bool.Any(x, dimensions));
  }

  /**
   * Argmax array reduction operation, optionally along specified dimensions.<br>
   * Output values are the index of the maximum value of each slice along the specified dimension.<br>
   *
   * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
   * the mean along a dimension).<br>
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
   * keepDims = true: [a,1,c]<br>
   * keepDims = false: [a,c]<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output reduced array of rank (input rank - num dimensions) if keepDims = false, or
   *  of rank (input rank) if keepdims = true (NUMERIC type)
   */
  public INDArray argmax(INDArray in, boolean keepDims, int... dimensions) {
    NDValidation.validateNumerical("argmax", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.indexaccum.custom.ArgMax(in, keepDims, dimensions))[0];
  }

  /**
   * Argmax array reduction operation, optionally along specified dimensions.<br>
   * Output values are the index of the maximum value of each slice along the specified dimension.<br>
   *
   * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
   * the mean along a dimension).<br>
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
   * keepDims = true: [a,1,c]<br>
   * keepDims = false: [a,c]<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output reduced array of rank (input rank - num dimensions) if keepDims = false, or
   *  of rank (input rank) if keepdims = true (NUMERIC type)
   */
  public INDArray argmax(INDArray in, int... dimensions) {
    NDValidation.validateNumerical("argmax", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.indexaccum.custom.ArgMax(in, false, dimensions))[0];
  }

  /**
   * Argmin array reduction operation, optionally along specified dimensions.<br>
   * Output values are the index of the minimum value of each slice along the specified dimension.<br>
   *
   * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
   * the mean along a dimension).<br>
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
   * keepDims = true: [a,1,c]<br>
   * keepDims = false: [a,c]<br>
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]<br>
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output reduced array of rank (input rank - num dimensions) if keepDims = false, or of rank (input rank) if keepdims = true (NUMERIC type)
   */
  public INDArray argmin(INDArray in, boolean keepDims, int... dimensions) {
    NDValidation.validateNumerical("argmin", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.indexaccum.custom.ArgMin(in, keepDims, dimensions))[0];
  }

  /**
   * Argmin array reduction operation, optionally along specified dimensions.<br>
   * Output values are the index of the minimum value of each slice along the specified dimension.<br>
   *
   * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
   * the mean along a dimension).<br>
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
   * keepDims = true: [a,1,c]<br>
   * keepDims = false: [a,c]<br>
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]<br>
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output reduced array of rank (input rank - num dimensions) if keepDims = false, or of rank (input rank) if keepdims = true (NUMERIC type)
   */
  public INDArray argmin(INDArray in, int... dimensions) {
    NDValidation.validateNumerical("argmin", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.indexaccum.custom.ArgMin(in, false, dimensions))[0];
  }

  /**
   * Matrix multiply a batch of matrices. matricesA and matricesB have to be arrays of same<br>
   * length and each pair taken from these sets has to have dimensions (M, N) and (N, K),<br>
   * respectively. If transposeA is true, matrices from matricesA will have shape (N, M) instead.<br>
   * Likewise, if transposeB is true, matrices from matricesB will have shape (K, N).<br>
   * <br>
   * The result of this operation will be a batch of multiplied matrices. The<br>
   * result has the same length as both input batches and each output matrix is of shape (M, K).<br>
   *
   * @param inputsA First array of input matrices, all of shape (M, N) or (N, M) (NUMERIC type)
   * @param inputsB  Second array of input matrices, all of shape (N, K) or (K, N) (NUMERIC type)
   * @param transposeA Whether to transpose A arrays or not
   * @param transposeB Whether to transpose B arrays or not
   */
  public INDArray[] batchMmul(INDArray[] inputsA, INDArray[] inputsB, boolean transposeA,
      boolean transposeB) {
    NDValidation.validateNumerical("batchMmul", "inputsA", inputsA);
    Preconditions.checkArgument(inputsA.length >= 1, "inputsA has incorrect size/length. Expected: inputsA.length >= 1, got %s", inputsA.length);
    NDValidation.validateNumerical("batchMmul", "inputsB", inputsB);
    Preconditions.checkArgument(inputsB.length >= 1, "inputsB has incorrect size/length. Expected: inputsB.length >= 1, got %s", inputsB.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.custom.BatchMmul(inputsA, inputsB, transposeA, transposeB));
  }

  /**
   * Matrix multiply a batch of matrices. matricesA and matricesB have to be arrays of same<br>
   * length and each pair taken from these sets has to have dimensions (M, N) and (N, K),<br>
   * respectively. If transposeA is true, matrices from matricesA will have shape (N, M) instead.<br>
   * Likewise, if transposeB is true, matrices from matricesB will have shape (K, N).<br>
   * <br>
   * The result of this operation will be a batch of multiplied matrices. The<br>
   * result has the same length as both input batches and each output matrix is of shape (M, K).<br>
   *
   * @param inputsA First array of input matrices, all of shape (M, N) or (N, M) (NUMERIC type)
   * @param inputsB  Second array of input matrices, all of shape (N, K) or (K, N) (NUMERIC type)
   */
  public INDArray[] batchMmul(INDArray[] inputsA, INDArray... inputsB) {
    NDValidation.validateNumerical("batchMmul", "inputsA", inputsA);
    Preconditions.checkArgument(inputsA.length >= 1, "inputsA has incorrect size/length. Expected: inputsA.length >= 1, got %s", inputsA.length);
    NDValidation.validateNumerical("batchMmul", "inputsB", inputsB);
    Preconditions.checkArgument(inputsB.length >= 1, "inputsB has incorrect size/length. Expected: inputsB.length >= 1, got %s", inputsB.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.custom.BatchMmul(inputsA, inputsB, false, false));
  }

  /**
   * Cast the array to a new datatype - for example, Integer -> Float<br>
   *
   * @param arg Input variable to cast (NDARRAY type)
   * @param datatype Datatype to cast to
   * @return output Output array (after casting) (NDARRAY type)
   */
  public INDArray castTo(INDArray arg, DataType datatype) {
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.dtype.Cast(arg, datatype))[0];
  }

  /**
   * Concatenate a set of inputs along the specified dimension.<br>
   * Note that inputs must have identical rank and identical dimensions, other than the dimension to stack on.<br>
   * For example, if 2 inputs have shape [a, x, c] and [a, y, c] and dimension = 1, then the output has shape [a, x+y, c]<br>
   *
   * Inputs must satisfy the following constraints: <br>
   * Input arrays must all be the same datatype: isSameType(inputs)<br>
   *
   * @param inputs Input variables (NUMERIC type)
   * @param dimension Dimension to concatenate on
   * @return output  (NUMERIC type)
   */
  public INDArray concat(int dimension, INDArray... inputs) {
    NDValidation.validateNumerical("concat", "inputs", inputs);
    Preconditions.checkArgument(inputs.length >= 1, "inputs has incorrect size/length. Expected: inputs.length >= 1, got %s", inputs.length);
    Preconditions.checkArgument(isSameType(inputs), "Input arrays must all be the same datatype");
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.Concat(inputs, dimension))[0];
  }

  /**
   * Cumulative product operation.<br>
   * For input: [ a, b, c], output is:<br>
   * exclusive=false, reverse=false: [a, a*b, a*b*c]<br>
   * exclusive=true, reverse=false, [0, a, a*b]<br>
   * exclusive=false, reverse=true: [a*b*c, b*c, c]<br>
   * exclusive=true, reverse=true: [b*c, c, 0]<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param exclusive If true: exclude the first value
   * @param reverse If true: reverse the direction of the accumulation
   * @param axis Scalar axis argument for dimension to perform cumululative sum operations along (Size: AtLeast(min=1))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray cumprod(INDArray in, boolean exclusive, boolean reverse, int... axis) {
    NDValidation.validateNumerical("cumprod", "in", in);
    Preconditions.checkArgument(axis.length >= 1, "axis has incorrect size/length. Expected: axis.length >= 1, got %s", axis.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.CumProd(in, exclusive, reverse, axis))[0];
  }

  /**
   * Cumulative product operation.<br>
   * For input: [ a, b, c], output is:<br>
   * exclusive=false, reverse=false: [a, a*b, a*b*c]<br>
   * exclusive=true, reverse=false, [0, a, a*b]<br>
   * exclusive=false, reverse=true: [a*b*c, b*c, c]<br>
   * exclusive=true, reverse=true: [b*c, c, 0]<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param axis Scalar axis argument for dimension to perform cumululative sum operations along (Size: AtLeast(min=1))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray cumprod(INDArray in, int... axis) {
    NDValidation.validateNumerical("cumprod", "in", in);
    Preconditions.checkArgument(axis.length >= 1, "axis has incorrect size/length. Expected: axis.length >= 1, got %s", axis.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.CumProd(in, false, false, axis))[0];
  }

  /**
   * Cumulative sum operation.<br>
   * For input: [ a, b, c], output is:<br>
   * exclusive=false, reverse=false: [a, a+b, a+b+c]<br>
   * exclusive=true, reverse=false, [0, a, a+b]<br>
   * exclusive=false, reverse=true: [a+b+c, b+c, c]<br>
   * exclusive=true, reverse=true: [b+c, c, 0]<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param exclusive If true: exclude the first value
   * @param reverse If true: reverse the direction of the accumulation
   * @param axis Scalar axis argument for dimension to perform cumululative sum operations along (Size: AtLeast(min=1))
   * @return output  (NUMERIC type)
   */
  public INDArray cumsum(INDArray in, boolean exclusive, boolean reverse, int... axis) {
    NDValidation.validateNumerical("cumsum", "in", in);
    Preconditions.checkArgument(axis.length >= 1, "axis has incorrect size/length. Expected: axis.length >= 1, got %s", axis.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.CumSum(in, exclusive, reverse, axis))[0];
  }

  /**
   * Cumulative sum operation.<br>
   * For input: [ a, b, c], output is:<br>
   * exclusive=false, reverse=false: [a, a+b, a+b+c]<br>
   * exclusive=true, reverse=false, [0, a, a+b]<br>
   * exclusive=false, reverse=true: [a+b+c, b+c, c]<br>
   * exclusive=true, reverse=true: [b+c, c, 0]<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param axis Scalar axis argument for dimension to perform cumululative sum operations along (Size: AtLeast(min=1))
   * @return output  (NUMERIC type)
   */
  public INDArray cumsum(INDArray in, int... axis) {
    NDValidation.validateNumerical("cumsum", "in", in);
    Preconditions.checkArgument(axis.length >= 1, "axis has incorrect size/length. Expected: axis.length >= 1, got %s", axis.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.CumSum(in, false, false, axis))[0];
  }

  /**
   * Pairwise dot product reduction along dimension<br>
   * output = sum(i=0 ... size(dim)-1) x[i] * y[i]<br>
   *
   * @param x first input (NUMERIC type)
   * @param y second input (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output output variable (NUMERIC type)
   */
  public INDArray dot(INDArray x, INDArray y, int... dimensions) {
    NDValidation.validateNumerical("dot", "x", x);
    NDValidation.validateNumerical("dot", "y", y);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce3.Dot(x, y, dimensions));
  }

  /**
   * Dynamically partition the input variable values into the specified number of paritions, using the indices.<br>
   * Example:<br>
   * <pre><br>
   * input = [1,2,3,4,5]<br>
   * numPartitions = 2<br>
   * partitions = [1,0,0,1,0]<br>
   * out[0] = [2,3,5]<br>
   * out[1] = [1,4] }<br>
   * </pre><br>
   *
   * @param x Input variable (NUMERIC type)
   * @param partitions 1D input with values 0 to numPartitions-1 (INT type)
   * @param numPartitions Number of partitions, >= 1
   */
  public INDArray[] dynamicPartition(INDArray x, INDArray partitions, int numPartitions) {
    NDValidation.validateNumerical("dynamicPartition", "x", x);
    NDValidation.validateInteger("dynamicPartition", "partitions", partitions);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.DynamicPartition(x, partitions, numPartitions));
  }

  /**
   * Dynamically merge the specified input arrays into a single array, using the specified indices<br>
   *
   * @param indices Indices to use when merging. Must be >= 1, same length as input variables (INT type)
   * @param x Input variables. (NUMERIC type)
   * @return output Merged output variable (NUMERIC type)
   */
  public INDArray dynamicStitch(INDArray[] indices, INDArray... x) {
    NDValidation.validateInteger("dynamicStitch", "indices", indices);
    Preconditions.checkArgument(indices.length >= 1, "indices has incorrect size/length. Expected: indices.length >= 1, got %s", indices.length);
    NDValidation.validateNumerical("dynamicStitch", "x", x);
    Preconditions.checkArgument(x.length >= 1, "x has incorrect size/length. Expected: x.length >= 1, got %s", x.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.DynamicStitch(indices, x))[0];
  }

  /**
   * Equals operation: elementwise x == y<br>
   *
   * Return boolean array with values true where satisfied, or false otherwise.<br>
   *
   * @param x Input array (NUMERIC type)
   * @param y Double value argument to use in operation
   * @return output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
   */
  public INDArray eq(INDArray x, double y) {
    NDValidation.validateNumerical("eq", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarEquals(x, y));
  }

  /**
   * Equal to operation: elementwise x == y<br>
   * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]<br>
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html<br>
   *
   * Return boolean array with values true where satisfied, or false otherwise.<br>
   *
   * @param x Input 1 (NUMERIC type)
   * @param y Input 2 (NUMERIC type)
   * @return output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
   */
  public INDArray eq(INDArray x, INDArray y) {
    NDValidation.validateNumerical("eq", "x", x);
    NDValidation.validateNumerical("eq", "y", y);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.EqualTo(x, y))[0];
  }

  /**
   * Reshape the input by adding a 1 at the specified location.<br>
   * For example, if input has shape [a, b], then output shape is:<br>
   * axis = 0: [1, a, b]<br>
   * axis = 1: [a, 1, b]<br>
   * axis = 2: [a, b, 1]<br>
   *
   * @param x Input variable (NDARRAY type)
   * @param axis Axis to expand
   * @return output Output variable (NUMERIC type)
   */
  public INDArray expandDims(INDArray x, int axis) {
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.ExpandDims(x, axis))[0];
  }

  /**
   * Generate an output variable with the specified (dynamic) shape with all elements set to the specified value<br>
   *
   * @param shape Shape: must be a 1D array/variable (INT type)
   * @param dataType Datatype of the output array
   * @param value Value to set all elements to
   * @return output Output variable (NUMERIC type)
   */
  public INDArray fill(INDArray shape, DataType dataType, double value) {
    NDValidation.validateInteger("fill", "shape", shape);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.Fill(shape, dataType, value))[0];
  }

  /**
   * Gather slices from the input variable where the indices are specified as fixed int[] values.<br>
   * Output shape is same as input shape, except for axis dimension, which has size equal to indices.length.<br>
   *
   * @param df Input variable (NUMERIC type)
   * @param indices Indices to get (Size: AtLeast(min=1))
   * @param axis Axis that the indices refer to
   * @return output Output variable with slices pulled from the specified axis (NUMERIC type)
   */
  public INDArray gather(INDArray df, int[] indices, int axis) {
    NDValidation.validateNumerical("gather", "df", df);
    Preconditions.checkArgument(indices.length >= 1, "indices has incorrect size/length. Expected: indices.length >= 1, got %s", indices.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.Gather(df, indices, axis))[0];
  }

  /**
   * Gather slices from the input variable where the indices are specified as dynamic array values.<br>
   * Output shape is same as input shape, except for axis dimension, which has size equal to indices.length.<br>
   *
   * @param df Input variable (NUMERIC type)
   * @param indices Indices to get slices for. Rank 0 or 1 input (INT type)
   * @param axis Axis that the indices refer to
   * @return output Output variable with slices pulled from the specified axis (NUMERIC type)
   */
  public INDArray gather(INDArray df, INDArray indices, int axis) {
    NDValidation.validateNumerical("gather", "df", df);
    NDValidation.validateInteger("gather", "indices", indices);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.Gather(df, indices, axis))[0];
  }

  /**
   * Gather slices from df with shape specified by indices. <br>
   *
   * @param df  (NUMERIC type)
   * @param indices  (NUMERIC type)
   * @return output  (NUMERIC type)
   */
  public INDArray gatherNd(INDArray df, INDArray indices) {
    NDValidation.validateNumerical("gatherNd", "df", df);
    NDValidation.validateNumerical("gatherNd", "indices", indices);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.GatherNd(df, indices))[0];
  }

  /**
   * Greater than operation: elementwise x > y<br>
   *
   * Return boolean array with values true where satisfied, or false otherwise.<br>
   *
   * @param x Input array (NUMERIC type)
   * @param y Double value argument to use in operation
   * @return output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
   */
  public INDArray gt(INDArray x, double y) {
    NDValidation.validateNumerical("gt", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarGreaterThan(x, y));
  }

  /**
   * Greater than operation: elementwise x > y<br>
   * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]<br>
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html<br>
   *
   * Return boolean array with values true where satisfied, or false otherwise.<br>
   *
   * @param x Input 1 (NUMERIC type)
   * @param y Input 2 (NUMERIC type)
   * @return output Output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
   */
  public INDArray gt(INDArray x, INDArray y) {
    NDValidation.validateNumerical("gt", "x", x);
    NDValidation.validateNumerical("gt", "y", y);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.GreaterThan(x, y))[0];
  }

  /**
   * Greater than or equals operation: elementwise x >= y<br>
   *
   * Return boolean array with values true where satisfied, or false otherwise.<br>
   *
   * @param x Input array (NUMERIC type)
   * @param y Double value argument to use in operation
   * @return output Output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
   */
  public INDArray gte(INDArray x, double y) {
    NDValidation.validateNumerical("gte", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarGreaterThanOrEqual(x, y));
  }

  /**
   * Greater than or equal to operation: elementwise x >= y<br>
   * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]<br>
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html<br>
   *
   * Return boolean array with values true where satisfied, or false otherwise.<br>
   *
   * @param x Input 1 (NUMERIC type)
   * @param y Input 2 (NUMERIC type)
   * @return output  (NUMERIC type)
   */
  public INDArray gte(INDArray x, INDArray y) {
    NDValidation.validateNumerical("gte", "x", x);
    NDValidation.validateNumerical("gte", "y", y);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.GreaterThanOrEqual(x, y))[0];
  }

  /**
   * Elementwise identity operation: out = x<br>
   *
   * @param input Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray identity(INDArray input) {
    NDValidation.validateNumerical("identity", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.same.Identity(input))[0];
  }

  /**
   * Compute the inverse permutation indices for a permutation operation<br>
   * Example: if input is [2, 0, 1] then output is [1, 2, 0]<br>
   * The idea is that x.permute(input).permute(invertPermutation(input)) == x<br>
   *
   * @param input 1D indices for permutation (INT type)
   * @return output 1D inverted permutation (INT type)
   */
  public INDArray invertPermutation(INDArray input) {
    NDValidation.validateInteger("invertPermutation", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.InvertPermutation(input))[0];
  }

  /**
   * Is the director a numeric tensor? In the current version of ND4J/SameDiff, this always returns true/1<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output scalar boolean with value true or false (NDARRAY type)
   */
  public INDArray isNumericTensor(INDArray x) {
    NDValidation.validateNumerical("isNumericTensor", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.IsNumericTensor(x))[0];
  }

  /**
   * Create a new 1d array with values evenly spaced between values 'start' and 'stop'<br>
   * For example, linspace(start=3.0, stop=4.0, number=3) will generate [3.0, 3.5, 4.0]<br>
   *
   * @param dataType Data type of the output array
   * @param start Start value
   * @param stop Stop value
   * @param number Number of values to generate
   * @return output INDArray  with linearly spaced elements (NUMERIC type)
   */
  public INDArray linspace(DataType dataType, double start, double stop, long number) {
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.Linspace(dataType, start, stop, number))[0];
  }

  /**
   * Create a new 1d array with values evenly spaced between values 'start' and 'stop'<br>
   * For example, linspace(start=3.0, stop=4.0, number=3) will generate [3.0, 3.5, 4.0]<br>
   *
   * @param start Start value (NUMERIC type)
   * @param stop Stop value (NUMERIC type)
   * @param number Number of values to generate (LONG type)
   * @param dataType Data type of the output array
   * @return output INDArray  with linearly spaced elements (NUMERIC type)
   */
  public INDArray linspace(INDArray start, INDArray stop, INDArray number, DataType dataType) {
    NDValidation.validateNumerical("linspace", "start", start);
    NDValidation.validateNumerical("linspace", "stop", stop);
    NDValidation.validateInteger("linspace", "number", number);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.Linspace(start, stop, number, dataType))[0];
  }

  /**
   * Less than operation: elementwise x < y<br>
   *
   * Return boolean array with values true where satisfied, or false otherwise.<br>
   *
   * @param x Input array (NUMERIC type)
   * @param y Double value argument to use in operation
   * @return output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
   */
  public INDArray lt(INDArray x, double y) {
    NDValidation.validateNumerical("lt", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarLessThan(x, y));
  }

  /**
   * Less than operation: elementwise x < y<br>
   * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]<br>
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html<br>
   *
   * Return boolean array with values true where satisfied, or false otherwise.<br>
   *
   * @param x Input 1 (NUMERIC type)
   * @param y Input 2 (NUMERIC type)
   * @return output Output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
   */
  public INDArray lt(INDArray x, INDArray y) {
    NDValidation.validateNumerical("lt", "x", x);
    NDValidation.validateNumerical("lt", "y", y);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.LessThan(x, y))[0];
  }

  /**
   * Less than or equals operation: elementwise x <= y<br>
   *
   * Return boolean array with values true where satisfied, or false otherwise.<br>
   *
   * @param x Input array (NUMERIC type)
   * @param y Double value argument to use in operation
   * @return output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
   */
  public INDArray lte(INDArray x, double y) {
    NDValidation.validateNumerical("lte", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarLessThanOrEqual(x, y));
  }

  /**
   * Less than or equal to operation: elementwise x <= y<br>
   * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]<br>
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html<br>
   *
   * Return boolean array with values true where satisfied, or false otherwise.<br>
   *
   * @param x Input 1 (NUMERIC type)
   * @param y Input 2 (NUMERIC type)
   * @return output Output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
   */
  public INDArray lte(INDArray x, INDArray y) {
    NDValidation.validateNumerical("lte", "x", x);
    NDValidation.validateNumerical("lte", "y", y);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.LessThanOrEqual(x, y))[0];
  }

  /**
   * Returns a boolean mask of equal shape to the input, where the condition is satisfied - value 1 where satisfied, 0 otherwise<br>
   *
   * @param in Input (NUMERIC type)
   * @param condition Condition
   * @return output Boolean mask (NUMERIC type)
   */
  public INDArray matchCondition(INDArray in, Condition condition) {
    NDValidation.validateNumerical("matchCondition", "in", in);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.bool.MatchConditionTransform(in, condition));
  }

  /**
   * Returns a count of the number of elements that satisfy the condition<br>
   *
   * @param in Input (NUMERIC type)
   * @param condition Condition
   * @return output Number of elements that the condition is satisfied for (NUMERIC type)
   */
  public INDArray matchConditionCount(INDArray in, Condition condition) {
    NDValidation.validateNumerical("matchConditionCount", "in", in);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.longer.MatchCondition(in, condition));
  }

  /**
   * Returns a count of the number of elements that satisfy the condition (for each slice along the specified dimensions)<br>
   *
   * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
   * the mean along a dimension).<br>
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
   * keepDims = true: [a,1,c]<br>
   * keepDims = false: [a,c]<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param condition Condition
   * @param keepDim If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Number of elements that the condition is satisfied for (NUMERIC type)
   */
  public INDArray matchConditionCount(INDArray in, Condition condition, boolean keepDim,
      int... dimensions) {
    NDValidation.validateNumerical("matchConditionCount", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.longer.MatchCondition(in, condition, keepDim, dimensions));
  }

  /**
   * Returns a count of the number of elements that satisfy the condition (for each slice along the specified dimensions)<br>
   *
   * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
   * the mean along a dimension).<br>
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
   * keepDims = true: [a,1,c]<br>
   * keepDims = false: [a,c]<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param condition Condition
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Number of elements that the condition is satisfied for (NUMERIC type)
   */
  public INDArray matchConditionCount(INDArray in, Condition condition, int... dimensions) {
    NDValidation.validateNumerical("matchConditionCount", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.longer.MatchCondition(in, condition, false, dimensions));
  }

  /**
   * Max array reduction operation, optionally along specified dimensions<br>
   *
   * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
   * the mean along a dimension).<br>
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
   * keepDims = true: [a,1,c]<br>
   * keepDims = false: [a,c]<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray max(INDArray x, boolean keepDims, int... dimensions) {
    NDValidation.validateNumerical("max", "x", x);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.same.Max(x, keepDims, dimensions));
  }

  /**
   * Max array reduction operation, optionally along specified dimensions<br>
   *
   * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
   * the mean along a dimension).<br>
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
   * keepDims = true: [a,1,c]<br>
   * keepDims = false: [a,c]<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray max(INDArray x, int... dimensions) {
    NDValidation.validateNumerical("max", "x", x);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.same.Max(x, false, dimensions));
  }

  /**
   * Element-wise maximum operation: out[i] = max(first[i], second[i])<br>
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]<br>
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html<br>
   *
   * @param first First input array (NUMERIC type)
   * @param second Second input array (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray max(INDArray first, INDArray second) {
    NDValidation.validateNumerical("max", "first", first);
    NDValidation.validateNumerical("max", "second", second);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.Max(first, second))[0];
  }

  /**
   * Mean (average) array reduction operation, optionally along specified dimensions<br>
   *
   * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
   * the mean along a dimension).<br>
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
   * keepDims = true: [a,1,c]<br>
   * keepDims = false: [a,c]<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray mean(INDArray x, boolean keepDims, int... dimensions) {
    NDValidation.validateNumerical("mean", "x", x);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.floating.Mean(x, keepDims, dimensions));
  }

  /**
   * Mean (average) array reduction operation, optionally along specified dimensions<br>
   *
   * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
   * the mean along a dimension).<br>
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
   * keepDims = true: [a,1,c]<br>
   * keepDims = false: [a,c]<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray mean(INDArray x, int... dimensions) {
    NDValidation.validateNumerical("mean", "x", x);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.floating.Mean(x, false, dimensions));
  }

  /**
   * The merge operation is a control operation that forwards the either of the inputs to the output, when<br>
   * the first of them becomes available. If both are available, the output is undefined (either input could<br>
   * be forwarded to the output)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param y Input variable (NUMERIC type)
   * @return output Output (NUMERIC type)
   */
  public INDArray merge(INDArray x, INDArray y) {
    NDValidation.validateNumerical("merge", "x", x);
    NDValidation.validateNumerical("merge", "y", y);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.controlflow.compat.Merge(x, y))[0];
  }

  /**
   * Minimum array reduction operation, optionally along specified dimensions. out = min(in)<br>
   *
   * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
   * the mean along a dimension).<br>
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
   * keepDims = true: [a,1,c]<br>
   * keepDims = false: [a,c]<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray min(INDArray x, boolean keepDims, int... dimensions) {
    NDValidation.validateNumerical("min", "x", x);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.same.Min(x, keepDims, dimensions));
  }

  /**
   * Minimum array reduction operation, optionally along specified dimensions. out = min(in)<br>
   *
   * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
   * the mean along a dimension).<br>
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
   * keepDims = true: [a,1,c]<br>
   * keepDims = false: [a,c]<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray min(INDArray x, int... dimensions) {
    NDValidation.validateNumerical("min", "x", x);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.same.Min(x, false, dimensions));
  }

  /**
   * Element-wise minimum operation: out[i] = min(first[i], second[i])<br>
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]<br>
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html<br>
   *
   * @param first First input array (NUMERIC type)
   * @param second Second input array (NUMERIC type)
   * @return output Second input array (NUMERIC type)
   */
  public INDArray min(INDArray first, INDArray second) {
    NDValidation.validateNumerical("min", "first", first);
    NDValidation.validateNumerical("min", "second", second);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.Min(first, second))[0];
  }

  /**
   * Matrix multiplication: out = mmul(x,y)<br>
   * Supports specifying transpose argument to perform operation such as mmul(a^T, b), etc.<br>
   *
   * @param x First input variable (NUMERIC type)
   * @param y Second input variable (NUMERIC type)
   * @param transposeX Transpose x (first argument)
   * @param transposeY Transpose y (second argument)
   * @param transposeZ Transpose result array
   * @return output  (NUMERIC type)
   */
  public INDArray mmul(INDArray x, INDArray y, boolean transposeX, boolean transposeY,
      boolean transposeZ) {
    NDValidation.validateNumerical("mmul", "x", x);
    NDValidation.validateNumerical("mmul", "y", y);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.Mmul(x, y, transposeX, transposeY, transposeZ))[0];
  }

  /**
   * Matrix multiplication: out = mmul(x,y)<br>
   * Supports specifying transpose argument to perform operation such as mmul(a^T, b), etc.<br>
   *
   * @param x First input variable (NUMERIC type)
   * @param y Second input variable (NUMERIC type)
   * @return output  (NUMERIC type)
   */
  public INDArray mmul(INDArray x, INDArray y) {
    NDValidation.validateNumerical("mmul", "x", x);
    NDValidation.validateNumerical("mmul", "y", y);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.Mmul(x, y, false, false, false))[0];
  }

  /**
   * Not equals operation: elementwise x != y<br>
   *
   * Return boolean array with values true where satisfied, or false otherwise.<br>
   *
   * @param x Input array (NUMERIC type)
   * @param y Double value argument to use in operation
   * @return output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
   */
  public INDArray neq(INDArray x, double y) {
    NDValidation.validateNumerical("neq", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarNotEquals(x, y));
  }

  /**
   * Not equal to operation: elementwise x != y<br>
   * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]<br>
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html<br>
   *
   * Return boolean array with values true where satisfied, or false otherwise.<br>
   *
   * @param x Input 1 (NUMERIC type)
   * @param y Input 2 (NUMERIC type)
   * @return output Boolean array out, with values true/false based on where the condition is satisfied (NUMERIC type)
   */
  public INDArray neq(INDArray x, INDArray y) {
    NDValidation.validateNumerical("neq", "x", x);
    NDValidation.validateNumerical("neq", "y", y);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.NotEqualTo(x, y))[0];
  }

  /**
   * Norm1 (L1 norm) reduction operation: The output contains the L1 norm for each tensor/subset along the specified dimensions: <br>
   * out = sum_i abs(x[i])<br>
   *
   * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
   * the mean along a dimension).<br>
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
   * keepDims = true: [a,1,c]<br>
   * keepDims = false: [a,c]<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
   * @param dimensions dimensions to reduce over (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray norm1(INDArray x, boolean keepDims, int... dimensions) {
    NDValidation.validateNumerical("norm1", "x", x);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.floating.Norm1(x, keepDims, dimensions));
  }

  /**
   * Norm1 (L1 norm) reduction operation: The output contains the L1 norm for each tensor/subset along the specified dimensions: <br>
   * out = sum_i abs(x[i])<br>
   *
   * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
   * the mean along a dimension).<br>
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
   * keepDims = true: [a,1,c]<br>
   * keepDims = false: [a,c]<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param dimensions dimensions to reduce over (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray norm1(INDArray x, int... dimensions) {
    NDValidation.validateNumerical("norm1", "x", x);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.floating.Norm1(x, false, dimensions));
  }

  /**
   * Norm2 (L2 norm) reduction operation: The output contains the L2 norm for each tensor/subset along the specified dimensions:<br>
   * out = sqrt(sum_i x[i]^2)<br>
   *
   * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
   * the mean along a dimension).<br>
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
   * keepDims = true: [a,1,c]<br>
   * keepDims = false: [a,c]<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
   * @param dimensions dimensions dimensions to reduce over (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray norm2(INDArray x, boolean keepDims, int... dimensions) {
    NDValidation.validateNumerical("norm2", "x", x);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.floating.Norm2(x, keepDims, dimensions));
  }

  /**
   * Norm2 (L2 norm) reduction operation: The output contains the L2 norm for each tensor/subset along the specified dimensions:<br>
   * out = sqrt(sum_i x[i]^2)<br>
   *
   * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
   * the mean along a dimension).<br>
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
   * keepDims = true: [a,1,c]<br>
   * keepDims = false: [a,c]<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param dimensions dimensions dimensions to reduce over (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray norm2(INDArray x, int... dimensions) {
    NDValidation.validateNumerical("norm2", "x", x);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.floating.Norm2(x, false, dimensions));
  }

  /**
   * Max norm (infinity norm) reduction operation: The output contains the max norm for each tensor/subset along the<br>
   * specified dimensions:<br>
   * out = max(abs(x[i]))<br>
   *
   * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
   * the mean along a dimension).<br>
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
   * keepDims = true: [a,1,c]<br>
   * keepDims = false: [a,c]<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
   * @param dimensions dimensions to reduce over (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray normmax(INDArray x, boolean keepDims, int... dimensions) {
    NDValidation.validateNumerical("normmax", "x", x);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.floating.NormMax(x, keepDims, dimensions));
  }

  /**
   * Max norm (infinity norm) reduction operation: The output contains the max norm for each tensor/subset along the<br>
   * specified dimensions:<br>
   * out = max(abs(x[i]))<br>
   *
   * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
   * the mean along a dimension).<br>
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
   * keepDims = true: [a,1,c]<br>
   * keepDims = false: [a,c]<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param dimensions dimensions to reduce over (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray normmax(INDArray x, int... dimensions) {
    NDValidation.validateNumerical("normmax", "x", x);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.floating.NormMax(x, false, dimensions));
  }

  /**
   * Convert the array to a one-hot array with walues and  for each entry<br>
   * If input has shape [ a, ..., n] then output has shape [ a, ..., n, depth],<br>
   * with {out[i, ..., j, in[i,...,j]]  with other values being set to<br>
   *
   * @param indices Indices - value 0 to depth-1 (NUMERIC type)
   * @param depth Number of classes
   * @param axis 
   * @param on 
   * @param off 
   * @param dataType Output data type
   * @return output Output variable (NUMERIC type)
   */
  public INDArray oneHot(INDArray indices, int depth, int axis, double on, double off,
      DataType dataType) {
    NDValidation.validateNumerical("oneHot", "indices", indices);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.OneHot(indices, depth, axis, on, off, dataType))[0];
  }

  /**
   * Convert the array to a one-hot array with walues and  for each entry<br>
   * If input has shape [ a, ..., n] then output has shape [ a, ..., n, depth],<br>
   * with {out[i, ..., j, in[i,...,j]]  with other values being set to<br>
   *
   * @param indices Indices - value 0 to depth-1 (NUMERIC type)
   * @param depth Number of classes
   * @param axis 
   * @param on 
   * @param off 
   * @return output Output variable (NUMERIC type)
   */
  public INDArray oneHot(INDArray indices, int depth, int axis, double on, double off) {
    NDValidation.validateNumerical("oneHot", "indices", indices);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.OneHot(indices, depth, axis, on, off, DataType.FLOAT))[0];
  }

  /**
   * Convert the array to a one-hot array with walues 0 and 1 for each entry<br>
   * If input has shape [ a, ..., n] then output has shape [ a, ..., n, depth],<br>
   * with out[i, ..., j, in[i,...,j]] = 1 with other values being set to 0<br>
   * see oneHot(SDVariable, int, int, double, double)<br>
   *
   * @param indices Indices - value 0 to depth-1 (NUMERIC type)
   * @param depth Number of classes
   * @return output Output variable (NUMERIC type)
   */
  public INDArray oneHot(INDArray indices, int depth) {
    NDValidation.validateNumerical("oneHot", "indices", indices);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.OneHot(indices, depth))[0];
  }

  /**
   * Return a variable of all 1s, with the same shape as the input variable. Note that this is dynamic:<br>
   * if the input shape changes in later execution, the returned variable's shape will also be updated<br>
   *
   * @param input Input INDArray  (NUMERIC type)
   * @return output A new INDArray  with the same (dynamic) shape as the input (NUMERIC type)
   */
  public INDArray onesLike(INDArray input) {
    NDValidation.validateNumerical("onesLike", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.OnesLike(input))[0];
  }

  /**
   * As per onesLike(String, SDVariable) but the output datatype may be specified<br>
   *
   * @param input  (NUMERIC type)
   * @param dataType 
   * @return output  (NUMERIC type)
   */
  public INDArray onesLike(INDArray input, DataType dataType) {
    NDValidation.validateNumerical("onesLike", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.OnesLike(input, dataType))[0];
  }

  /**
   * Array permutation operation: permute the dimensions according to the specified permutation indices.<br>
   * Example: if input has shape [a,b,c] and dimensions = [2,0,1] the output has shape [c,a,b]<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param dimensions Permute dimensions (INT type)
   * @return output Output variable (permuted input) (NUMERIC type)
   */
  public INDArray permute(INDArray x, INDArray dimensions) {
    NDValidation.validateNumerical("permute", "x", x);
    NDValidation.validateInteger("permute", "dimensions", dimensions);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.Permute(x, dimensions))[0];
  }

  /**
   * Array permutation operation: permute the dimensions according to the specified permutation indices.<br>
   * Example: if input has shape [a,b,c] and dimensions = [2,0,1] the output has shape [c,a,b]<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param dimensions  (Size: AtLeast(min=0))
   * @return output Output variable (permuted input) (NUMERIC type)
   */
  public INDArray permute(INDArray x, int... dimensions) {
    NDValidation.validateNumerical("permute", "x", x);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.Permute(x, dimensions))[0];
  }

  /**
   * Product array reduction operation, optionally along specified dimensions<br>
   *
   * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
   * the mean along a dimension).<br>
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
   * keepDims = true: [a,1,c]<br>
   * keepDims = false: [a,c]<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output  (NUMERIC type)
   */
  public INDArray prod(INDArray x, boolean keepDims, int... dimensions) {
    NDValidation.validateNumerical("prod", "x", x);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.same.Prod(x, keepDims, dimensions));
  }

  /**
   * Product array reduction operation, optionally along specified dimensions<br>
   *
   * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
   * the mean along a dimension).<br>
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
   * keepDims = true: [a,1,c]<br>
   * keepDims = false: [a,c]<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output  (NUMERIC type)
   */
  public INDArray prod(INDArray x, int... dimensions) {
    NDValidation.validateNumerical("prod", "x", x);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.same.Prod(x, false, dimensions));
  }

  /**
   * Create a new variable with a 1d array, where the values start at from and increment by step<br>
   * up to (but not including) limit.<br>
   * For example, range(1.0, 3.0, 0.5) will return [1.0, 1.5, 2.0, 2.5]<br>
   *
   * @param from Initial/smallest value
   * @param to Largest value (exclusive)
   * @param step Step size
   * @param dataType 
   * @return output INDArray  with the specified values (NUMERIC type)
   */
  public INDArray range(double from, double to, double step, DataType dataType) {
    return Nd4j.exec(new org.nd4j.linalg.api.ops.random.impl.Range(from, to, step, dataType))[0];
  }

  /**
   * Create a new variable with a 1d array, where the values start at from and increment by step<br>
   * up to (but not including) limit.<br>
   * For example, range(1.0, 3.0, 0.5) will return [1.0, 1.5, 2.0, 2.5]<br>
   *
   * @param from Initial/smallest value (NUMERIC type)
   * @param to Largest value (exclusive) (NUMERIC type)
   * @param step Step size (NUMERIC type)
   * @param dataType 
   * @return output INDArray  with the specified values (NUMERIC type)
   */
  public INDArray range(INDArray from, INDArray to, INDArray step, DataType dataType) {
    NDValidation.validateNumerical("range", "from", from);
    NDValidation.validateNumerical("range", "to", to);
    NDValidation.validateNumerical("range", "step", step);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.random.impl.Range(from, to, step, dataType))[0];
  }

  /**
   * Returns the rank (number of dimensions, i.e., length(shape)) of the specified INDArray  as a 0D scalar variable<br>
   *
   * @param in Input variable (NUMERIC type)
   * @return output (scalar) output variable with value equal to the rank of the input variable (NUMERIC type)
   */
  public INDArray rank(INDArray in) {
    NDValidation.validateNumerical("rank", "in", in);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.Rank(in))[0];
  }

  /**
   * Element-wise replace where condition:<br>
   * out[i] = from[i] if condition(update[i]) is satisfied, or<br>
   * out[i] = update[i] if condition(update[i]) is NOT satisfied<br>
   *
   * @param update Source array (NUMERIC type)
   * @param from Replacement values array (used conditionally). Must be same shape as 'update' array (NUMERIC type)
   * @param condition Condition to check on update array elements
   * @return output New array with values replaced where condition is satisfied (NUMERIC type)
   */
  public INDArray replaceWhere(INDArray update, INDArray from, Condition condition) {
    NDValidation.validateNumerical("replaceWhere", "update", update);
    NDValidation.validateNumerical("replaceWhere", "from", from);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.comparison.CompareAndReplace(update, from, condition));
  }

  /**
   * Element-wise replace where condition:<br>
   * out[i] = value if condition(update[i]) is satisfied, or<br>
   * out[i] = update[i] if condition(update[i]) is NOT satisfied<br>
   *
   * @param update Source array (NUMERIC type)
   * @param value Value to set at the output, if the condition is satisfied
   * @param condition Condition to check on update array elements
   * @return output New array with values replaced where condition is satisfied (NUMERIC type)
   */
  public INDArray replaceWhere(INDArray update, double value, Condition condition) {
    NDValidation.validateNumerical("replaceWhere", "update", update);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.comparison.CompareAndSet(update, value, condition));
  }

  /**
   * Reshape the input variable to the specified (fixed) shape. The output variable will have the same values as the<br>
   * input, but with the specified shape.<br>
   * Note that prod(shape) must match length(input) == prod(input.shape)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param shape New shape for variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray reshape(INDArray x, INDArray shape) {
    NDValidation.validateNumerical("reshape", "x", x);
    NDValidation.validateNumerical("reshape", "shape", shape);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.Reshape(x, shape))[0];
  }

  /**
   * Reshape the input variable to the specified (fixed) shape. The output variable will have the same values as the<br>
   * input, but with the specified shape.<br>
   * Note that prod(shape) must match length(input) == prod(input.shape)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param shape New shape for variable (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray reshape(INDArray x, long... shape) {
    NDValidation.validateNumerical("reshape", "x", x);
    Preconditions.checkArgument(shape.length >= 0, "shape has incorrect size/length. Expected: shape.length >= 0, got %s", shape.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.Reshape(x, shape))[0];
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
   * reverse(in, 1):<br>
   * [4, 5, 6]<br>
   * [1, 2 3]<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param dimensions Input variable (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray reverse(INDArray x, int... dimensions) {
    NDValidation.validateNumerical("reverse", "x", x);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.Reverse(x, dimensions))[0];
  }

  /**
   * Reverse sequence op: for each slice along dimension seqDimension, the first seqLength values are reversed<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param seq_lengths Length of the sequences (INT type)
   * @param seqDim Sequence dimension
   * @param batchDim Batch dimension
   * @return output Reversed sequences (NUMERIC type)
   */
  public INDArray reverseSequence(INDArray x, INDArray seq_lengths, int seqDim, int batchDim) {
    NDValidation.validateNumerical("reverseSequence", "x", x);
    NDValidation.validateInteger("reverseSequence", "seq_lengths", seq_lengths);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.ReverseSequence(x, seq_lengths, seqDim, batchDim))[0];
  }

  /**
   * Reverse sequence op: for each slice along dimension seqDimension, the first seqLength values are reversed<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param seq_lengths Length of the sequences (INT type)
   * @return output Reversed sequences (NUMERIC type)
   */
  public INDArray reverseSequence(INDArray x, INDArray seq_lengths) {
    NDValidation.validateNumerical("reverseSequence", "x", x);
    NDValidation.validateInteger("reverseSequence", "seq_lengths", seq_lengths);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.ReverseSequence(x, seq_lengths, -1, 0))[0];
  }

  /**
   * Element-wise scalar floor modulus operation: out = floorMod(in, value).<br>
   * i.e., returns the remainder after division by 'value'<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param value Scalar value to compare
   * @return output Output variable (NUMERIC type)
   */
  public INDArray scalarFloorMod(INDArray in, double value) {
    NDValidation.validateNumerical("scalarFloorMod", "in", in);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.scalar.ScalarFMod(in, value));
  }

  /**
   * Element-wise scalar maximum operation: out = max(in, value)<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param value Scalar value to compare
   * @return output Scalar value to compare (NUMERIC type)
   */
  public INDArray scalarMax(INDArray in, double value) {
    NDValidation.validateNumerical("scalarMax", "in", in);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.scalar.ScalarMax(in, value));
  }

  /**
   * Element-wise scalar minimum operation: out = min(in, value)<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param value Scalar value to compare
   * @return output Output variable (NUMERIC type)
   */
  public INDArray scalarMin(INDArray in, double value) {
    NDValidation.validateNumerical("scalarMin", "in", in);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.scalar.ScalarMin(in, value));
  }

  /**
   * Return a variable with equal shape to the input, but all elements set to value 'set'<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param set Value to set
   * @return output Output variable (NUMERIC type)
   */
  public INDArray scalarSet(INDArray in, double set) {
    NDValidation.validateNumerical("scalarSet", "in", in);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.scalar.ScalarSet(in, set));
  }

  /**
   * Scatter addition operation.<br>
   *
   * If indices is rank 0 (a scalar), then out[index, ...] = out[index, ...] + op(updates[...])<br>
   * If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] = out[indices[i], ...] + op(updates[i, ...])<br>
   * If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] = out[indices[i], ..., indices[k], ...]  + op(updates[i, ..., k, ...]) <br>
   * Note that if multiple indices refer to the same location, the contributions from each is handled correctly. <br>
   *
   * @param ref Initial/source variable (NUMERIC type)
   * @param indices Indices array (NUMERIC type)
   * @param updates Updates to add to the initial/source array (NUMERIC type)
   * @return output The updated variable (NUMERIC type)
   */
  public INDArray scatterAdd(INDArray ref, INDArray indices, INDArray updates) {
    NDValidation.validateNumerical("scatterAdd", "ref", ref);
    NDValidation.validateNumerical("scatterAdd", "indices", indices);
    NDValidation.validateNumerical("scatterAdd", "updates", updates);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.scatter.ScatterAdd(ref, indices, updates))[0];
  }

  /**
   * Scatter division operation.<br>
   *
   * If indices is rank 0 (a scalar), then out[index, ...] = out[index, ...] + op(updates[...])<br>
   * If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] = out[indices[i], ...] + op(updates[i, ...])<br>
   * If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] = out[indices[i], ..., indices[k], ...]  + op(updates[i, ..., k, ...]) <br>
   * Note that if multiple indices refer to the same location, the contributions from each is handled correctly. <br>
   *
   * @param ref Initial/source variable (NUMERIC type)
   * @param indices Indices array (NUMERIC type)
   * @param updates Updates to add to the initial/source array (NUMERIC type)
   * @return output The updated variable (NUMERIC type)
   */
  public INDArray scatterDiv(INDArray ref, INDArray indices, INDArray updates) {
    NDValidation.validateNumerical("scatterDiv", "ref", ref);
    NDValidation.validateNumerical("scatterDiv", "indices", indices);
    NDValidation.validateNumerical("scatterDiv", "updates", updates);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.scatter.ScatterDiv(ref, indices, updates))[0];
  }

  /**
   * Scatter max operation.<br>
   *
   * If indices is rank 0 (a scalar), then out[index, ...] = out[index, ...] + op(updates[...])<br>
   * If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] = out[indices[i], ...] + op(updates[i, ...])<br>
   * If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] = out[indices[i], ..., indices[k], ...]  + op(updates[i, ..., k, ...]) <br>
   * Note that if multiple indices refer to the same location, the contributions from each is handled correctly. <br>
   *
   * @param ref Initial/source variable (NUMERIC type)
   * @param indices Indices array (NUMERIC type)
   * @param updates Updates to add to the initial/source array (NUMERIC type)
   * @return output The updated variable (NUMERIC type)
   */
  public INDArray scatterMax(INDArray ref, INDArray indices, INDArray updates) {
    NDValidation.validateNumerical("scatterMax", "ref", ref);
    NDValidation.validateNumerical("scatterMax", "indices", indices);
    NDValidation.validateNumerical("scatterMax", "updates", updates);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.scatter.ScatterMax(ref, indices, updates))[0];
  }

  /**
   * Scatter min operation.<br>
   *
   * If indices is rank 0 (a scalar), then out[index, ...] = out[index, ...] + op(updates[...])<br>
   * If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] = out[indices[i], ...] + op(updates[i, ...])<br>
   * If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] = out[indices[i], ..., indices[k], ...]  + op(updates[i, ..., k, ...]) <br>
   * Note that if multiple indices refer to the same location, the contributions from each is handled correctly. <br>
   *
   * @param ref Initial/source variable (NUMERIC type)
   * @param indices Indices array (NUMERIC type)
   * @param updates Updates to add to the initial/source array (NUMERIC type)
   * @return output The updated variable (NUMERIC type)
   */
  public INDArray scatterMin(INDArray ref, INDArray indices, INDArray updates) {
    NDValidation.validateNumerical("scatterMin", "ref", ref);
    NDValidation.validateNumerical("scatterMin", "indices", indices);
    NDValidation.validateNumerical("scatterMin", "updates", updates);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.scatter.ScatterMin(ref, indices, updates))[0];
  }

  /**
   * Scatter multiplication operation.<br>
   *
   * If indices is rank 0 (a scalar), then out[index, ...] = out[index, ...] + op(updates[...])<br>
   * If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] = out[indices[i], ...] + op(updates[i, ...])<br>
   * If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] = out[indices[i], ..., indices[k], ...]  + op(updates[i, ..., k, ...]) <br>
   * Note that if multiple indices refer to the same location, the contributions from each is handled correctly. <br>
   *
   * @param ref Initial/source variable (NUMERIC type)
   * @param indices Indices array (NUMERIC type)
   * @param updates Updates to add to the initial/source array (NUMERIC type)
   * @return output The updated variable (NUMERIC type)
   */
  public INDArray scatterMul(INDArray ref, INDArray indices, INDArray updates) {
    NDValidation.validateNumerical("scatterMul", "ref", ref);
    NDValidation.validateNumerical("scatterMul", "indices", indices);
    NDValidation.validateNumerical("scatterMul", "updates", updates);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.scatter.ScatterMul(ref, indices, updates))[0];
  }

  /**
   * Scatter subtraction operation.<br>
   *
   * If indices is rank 0 (a scalar), then out[index, ...] = out[index, ...] + op(updates[...])<br>
   * If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] = out[indices[i], ...] + op(updates[i, ...])<br>
   * If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] = out[indices[i], ..., indices[k], ...]  + op(updates[i, ..., k, ...]) <br>
   * Note that if multiple indices refer to the same location, the contributions from each is handled correctly. <br>
   *
   * @param ref Initial/source variable (NUMERIC type)
   * @param indices Indices array (NUMERIC type)
   * @param updates Updates to add to the initial/source array (NUMERIC type)
   * @return output The updated variable (NUMERIC type)
   */
  public INDArray scatterSub(INDArray ref, INDArray indices, INDArray updates) {
    NDValidation.validateNumerical("scatterSub", "ref", ref);
    NDValidation.validateNumerical("scatterSub", "indices", indices);
    NDValidation.validateNumerical("scatterSub", "updates", updates);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.scatter.ScatterSub(ref, indices, updates))[0];
  }

  /**
   * Scatter update operation.<br>
   *
   * If indices is rank 0 (a scalar), then out[index, ...] = out[index, ...] + op(updates[...])<br>
   * If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] = out[indices[i], ...] + op(updates[i, ...])<br>
   * If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] = out[indices[i], ..., indices[k], ...]  + op(updates[i, ..., k, ...]) <br>
   * Note that if multiple indices refer to the same location, the contributions from each is handled correctly. <br>
   *
   * @param ref Initial/source variable (NUMERIC type)
   * @param indices Indices array (NUMERIC type)
   * @param updates Updates to add to the initial/source array (NUMERIC type)
   * @return output The updated variable (NUMERIC type)
   */
  public INDArray scatterUpdate(INDArray ref, INDArray indices, INDArray updates) {
    NDValidation.validateNumerical("scatterUpdate", "ref", ref);
    NDValidation.validateNumerical("scatterUpdate", "indices", indices);
    NDValidation.validateNumerical("scatterUpdate", "updates", updates);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.scatter.ScatterUpdate(ref, indices, updates))[0];
  }

  /**
   * Segment max operation.<br>
   *
   * If data =     [3, 6, 1, 4, 9, 2, 8]<br>
   * segmentIds =  [0, 0, 1, 1, 1, 2, 2]<br>
   * then output = [6, 9, 8] = [op(3,6), op(1,4,9), op(2,8)]<br>
   * Note that the segment IDs must be sorted from smallest to largest segment.<br>
   * See {unsortedSegment (String, SDVariable, SDVariable, int) ops<br>
   * for the same op without this sorted requirement<br>
   *
   * @param data Data to perform segment max on (NDARRAY type)
   * @param segmentIds Variable for the segment IDs (NUMERIC type)
   * @return output Segment output (NUMERIC type)
   */
  public INDArray segmentMax(INDArray data, INDArray segmentIds) {
    NDValidation.validateNumerical("segmentMax", "segmentIds", segmentIds);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.segment.SegmentMax(data, segmentIds))[0];
  }

  /**
   * Segment mean operation.<br>
   *
   * If data =     [3, 6, 1, 4, 9, 2, 8]<br>
   * segmentIds =  [0, 0, 1, 1, 1, 2, 2]<br>
   * then output = [6, 9, 8] = [op(3,6), op(1,4,9), op(2,8)]<br>
   * Note that the segment IDs must be sorted from smallest to largest segment.<br>
   * See {unsortedSegment (String, SDVariable, SDVariable, int) ops<br>
   * for the same op without this sorted requirement<br>
   *
   * @param data Data to perform segment max on (NDARRAY type)
   * @param segmentIds Variable for the segment IDs (NUMERIC type)
   * @return output Segment output (NUMERIC type)
   */
  public INDArray segmentMean(INDArray data, INDArray segmentIds) {
    NDValidation.validateNumerical("segmentMean", "segmentIds", segmentIds);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.segment.SegmentMean(data, segmentIds))[0];
  }

  /**
   * Segment min operation.<br>
   *
   * If data =     [3, 6, 1, 4, 9, 2, 8]<br>
   * segmentIds =  [0, 0, 1, 1, 1, 2, 2]<br>
   * then output = [6, 9, 8] = [op(3,6), op(1,4,9), op(2,8)]<br>
   * Note that the segment IDs must be sorted from smallest to largest segment.<br>
   * See {unsortedSegment (String, SDVariable, SDVariable, int) ops<br>
   * for the same op without this sorted requirement<br>
   *
   * @param data Data to perform segment max on (NDARRAY type)
   * @param segmentIds Variable for the segment IDs (NUMERIC type)
   * @return output Segment output (NUMERIC type)
   */
  public INDArray segmentMin(INDArray data, INDArray segmentIds) {
    NDValidation.validateNumerical("segmentMin", "segmentIds", segmentIds);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.segment.SegmentMin(data, segmentIds))[0];
  }

  /**
   * Segment product operation.<br>
   *
   * If data =     [3, 6, 1, 4, 9, 2, 8]<br>
   * segmentIds =  [0, 0, 1, 1, 1, 2, 2]<br>
   * then output = [6, 9, 8] = [op(3,6), op(1,4,9), op(2,8)]<br>
   * Note that the segment IDs must be sorted from smallest to largest segment.<br>
   * See {unsortedSegment (String, SDVariable, SDVariable, int) ops<br>
   * for the same op without this sorted requirement<br>
   *
   * @param data Data to perform segment max on (NDARRAY type)
   * @param segmentIds Variable for the segment IDs (NUMERIC type)
   * @return output Segment output (NUMERIC type)
   */
  public INDArray segmentProd(INDArray data, INDArray segmentIds) {
    NDValidation.validateNumerical("segmentProd", "segmentIds", segmentIds);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.segment.SegmentProd(data, segmentIds))[0];
  }

  /**
   * Segment sum operation.<br>
   *
   * If data =     [3, 6, 1, 4, 9, 2, 8]<br>
   * segmentIds =  [0, 0, 1, 1, 1, 2, 2]<br>
   * then output = [6, 9, 8] = [op(3,6), op(1,4,9), op(2,8)]<br>
   * Note that the segment IDs must be sorted from smallest to largest segment.<br>
   * See {unsortedSegment (String, SDVariable, SDVariable, int) ops<br>
   * for the same op without this sorted requirement<br>
   *
   * @param data Data to perform segment max on (NDARRAY type)
   * @param segmentIds Variable for the segment IDs (NUMERIC type)
   * @return output Segment output (NUMERIC type)
   */
  public INDArray segmentSum(INDArray data, INDArray segmentIds) {
    NDValidation.validateNumerical("segmentSum", "segmentIds", segmentIds);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.segment.SegmentSum(data, segmentIds))[0];
  }

  /**
   * Generate a sequence mask (with values 0 or 1) based on the specified lengths <br>
   * Specifically, out[i, ..., k, j] = (j < lengths[i, ..., k] ? 1.0 : 0.0)<br>
   *
   * @param lengths Lengths of the sequences (NUMERIC type)
   * @param maxLen Maximum sequence length
   * @param dataType 
   * @return output Output variable (NUMERIC type)
   */
  public INDArray sequenceMask(INDArray lengths, int maxLen, DataType dataType) {
    NDValidation.validateNumerical("sequenceMask", "lengths", lengths);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.SequenceMask(lengths, maxLen, dataType))[0];
  }

  /**
   * Generate a sequence mask (with values 0 or 1) based on the specified lengths <br>
   * Specifically, out[i, ..., k, j] = (j < lengths[i, ..., k] ? 1.0 : 0.0)<br>
   *
   * @param lengths Lengths of the sequences (NUMERIC type)
   * @param maxLen Maximum sequence length (INT type)
   * @param dataType 
   * @return output Output variable (NUMERIC type)
   */
  public INDArray sequenceMask(INDArray lengths, INDArray maxLen, DataType dataType) {
    NDValidation.validateNumerical("sequenceMask", "lengths", lengths);
    NDValidation.validateInteger("sequenceMask", "maxLen", maxLen);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.SequenceMask(lengths, maxLen, dataType))[0];
  }

  /**
   * see sequenceMask(String, SDVariable, SDVariable, DataType)<br>
   *
   * @param lengths  (NUMERIC type)
   * @param dataType 
   * @return output  (NUMERIC type)
   */
  public INDArray sequenceMask(INDArray lengths, DataType dataType) {
    NDValidation.validateNumerical("sequenceMask", "lengths", lengths);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.SequenceMask(lengths, dataType))[0];
  }

  /**
   * Returns the shape of the specified INDArray  as a 1D INDArray <br>
   *
   * @param input Input variable (NUMERIC type)
   * @return output 1D output variable with contents equal to the shape of the input (NUMERIC type)
   */
  public INDArray shape(INDArray input) {
    NDValidation.validateNumerical("shape", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.Shape(input))[0];
  }

  /**
   * Returns the size (number of elements, i.e., prod(shape)) of the specified INDArray  as a 0D scalar variable<br>
   *
   * @param in Input variable (NUMERIC type)
   * @return output 0D (scalar) output variable with value equal to the number of elements in the specified array (NUMERIC type)
   */
  public INDArray size(INDArray in) {
    NDValidation.validateNumerical("size", "in", in);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.Size(in))[0];
  }

  /**
   * Returns a rank 0 (scalar) variable for the size of the specified dimension.<br>
   * For example, if X has shape [10,20,30] then sizeAt(X,1)=20. Similarly, sizeAt(X,-1)=30<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param dimension Dimension to get size of
   * @return output Scalar INDArray  for size at specified variable (NUMERIC type)
   */
  public INDArray sizeAt(INDArray in, int dimension) {
    NDValidation.validateNumerical("sizeAt", "in", in);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.SizeAt(in, dimension))[0];
  }

  /**
   * Get a subset of the specified input, by specifying the first element and the size of the array.<br>
   * For example, if input is:<br>
   * [a, b, c]<br>
   * [d, e, f]<br>
   * then slice(input, begin=[0,1], size=[2,1] will return:<br>
   * [b]<br>
   * [e]<br>
   * Note that for each dimension i, begin[i] + size[i] <= input.size(i)<br>
   *
   * @param input input Variable to get subset of (NUMERIC type)
   * @param begin Beginning index. Must be same length as rank of input array (Size: AtLeast(min=1))
   * @param size Size of the output array. Must be same length as rank of input array (Size: AtLeast(min=1))
   * @return output Subset of the input (NUMERIC type)
   */
  public INDArray slice(INDArray input, int[] begin, int... size) {
    NDValidation.validateNumerical("slice", "input", input);
    Preconditions.checkArgument(begin.length >= 1, "begin has incorrect size/length. Expected: begin.length >= 1, got %s", begin.length);
    Preconditions.checkArgument(size.length >= 1, "size has incorrect size/length. Expected: size.length >= 1, got %s", size.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.Slice(input, begin, size))[0];
  }

  /**
   * Get a subset of the specified input, by specifying the first element and the size of the array.<br>
   * For example, if input is:<br>
   * [a, b, c]<br>
   * [d, e, f]<br>
   * then slice(input, begin=[0,1], size=[2,1] will return:<br>
   * [b]<br>
   * [e]<br>
   * Note that for each dimension i, begin[i] + size[i] <= input.size(i)<br>
   *
   * @param input input Variable to get subset of (NUMERIC type)
   * @param begin Beginning index. Must be same length as rank of input array (INT type)
   * @param size Size of the output array. Must be same length as rank of input array (INT type)
   * @return output Subset of the input (NUMERIC type)
   */
  public INDArray slice(INDArray input, INDArray begin, INDArray size) {
    NDValidation.validateNumerical("slice", "input", input);
    NDValidation.validateInteger("slice", "begin", begin);
    NDValidation.validateInteger("slice", "size", size);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.Slice(input, begin, size))[0];
  }

  /**
   * Squared L2 norm: see norm2(String, SDVariable, boolean, int...)<br>
   *
   * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
   * the mean along a dimension).<br>
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
   * keepDims = true: [a,1,c]<br>
   * keepDims = false: [a,c]<br>
   *
   * @param x  (NUMERIC type)
   * @param keepDims 
   * @param dimensions  (Size: AtLeast(min=0))
   * @return output  (NUMERIC type)
   */
  public INDArray squaredNorm(INDArray x, boolean keepDims, int... dimensions) {
    NDValidation.validateNumerical("squaredNorm", "x", x);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.floating.SquaredNorm(x, keepDims, dimensions));
  }

  /**
   * Squared L2 norm: see norm2(String, SDVariable, boolean, int...)<br>
   *
   * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
   * the mean along a dimension).<br>
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
   * keepDims = true: [a,1,c]<br>
   * keepDims = false: [a,c]<br>
   *
   * @param x  (NUMERIC type)
   * @param dimensions  (Size: AtLeast(min=0))
   * @return output  (NUMERIC type)
   */
  public INDArray squaredNorm(INDArray x, int... dimensions) {
    NDValidation.validateNumerical("squaredNorm", "x", x);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.floating.SquaredNorm(x, false, dimensions));
  }

  /**
   * Remove a single dimension of size 1.<br>
   * For example, if input has shape [a,b,1,c] then squeeze(input, 2) returns an array of shape [a,b,c]<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param axis Size 1 dimension to remove
   * @return output Output variable (NUMERIC type)
   */
  public INDArray squeeze(INDArray x, int axis) {
    NDValidation.validateNumerical("squeeze", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.Squeeze(x, axis))[0];
  }

  /**
   * Stack a set of N INDArray of rank X into one rank X+1 variable.<br>
   * If inputs have shape [a,b,c] then output has shape:<br>
   * axis = 0: [N,a,b,c]<br>
   * axis = 1: [a,N,b,c]<br>
   * axis = 2: [a,b,N,c]<br>
   * axis = 3: [a,b,c,N]<br>
   * see unstack(String[], SDVariable, int, int)<br>
   *
   * @param values Input variables to stack. Must have the same shape for all inputs (NDARRAY type)
   * @param axis Axis to stack on
   * @return output Output variable (NDARRAY type)
   */
  public INDArray stack(int axis, INDArray... values) {
    Preconditions.checkArgument(values.length >= 1, "values has incorrect size/length. Expected: values.length >= 1, got %s", values.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.Stack(values, axis))[0];
  }

  /**
   * Stardard deviation array reduction operation, optionally along specified dimensions<br>
   *
   * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
   * the mean along a dimension).<br>
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
   * keepDims = true: [a,1,c]<br>
   * keepDims = false: [a,c]<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param biasCorrected If true: divide by (N-1) (i.e., sample stdev). If false: divide by N (population stdev)
   * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray standardDeviation(INDArray x, boolean biasCorrected, boolean keepDims,
      int... dimensions) {
    NDValidation.validateNumerical("standardDeviation", "x", x);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.summarystats.StandardDeviation(x, biasCorrected, keepDims, dimensions));
  }

  /**
   * Stardard deviation array reduction operation, optionally along specified dimensions<br>
   *
   * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
   * the mean along a dimension).<br>
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
   * keepDims = true: [a,1,c]<br>
   * keepDims = false: [a,c]<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param biasCorrected If true: divide by (N-1) (i.e., sample stdev). If false: divide by N (population stdev)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray standardDeviation(INDArray x, boolean biasCorrected, int... dimensions) {
    NDValidation.validateNumerical("standardDeviation", "x", x);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.summarystats.StandardDeviation(x, biasCorrected, false, dimensions));
  }

  /**
   * Get a subset of the specified input, by specifying the first element, last element, and the strides.<br>
   * For example, if input is:<br>
   * [a, b, c]<br>
   * [d, e, f]<br>
   * [g, h, i]<br>
   * then stridedSlice(input, begin=[0,1], end=[2,2], strides=[2,1], all masks = 0) will return:<br>
   * [b, c]<br>
   * [h, i]<br>
   *
   * @param in Variable to get subset of (NUMERIC type)
   * @param begin Beginning index (Size: AtLeast(min=1))
   * @param end End index (Size: AtLeast(min=1))
   * @param strides Stride ("step size") for each dimension. For example, stride of 2 means take every second element. (Size: AtLeast(min=1))
   * @param beginMask Bit mask: If the ith bit is set to 1, then the value in the begin long[] is ignored, and a value of 0 is used instead for the beginning index for that dimension
   * @param endMask Bit mask: If the ith bit is set to 1, then the value in the end long[] is ignored, and a value of size(i)-1 is used instead for the end index for that dimension
   * @param ellipsisMask Bit mask: only one non-zero value is allowed here. If a non-zero value is set, then other dimensions are inserted as required at the specified position
   * @param newAxisMask Bit mask: if the ith bit is set to 1, then the begin/end/stride values are ignored, and a size 1 dimension is inserted at this point
   * @param shrinkAxisMask Bit mask: if the ith bit is set to 1, then the begin/end/stride values are ignored, and a size 1 dimension is removed at this point. Note that begin/end/stride values must result in a size 1 output for these dimensions
   * @return output A subset of the input array (NUMERIC type)
   */
  public INDArray stridedSlice(INDArray in, long[] begin, long[] end, long[] strides, int beginMask,
      int endMask, int ellipsisMask, int newAxisMask, int shrinkAxisMask) {
    NDValidation.validateNumerical("stridedSlice", "in", in);
    Preconditions.checkArgument(begin.length >= 1, "begin has incorrect size/length. Expected: begin.length >= 1, got %s", begin.length);
    Preconditions.checkArgument(end.length >= 1, "end has incorrect size/length. Expected: end.length >= 1, got %s", end.length);
    Preconditions.checkArgument(strides.length >= 1, "strides has incorrect size/length. Expected: strides.length >= 1, got %s", strides.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.StridedSlice(in, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask))[0];
  }

  /**
   * Get a subset of the specified input, by specifying the first element, last element, and the strides.<br>
   * For example, if input is:<br>
   * [a, b, c]<br>
   * [d, e, f]<br>
   * [g, h, i]<br>
   * then stridedSlice(input, begin=[0,1], end=[2,2], strides=[2,1], all masks = 0) will return:<br>
   * [b, c]<br>
   * [h, i]<br>
   *
   * @param in Variable to get subset of (NUMERIC type)
   * @param begin Beginning index (Size: AtLeast(min=1))
   * @param end End index (Size: AtLeast(min=1))
   * @param strides Stride ("step size") for each dimension. For example, stride of 2 means take every second element. (Size: AtLeast(min=1))
   * @return output A subset of the input array (NUMERIC type)
   */
  public INDArray stridedSlice(INDArray in, long[] begin, long[] end, long... strides) {
    NDValidation.validateNumerical("stridedSlice", "in", in);
    Preconditions.checkArgument(begin.length >= 1, "begin has incorrect size/length. Expected: begin.length >= 1, got %s", begin.length);
    Preconditions.checkArgument(end.length >= 1, "end has incorrect size/length. Expected: end.length >= 1, got %s", end.length);
    Preconditions.checkArgument(strides.length >= 1, "strides has incorrect size/length. Expected: strides.length >= 1, got %s", strides.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.StridedSlice(in, begin, end, strides, 0, 0, 0, 0, 0))[0];
  }

  /**
   * Sum array reduction operation, optionally along specified dimensions.<br>
   *
   * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
   * the mean along a dimension).<br>
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
   * keepDims = true: [a,1,c]<br>
   * keepDims = false: [a,c]<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param keepDims If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output reduced array of rank (input rank - num dimensions) if keepDims = false, or of rank (input rank) if keepdims = true (NUMERIC type)
   */
  public INDArray sum(INDArray x, boolean keepDims, int... dimensions) {
    NDValidation.validateNumerical("sum", "x", x);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.same.Sum(x, keepDims, dimensions));
  }

  /**
   * Sum array reduction operation, optionally along specified dimensions.<br>
   *
   * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
   * the mean along a dimension).<br>
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
   * keepDims = true: [a,1,c]<br>
   * keepDims = false: [a,c]<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output reduced array of rank (input rank - num dimensions) if keepDims = false, or of rank (input rank) if keepdims = true (NUMERIC type)
   */
  public INDArray sum(INDArray x, int... dimensions) {
    NDValidation.validateNumerical("sum", "x", x);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.same.Sum(x, false, dimensions));
  }

  /**
   * Switch operation<br>
   * Predictate - if false, values are output to left (first) branch/output; if true, to right (second) branch/output<br>
   *
   * @param x Input variable (NDARRAY type)
   * @param predicate Predictate - if false, values are output to left (first) branch/output; if true, to right (second) branch/output (BOOL type)
   */
  public INDArray[] switchOp(INDArray x, INDArray predicate) {
    NDValidation.validateBool("switchOp", "predicate", predicate);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.controlflow.compat.Switch(x, predicate));
  }

  /**
   * //TODO: Ops must be documented.<br>
   *
   * @param x Input variable x (NUMERIC type)
   * @param y Input variable y (NUMERIC type)
   * @param dimensionsX dimensions for first input array (x) (Size: AtLeast(min=1))
   * @param dimensionsY dimensions for second input array (y) (Size: AtLeast(min=1))
   * @param transposeX Transpose x (first argument)
   * @param transposeY Transpose y (second argument)
   * @param transposeZ Transpose result array
   * @return output Output variable (NUMERIC type)
   */
  public INDArray tensorMmul(INDArray x, INDArray y, int[] dimensionsX, int[] dimensionsY,
      boolean transposeX, boolean transposeY, boolean transposeZ) {
    NDValidation.validateNumerical("tensorMmul", "x", x);
    NDValidation.validateNumerical("tensorMmul", "y", y);
    Preconditions.checkArgument(dimensionsX.length >= 1, "dimensionsX has incorrect size/length. Expected: dimensionsX.length >= 1, got %s", dimensionsX.length);
    Preconditions.checkArgument(dimensionsY.length >= 1, "dimensionsY has incorrect size/length. Expected: dimensionsY.length >= 1, got %s", dimensionsY.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.TensorMmul(x, y, dimensionsX, dimensionsY, transposeX, transposeY, transposeZ))[0];
  }

  /**
   * //TODO: Ops must be documented.<br>
   *
   * @param x Input variable x (NUMERIC type)
   * @param y Input variable y (NUMERIC type)
   * @param dimensionsX dimensions for first input array (x) (Size: AtLeast(min=1))
   * @param dimensionsY dimensions for second input array (y) (Size: AtLeast(min=1))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray tensorMmul(INDArray x, INDArray y, int[] dimensionsX, int... dimensionsY) {
    NDValidation.validateNumerical("tensorMmul", "x", x);
    NDValidation.validateNumerical("tensorMmul", "y", y);
    Preconditions.checkArgument(dimensionsX.length >= 1, "dimensionsX has incorrect size/length. Expected: dimensionsX.length >= 1, got %s", dimensionsX.length);
    Preconditions.checkArgument(dimensionsY.length >= 1, "dimensionsY has incorrect size/length. Expected: dimensionsY.length >= 1, got %s", dimensionsY.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.TensorMmul(x, y, dimensionsX, dimensionsY, false, false, false))[0];
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
   *
   * @param x Input variable (NDARRAY type)
   * @param repeat Number of times to repeat in each axis. Must have length equal to the rank of the input array (INT type)
   * @return output Output variable (NDARRAY type)
   */
  public INDArray tile(INDArray x, INDArray repeat) {
    NDValidation.validateInteger("tile", "repeat", repeat);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.Tile(x, repeat))[0];
  }

  /**
   * see tile(String, SDVariable, int...)<br>
   *
   * @param x  (NDARRAY type)
   * @param repeat  (Size: AtLeast(min=1))
   * @return output  (NDARRAY type)
   */
  public INDArray tile(INDArray x, int... repeat) {
    Preconditions.checkArgument(repeat.length >= 1, "repeat has incorrect size/length. Expected: repeat.length >= 1, got %s", repeat.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.Tile(x, repeat))[0];
  }

  /**
   * Matrix transpose operation: If input has shape [a,b] output has shape [b,a]<br>
   *
   * @param x Input variable (NDARRAY type)
   * @return output transposed input (NDARRAY type)
   */
  public INDArray transpose(INDArray x) {
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.Transpose(x))[0];
  }

  /**
   * Unsorted segment max operation. As per segmentMax(String, SDVariable, SDVariable) but without<br>
   * the requirement for the indices to be sorted.<br>
   * If data =     [1, 3, 2, 6, 4, 9, 8]<br>
   * segmentIds =  [1, 0, 2, 0, 1, 1, 2]<br>
   * then output = [6, 9, 8] = [max(3,6), max(1,4,9), max(2,8)]<br>
   *
   * @param data Data (variable) to perform unsorted segment max on (NUMERIC type)
   * @param segmentIds Variable for the segment IDs (NUMERIC type)
   * @param numSegments Number of segments
   * @return output Unsorted segment output (NUMERIC type)
   */
  public INDArray unsortedSegmentMax(INDArray data, INDArray segmentIds, int numSegments) {
    NDValidation.validateNumerical("unsortedSegmentMax", "data", data);
    NDValidation.validateNumerical("unsortedSegmentMax", "segmentIds", segmentIds);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentMax(data, segmentIds, numSegments))[0];
  }

  /**
   * Unsorted segment mean operation. As per segmentMean(String, SDVariable, SDVariable) but without<br>
   * the requirement for the indices to be sorted.<br>
   * If data =     [1, 3, 2, 6, 4, 9, 8]<br>
   * segmentIds =  [1, 0, 2, 0, 1, 1, 2]<br>
   * then output = [4.5, 4.666, 5] = [mean(3,6), mean(1,4,9), mean(2,8)]<br>
   *
   * @param data Data (variable) to perform unsorted segment max on (NUMERIC type)
   * @param segmentIds Variable for the segment IDs (NUMERIC type)
   * @param numSegments Number of segments
   * @return output Unsorted segment output (NUMERIC type)
   */
  public INDArray unsortedSegmentMean(INDArray data, INDArray segmentIds, int numSegments) {
    NDValidation.validateNumerical("unsortedSegmentMean", "data", data);
    NDValidation.validateNumerical("unsortedSegmentMean", "segmentIds", segmentIds);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentMean(data, segmentIds, numSegments))[0];
  }

  /**
   * Unsorted segment min operation. As per segmentMin(String, SDVariable, SDVariable) but without<br>
   * the requirement for the indices to be sorted.<br>
   * If data =     [1, 3, 2, 6, 4, 9, 8]<br>
   * segmentIds =  [1, 0, 2, 0, 1, 1, 2]<br>
   * then output = [3, 1, 2] = [min(3,6), min(1,4,9), min(2,8)]<br>
   *
   * @param data Data (variable) to perform unsorted segment max on (NUMERIC type)
   * @param segmentIds Variable for the segment IDs (NUMERIC type)
   * @param numSegments Number of segments
   * @return output Unsorted segment output (NUMERIC type)
   */
  public INDArray unsortedSegmentMin(INDArray data, INDArray segmentIds, int numSegments) {
    NDValidation.validateNumerical("unsortedSegmentMin", "data", data);
    NDValidation.validateNumerical("unsortedSegmentMin", "segmentIds", segmentIds);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentMin(data, segmentIds, numSegments))[0];
  }

  /**
   * Unsorted segment product operation. As per segmentProd(String, SDVariable, SDVariable) but without<br>
   * the requirement for the indices to be sorted.<br>
   * If data =     [1, 3, 2, 6, 4, 9, 8]<br>
   * segmentIds =  [1, 0, 2, 0, 1, 1, 2]<br>
   * then output = [4.5, 4.666, 5] = [mean(3,6), mean(1,4,9), mean(2,8)]<br>
   *
   * @param data Data (variable) to perform unsorted segment max on (NUMERIC type)
   * @param segmentIds Variable for the segment IDs (NUMERIC type)
   * @param numSegments Number of segments
   * @return output Unsorted segment output (NUMERIC type)
   */
  public INDArray unsortedSegmentProd(INDArray data, INDArray segmentIds, int numSegments) {
    NDValidation.validateNumerical("unsortedSegmentProd", "data", data);
    NDValidation.validateNumerical("unsortedSegmentProd", "segmentIds", segmentIds);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentProd(data, segmentIds, numSegments))[0];
  }

  /**
   * Unsorted segment sqrtN operation. Simply returns the sqrt of the count of the number of values in each segment<br>
   * If data =     [1, 3, 2, 6, 4, 9, 8]<br>
   * segmentIds =  [1, 0, 2, 0, 1, 1, 2]<br>
   * then output = [1.414, 1.732, 1.414] = [sqrt(2), sqrtN(3), sqrtN(2)]<br>
   *
   * @param data Data (variable) to perform unsorted segment max on (NUMERIC type)
   * @param segmentIds Variable for the segment IDs (NUMERIC type)
   * @param numSegments Number of segments
   * @return output Unsorted segment output (NUMERIC type)
   */
  public INDArray unsortedSegmentSqrtN(INDArray data, INDArray segmentIds, int numSegments) {
    NDValidation.validateNumerical("unsortedSegmentSqrtN", "data", data);
    NDValidation.validateNumerical("unsortedSegmentSqrtN", "segmentIds", segmentIds);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentSqrtN(data, segmentIds, numSegments))[0];
  }

  /**
   * Unsorted segment sum operation. As per segmentSum(String, SDVariable, SDVariable) but without<br>
   * the requirement for the indices to be sorted.<br>
   * If data =     [1, 3, 2, 6, 4, 9, 8]<br>
   * segmentIds =  [1, 0, 2, 0, 1, 1, 2]<br>
   * then output = [9, 14, 10] = [sum(3,6), sum(1,4,9), sum(2,8)]<br>
   *
   * @param data Data (variable) to perform unsorted segment max on (NUMERIC type)
   * @param segmentIds Variable for the segment IDs (NUMERIC type)
   * @param numSegments Number of segments
   * @return output Unsorted segment output (NUMERIC type)
   */
  public INDArray unsortedSegmentSum(INDArray data, INDArray segmentIds, int numSegments) {
    NDValidation.validateNumerical("unsortedSegmentSum", "data", data);
    NDValidation.validateNumerical("unsortedSegmentSum", "segmentIds", segmentIds);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentSum(data, segmentIds, numSegments))[0];
  }

  /**
   * Unstack a variable of rank X into N rank X-1 variables by taking slices along the specified axis.<br>
   * If input has shape [a,b,c] then output has shape:<br>
   * axis = 0: [b,c]<br>
   * axis = 1: [a,c]<br>
   * axis = 2: [a,b]<br>
   *
   * @param value Input variable to unstack (NDARRAY type)
   * @param axis Axis to unstack on
   * @param num Number of output variables
   */
  public INDArray[] unstack(INDArray value, int axis, int num) {
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.Unstack(value, axis, num));
  }

  /**
   * Variance array reduction operation, optionally along specified dimensions<br>
   *
   * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
   * the mean along a dimension).<br>
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
   * keepDims = true: [a,1,c]<br>
   * keepDims = false: [a,c]<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param biasCorrected If true: divide by (N-1) (i.e., sample variable). If false: divide by N (population variance)
   * @param keepDims If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray variance(INDArray x, boolean biasCorrected, boolean keepDims, int... dimensions) {
    NDValidation.validateNumerical("variance", "x", x);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.summarystats.Variance(x, biasCorrected, keepDims, dimensions));
  }

  /**
   * Variance array reduction operation, optionally along specified dimensions<br>
   *
   * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
   * the mean along a dimension).<br>
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
   * keepDims = true: [a,1,c]<br>
   * keepDims = false: [a,c]<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param biasCorrected If true: divide by (N-1) (i.e., sample variable). If false: divide by N (population variance)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray variance(INDArray x, boolean biasCorrected, int... dimensions) {
    NDValidation.validateNumerical("variance", "x", x);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.summarystats.Variance(x, biasCorrected, false, dimensions));
  }

  /**
   * Return a variable of all 0s, with the same shape as the input variable. Note that this is dynamic:<br>
   * if the input shape changes in later execution, the returned variable's shape will also be updated<br>
   *
   * @param input Input  (NUMERIC type)
   * @return output A new Variable with the same (dynamic) shape as the input (NUMERIC type)
   */
  public INDArray zerosLike(INDArray input) {
    NDValidation.validateNumerical("zerosLike", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.ZerosLike(input))[0];
  }
}

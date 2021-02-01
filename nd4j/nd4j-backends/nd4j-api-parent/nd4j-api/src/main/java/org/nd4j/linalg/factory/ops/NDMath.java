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

package org.nd4j.linalg.factory.ops;

import org.nd4j.common.base.Preconditions;
import org.nd4j.enums.PartitionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDValidation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Condition;

public class NDMath {
  public NDMath() {
  }

  /**
   * Clips tensor values to a maximum average L2-norm.<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param clipValue Value for clipping
   * @param dimensions Dimensions to reduce over (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray clipByAvgNorm(INDArray x, double clipValue, int... dimensions) {
    NDValidation.validateNumerical("ClipByAvgNorm", "x", x);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.clip.ClipByAvgNorm(x, clipValue, dimensions))[0];
  }

  /**
   * Looks up ids in a list of embedding tensors.<br>
   *
   * @param x Input tensor (NUMERIC type)
   * @param indices A Tensor containing the ids to be looked up. (INT type)
   * @param PartitionMode partition_mode == 0 - i.e. 'mod' , 1 - 'div'
   * @return output Shifted output (NUMERIC type)
   */
  public INDArray embeddingLookup(INDArray x, INDArray indices, PartitionMode PartitionMode) {
    NDValidation.validateNumerical("EmbeddingLookup", "x", x);
    NDValidation.validateInteger("EmbeddingLookup", "indices", indices);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.tensorops.EmbeddingLookup(x, indices, PartitionMode))[0];
  }

  /**
   * Return array of max elements indices with along tensor dimensions <br>
   *
   * @param x Input tensor (NUMERIC type)
   * @param dataType Data type
   * @return output Array max elements indices with along dimensions. (INT type)
   */
  public INDArray mergeMaxIndex(INDArray[] x, DataType dataType) {
    NDValidation.validateNumerical("MergeMaxIndex", "x", x);
    Preconditions.checkArgument(x.length >= 1, "x has incorrect size/length. Expected: x.length >= 1, got %s", x.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.MergeMaxIndex(x, dataType))[0];
  }

  /**
   * Return array of max elements indices with along tensor dimensions <br>
   *
   * @param x Input tensor (NUMERIC type)
   * @return output Array max elements indices with along dimensions. (INT type)
   */
  public INDArray mergeMaxIndex(INDArray... x) {
    NDValidation.validateNumerical("MergeMaxIndex", "x", x);
    Preconditions.checkArgument(x.length >= 1, "x has incorrect size/length. Expected: x.length >= 1, got %s", x.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.MergeMaxIndex(x, DataType.INT))[0];
  }

  /**
   * Elementwise absolute value operation: out = abs(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray abs(INDArray x) {
    NDValidation.validateNumerical("abs", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.same.Abs(x));
  }

  /**
   * Elementwise acos (arccosine, inverse cosine) operation: out = arccos(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray acos(INDArray x) {
    NDValidation.validateNumerical("acos", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.ACos(x));
  }

  /**
   * Elementwise acosh (inverse hyperbolic cosine) function: out = acosh(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray acosh(INDArray x) {
    NDValidation.validateNumerical("acosh", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.ACosh(x));
  }

  /**
   * Pairwise addition operation, out = x + y<br>
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]<br>
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param y Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray add(INDArray x, INDArray y) {
    NDValidation.validateNumerical("add", "x", x);
    NDValidation.validateNumerical("add", "y", y);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.AddOp(x, y))[0];
  }

  /**
   * Scalar add operation, out = in + scalar<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param value Scalar value for op
   * @return output Output variable (NUMERIC type)
   */
  public INDArray add(INDArray x, double value) {
    NDValidation.validateNumerical("add", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.scalar.ScalarAdd(x, value));
  }

  /**
   * Absolute max array reduction operation, optionally along specified dimensions: out = max(abs(x))<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray amax(INDArray in, int... dimensions) {
    NDValidation.validateNumerical("amax", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.same.AMax(in, dimensions));
  }

  /**
   * Absolute mean array reduction operation, optionally along specified dimensions: out = mean(abs(x))<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray amean(INDArray in, int... dimensions) {
    NDValidation.validateNumerical("amean", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.floating.AMean(in, dimensions));
  }

  /**
   * Absolute min array reduction operation, optionally along specified dimensions: out = min(abs(x))<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray amin(INDArray in, int... dimensions) {
    NDValidation.validateNumerical("amin", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.same.AMin(in, dimensions));
  }

  /**
   * Boolean AND operation: elementwise (x != 0) && (y != 0)<br>
   * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
   * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.<br>
   *
   * @param x Input 1 (BOOL type)
   * @param y Input 2 (BOOL type)
   * @return output INDArray with values 0 and 1 based on where the condition is satisfied (BOOL type)
   */
  public INDArray and(INDArray x, INDArray y) {
    NDValidation.validateBool("and", "x", x);
    NDValidation.validateBool("and", "y", y);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool.And(x, y));
  }

  /**
   * Elementwise asin (arcsin, inverse sine) operation: out = arcsin(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray asin(INDArray x) {
    NDValidation.validateNumerical("asin", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.ASin(x));
  }

  /**
   * Elementwise asinh (inverse hyperbolic sine) function: out = asinh(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray asinh(INDArray x) {
    NDValidation.validateNumerical("asinh", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.ASinh(x));
  }

  /**
   * Absolute sum array reduction operation, optionally along specified dimensions: out = sum(abs(x))<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray asum(INDArray in, int... dimensions) {
    NDValidation.validateNumerical("asum", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.same.ASum(in, dimensions));
  }

  /**
   * Elementwise atan (arctangent, inverse tangent) operation: out = arctangent(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray atan(INDArray x) {
    NDValidation.validateNumerical("atan", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.ATan(x));
  }

  /**
   * Elementwise atan (arctangent, inverse tangent) operation: out = atan2(x,y).<br>
   * Similar to atan(y/x) but sigts of x and y are used to determine the location of the result<br>
   *
   * @param y Input Y variable (NUMERIC type)
   * @param x Input X variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray atan2(INDArray y, INDArray x) {
    NDValidation.validateNumerical("atan2", "y", y);
    NDValidation.validateNumerical("atan2", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.ATan2(y, x))[0];
  }

  /**
   * Elementwise atanh (inverse hyperbolic tangent) function: out = atanh(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray atanh(INDArray x) {
    NDValidation.validateNumerical("atanh", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.ATanh(x));
  }

  /**
   * Bit shift operation<br>
   *
   * @param x input (NUMERIC type)
   * @param shift shift value (NUMERIC type)
   * @return output shifted output (NUMERIC type)
   */
  public INDArray bitShift(INDArray x, INDArray shift) {
    NDValidation.validateNumerical("bitShift", "x", x);
    NDValidation.validateNumerical("bitShift", "shift", shift);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.ShiftBits(x, shift))[0];
  }

  /**
   * Right bit shift operation<br>
   *
   * @param x Input tensor (NUMERIC type)
   * @param shift shift argument (NUMERIC type)
   * @return output shifted output (NUMERIC type)
   */
  public INDArray bitShiftRight(INDArray x, INDArray shift) {
    NDValidation.validateNumerical("bitShiftRight", "x", x);
    NDValidation.validateNumerical("bitShiftRight", "shift", shift);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.RShiftBits(x, shift))[0];
  }

  /**
   * Cyclic bit shift operation<br>
   *
   * @param x Input tensor (NUMERIC type)
   * @param shift shift argy=ument (NUMERIC type)
   * @return output shifted output (NUMERIC type)
   */
  public INDArray bitShiftRotl(INDArray x, INDArray shift) {
    NDValidation.validateNumerical("bitShiftRotl", "x", x);
    NDValidation.validateNumerical("bitShiftRotl", "shift", shift);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.CyclicShiftBits(x, shift))[0];
  }

  /**
   * Cyclic right shift operation<br>
   *
   * @param x Input tensor (NUMERIC type)
   * @param shift Shift argument (NUMERIC type)
   * @return output Shifted output (NUMERIC type)
   */
  public INDArray bitShiftRotr(INDArray x, INDArray shift) {
    NDValidation.validateNumerical("bitShiftRotr", "x", x);
    NDValidation.validateNumerical("bitShiftRotr", "shift", shift);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.CyclicRShiftBits(x, shift))[0];
  }

  /**
   * Element-wise ceiling function: out = ceil(x).<br>
   * Rounds each value up to the nearest integer value (if not already an integer)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray ceil(INDArray x) {
    NDValidation.validateNumerical("ceil", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.same.Ceil(x));
  }

  /**
   * Clipping by L2 norm, optionally along dimension(s)<br>
   * if l2Norm(x,dimension) < clipValue, then input is returned unmodifed<br>
   * Otherwise, out[i] = in[i] * clipValue / l2Norm(in, dimensions) where each value is clipped according<br>
   * to the corresponding l2Norm along the specified dimensions<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param clipValue Clipping value (maximum l2 norm)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray clipByNorm(INDArray x, double clipValue, int... dimensions) {
    NDValidation.validateNumerical("clipByNorm", "x", x);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.clip.ClipByNorm(x, clipValue, dimensions))[0];
  }

  /**
   * Element-wise clipping function:<br>
   * out[i] = in[i] if in[i] >= clipValueMin and in[i] <= clipValueMax<br>
   * out[i] = clipValueMin if in[i] < clipValueMin<br>
   * out[i] = clipValueMax if in[i] > clipValueMax<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param clipValueMin Minimum value for clipping
   * @param clipValueMax Maximum value for clipping
   * @return output Output variable (NUMERIC type)
   */
  public INDArray clipByValue(INDArray x, double clipValueMin, double clipValueMax) {
    NDValidation.validateNumerical("clipByValue", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.clip.ClipByValue(x, clipValueMin, clipValueMax))[0];
  }

  /**
   * Compute the 2d confusion matrix of size [numClasses, numClasses] from a pair of labels and predictions, both of<br>
   * which are represented as integer values. This version assumes the number of classes is 1 + max(max(labels), max(pred))<br>
   * For example, if labels = [0, 1, 1] and predicted = [0, 2, 1] then output is:<br>
   * [1, 0, 0]<br>
   * [0, 1, 1]<br>
   * [0, 0, 0]<br>
   *
   * @param labels Labels - 1D array of integer values representing label values (NUMERIC type)
   * @param pred Predictions - 1D array of integer values representing predictions. Same length as labels (NUMERIC type)
   * @param dataType Data type
   * @return output variable (2D, shape [numClasses, numClasses}) (NUMERIC type)
   */
  public INDArray confusionMatrix(INDArray labels, INDArray pred, DataType dataType) {
    NDValidation.validateNumerical("confusionMatrix", "labels", labels);
    NDValidation.validateNumerical("confusionMatrix", "pred", pred);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.ConfusionMatrix(labels, pred, dataType))[0];
  }

  /**
   * Compute the 2d confusion matrix of size [numClasses, numClasses] from a pair of labels and predictions, both of<br>
   * which are represented as integer values.<br>
   * For example, if labels = [0, 1, 1], predicted = [0, 2, 1], and numClasses=4 then output is:<br>
   * [1, 0, 0, 0]<br>
   * [0, 1, 1, 0]<br>
   * [0, 0, 0, 0]<br>
   * [0, 0, 0, 0]<br>
   *
   * @param labels Labels - 1D array of integer values representing label values (NUMERIC type)
   * @param pred Predictions - 1D array of integer values representing predictions. Same length as labels (NUMERIC type)
   * @param numClasses Number of classes
   * @return output variable (2D, shape [numClasses, numClasses}) (NUMERIC type)
   */
  public INDArray confusionMatrix(INDArray labels, INDArray pred, int numClasses) {
    NDValidation.validateNumerical("confusionMatrix", "labels", labels);
    NDValidation.validateNumerical("confusionMatrix", "pred", pred);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.ConfusionMatrix(labels, pred, numClasses))[0];
  }

  /**
   * Compute the 2d confusion matrix of size [numClasses, numClasses] from a pair of labels and predictions, both of<br>
   * which are represented as integer values. This version assumes the number of classes is 1 + max(max(labels), max(pred))<br>
   * For example, if labels = [0, 1, 1], predicted = [0, 2, 1] and weights = [1, 2, 3]<br>
   * [1, 0, 0]<br>
   * [0, 3, 2]<br>
   * [0, 0, 0]<br>
   *
   * @param labels Labels - 1D array of integer values representing label values (NUMERIC type)
   * @param pred Predictions - 1D array of integer values representing predictions. Same length as labels (NUMERIC type)
   * @param weights Weights - 1D array of values (may be real/decimal) representing the weight/contribution of each prediction. Must be same length as both labels and predictions arrays (NUMERIC type)
   * @return output variable (2D, shape [numClasses, numClasses}) (NUMERIC type)
   */
  public INDArray confusionMatrix(INDArray labels, INDArray pred, INDArray weights) {
    NDValidation.validateNumerical("confusionMatrix", "labels", labels);
    NDValidation.validateNumerical("confusionMatrix", "pred", pred);
    NDValidation.validateNumerical("confusionMatrix", "weights", weights);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.ConfusionMatrix(labels, pred, weights))[0];
  }

  /**
   * Compute the 2d confusion matrix of size [numClasses, numClasses] from a pair of labels and predictions, both of<br>
   * which are represented as integer values.<br>
   * For example, if labels = [0, 1, 1], predicted = [0, 2, 1], numClasses = 4, and weights = [1, 2, 3]<br>
   * [1, 0, 0, 0]<br>
   * [0, 3, 2, 0]<br>
   * [0, 0, 0, 0]<br>
   * [0, 0, 0, 0]<br>
   *
   * @param labels Labels - 1D array of integer values representing label values (NUMERIC type)
   * @param pred Predictions - 1D array of integer values representing predictions. Same length as labels (NUMERIC type)
   * @param weights Weights - 1D array of values (may be real/decimal) representing the weight/contribution of each prediction. Must be same length as both labels and predictions arrays (NUMERIC type)
   * @param numClasses 
   * @return output Output variable (2D, shape [numClasses, numClasses}) (NUMERIC type)
   */
  public INDArray confusionMatrix(INDArray labels, INDArray pred, INDArray weights,
      int numClasses) {
    NDValidation.validateNumerical("confusionMatrix", "labels", labels);
    NDValidation.validateNumerical("confusionMatrix", "pred", pred);
    NDValidation.validateNumerical("confusionMatrix", "weights", weights);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.ConfusionMatrix(labels, pred, weights, numClasses))[0];
  }

  /**
   * Elementwise cosine operation: out = cos(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray cos(INDArray x) {
    NDValidation.validateNumerical("cos", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.Cos(x));
  }

  /**
   * Elementwise cosh (hyperbolic cosine) operation: out = cosh(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray cosh(INDArray x) {
    NDValidation.validateNumerical("cosh", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.Cosh(x));
  }

  /**
   * Cosine distance reduction operation. The output contains the cosine distance for each<br>
   * tensor/subset along the specified dimensions:<br>
   * out = 1.0 - cosineSimilarity(x,y)<br>
   *
   * @param x Input variable x (NUMERIC type)
   * @param y Input variable y (NUMERIC type)
   * @param dimensions Dimensions to calculate cosineDistance over (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray cosineDistance(INDArray x, INDArray y, int... dimensions) {
    NDValidation.validateNumerical("cosineDistance", "x", x);
    NDValidation.validateNumerical("cosineDistance", "y", y);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce3.CosineDistance(x, y, dimensions));
  }

  /**
   * Cosine similarity pairwise reduction operation. The output contains the cosine similarity for each tensor/subset<br>
   * along the specified dimensions:<br>
   * out = (sum_i x[i] * y[i]) / ( sqrt(sum_i x[i]^2) * sqrt(sum_i y[i]^2)<br>
   *
   * @param x Input variable x (NUMERIC type)
   * @param y Input variable y (NUMERIC type)
   * @param dimensions Dimensions to calculate cosineSimilarity over (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray cosineSimilarity(INDArray x, INDArray y, int... dimensions) {
    NDValidation.validateNumerical("cosineSimilarity", "x", x);
    NDValidation.validateNumerical("cosineSimilarity", "y", y);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce3.CosineSimilarity(x, y, dimensions));
  }

  /**
   * Count non zero array reduction operation, optionally along specified dimensions: out = count(x != 0)<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray countNonZero(INDArray in, int... dimensions) {
    NDValidation.validateNumerical("countNonZero", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.longer.CountNonZero(in, dimensions));
  }

  /**
   * Count zero array reduction operation, optionally along specified dimensions: out = count(x == 0)<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray countZero(INDArray in, int... dimensions) {
    NDValidation.validateNumerical("countZero", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.longer.CountZero(in, dimensions));
  }

  /**
   * Returns the pair-wise cross product of equal size arrays a and b: a x b = ||a||x||b|| sin(theta).<br>
   * Can take rank 1 or above inputs (of equal shapes), but note that the last dimension must have dimension 3<br>
   *
   * @param a First input (NUMERIC type)
   * @param b Second input (NUMERIC type)
   * @return output Element-wise cross product (NUMERIC type)
   */
  public INDArray cross(INDArray a, INDArray b) {
    NDValidation.validateNumerical("cross", "a", a);
    NDValidation.validateNumerical("cross", "b", b);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.Cross(a, b))[0];
  }

  /**
   * Element-wise cube function: out = x^3<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray cube(INDArray x) {
    NDValidation.validateNumerical("cube", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.same.Cube(x));
  }

  /**
   * Returns an output variable with diagonal values equal to the specified values; off-diagonal values will be set to 0<br>
   * For example, if input = [1,2,3], then output is given by:<br>
   * [ 1, 0, 0]<br>
   * [ 0, 2, 0]<br>
   * [ 0, 0, 3]<br>
   * <br>
   * Higher input ranks are also supported: if input has shape [a,...,R-1] then output[i,...,k,i,...,k] = input[i,...,k].<br>
   * i.e., for input rank R, output has rank 2R<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray diag(INDArray x) {
    NDValidation.validateNumerical("diag", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.Diag(x))[0];
  }

  /**
   * Extract the diagonal part from the input array.<br>
   * If input is<br>
   * [ 1, 0, 0]<br>
   * [ 0, 2, 0]<br>
   * [ 0, 0, 3]<br>
   * then output is [1, 2, 3].<br>
   * Supports higher dimensions: in general, out[i,...,k] = in[i,...,k,i,...,k]<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Diagonal part of the input (NUMERIC type)
   */
  public INDArray diagPart(INDArray x) {
    NDValidation.validateNumerical("diagPart", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.DiagPart(x))[0];
  }

  /**
   * Pairwise division operation, out = x / y<br>
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]<br>
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param y Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray div(INDArray x, INDArray y) {
    NDValidation.validateNumerical("div", "x", x);
    NDValidation.validateNumerical("div", "y", y);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.DivOp(x, y))[0];
  }

  /**
   * Scalar division operation, out = in / scalar<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param value Scalar value for op
   * @return output Output variable (NUMERIC type)
   */
  public INDArray div(INDArray x, double value) {
    NDValidation.validateNumerical("div", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.scalar.ScalarDivision(x, value));
  }

  /**
   * Entropy reduction: -sum(x * log(x))<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray entropy(INDArray in, int... dimensions) {
    NDValidation.validateNumerical("entropy", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.floating.Entropy(in, dimensions));
  }

  /**
   * Element-wise Gaussian error function - out = erf(in)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray erf(INDArray x) {
    NDValidation.validateNumerical("erf", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.Erf(x));
  }

  /**
   * Element-wise complementary Gaussian error function - out = erfc(in) = 1 - erf(in)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray erfc(INDArray x) {
    NDValidation.validateNumerical("erfc", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.Erfc(x));
  }

  /**
   * Euclidean distance (l2 norm, l2 distance) reduction operation. The output contains the Euclidean distance for each<br>
   * tensor/subset along the specified dimensions:<br>
   * out = sqrt( sum_i (x[i] - y[i])^2 )<br>
   *
   * @param x Input variable x (NUMERIC type)
   * @param y Input variable y (NUMERIC type)
   * @param dimensions Dimensions to calculate euclideanDistance over (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray euclideanDistance(INDArray x, INDArray y, int... dimensions) {
    NDValidation.validateNumerical("euclideanDistance", "x", x);
    NDValidation.validateNumerical("euclideanDistance", "y", y);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce3.EuclideanDistance(x, y, dimensions));
  }

  /**
   * Elementwise exponent function: out = exp(x) = 2.71828...^x<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray exp(INDArray x) {
    NDValidation.validateNumerical("exp", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.Exp(x));
  }

  /**
   * Elementwise 1.0 - exponent function: out = 1.0 - exp(x) = 1.0 - 2.71828...^x<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray expm1(INDArray x) {
    NDValidation.validateNumerical("expm1", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.Expm1(x));
  }

  /**
   * Generate an identity matrix with the specified number of rows and columns.<br>
   *
   * @param rows Number of rows
   * @return output Identity matrix (NUMERIC type)
   */
  public INDArray eye(int rows) {
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.Eye(rows))[0];
  }

  /**
   * As per eye(String, int, int, DataType) but with the default datatype, Eye.DEFAULT_DTYPE<br>
   *
   * @param rows Number of rows
   * @param cols Number of columns
   * @return output  (NUMERIC type)
   */
  public INDArray eye(int rows, int cols) {
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.Eye(rows, cols))[0];
  }

  /**
   * Generate an identity matrix with the specified number of rows and columns<br>
   * Example:<br>
   * <pre><br>
   * {@code INDArray eye = eye(3,2)<br>
   * eye:<br>
   * [ 1, 0]<br>
   * [ 0, 1]<br>
   * [ 0, 0]}<br>
   * </pre><br>
   *
   * @param rows Number of rows
   * @param cols Number of columns
   * @param dataType Data type
   * @param dimensions  (Size: AtLeast(min=0))
   * @return output Identity matrix (NUMERIC type)
   */
  public INDArray eye(int rows, int cols, DataType dataType, int... dimensions) {
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.Eye(rows, cols, dataType, dimensions))[0];
  }

  /**
   * As per eye(int, int) bit with the number of rows/columns specified as scalar INDArrays<br>
   *
   * @param rows Number of rows (INT type)
   * @param cols Number of columns (INT type)
   * @return output Identity matrix (NUMERIC type)
   */
  public INDArray eye(INDArray rows, INDArray cols) {
    NDValidation.validateInteger("eye", "rows", rows);
    NDValidation.validateInteger("eye", "cols", cols);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.Eye(rows, cols))[0];
  }

  /**
   * As per eye(String, int) but with the number of rows specified as a scalar INDArray<br>
   *
   * @param rows Number of rows (INT type)
   * @return output SDVaribable identity matrix (NUMERIC type)
   */
  public INDArray eye(INDArray rows) {
    NDValidation.validateInteger("eye", "rows", rows);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.Eye(rows))[0];
  }

  /**
   * First index reduction operation.<br>
   * Returns a variable that contains the index of the first element that matches the specified condition (for each<br>
   * slice along the specified dimensions)<br>
   * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
   * the mean along a dimension).<br>
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
   * keepDims = true: [a,1,c]<br>
   * keepDims = false: [a,c]<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param condition Condition to check on input variable
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=1))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray firstIndex(INDArray in, Condition condition, int... dimensions) {
    NDValidation.validateNumerical("firstIndex", "in", in);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.indexaccum.FirstIndex(in, false, condition, dimensions));
  }

  /**
   * First index reduction operation.<br>
   * Returns a variable that contains the index of the first element that matches the specified condition (for each<br>
   * slice along the specified dimensions)<br>
   * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
   * the mean along a dimension).<br>
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
   * keepDims = true: [a,1,c]<br>
   * keepDims = false: [a,c]<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param condition Condition to check on input variable
   * @param keepDims If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=1))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray firstIndex(INDArray in, Condition condition, boolean keepDims,
      int... dimensions) {
    NDValidation.validateNumerical("firstIndex", "in", in);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.indexaccum.FirstIndex(in, keepDims, condition, dimensions));
  }

  /**
   * Element-wise floor function: out = floor(x).<br>
   * Rounds each value down to the nearest integer value (if not already an integer)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray floor(INDArray x) {
    NDValidation.validateNumerical("floor", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.same.Floor(x));
  }

  /**
   * Pairwise floor division operation, out = floor(x / y)<br>
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]<br>
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param y Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray floorDiv(INDArray x, INDArray y) {
    NDValidation.validateNumerical("floorDiv", "x", x);
    NDValidation.validateNumerical("floorDiv", "y", y);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.FloorDivOp(x, y))[0];
  }

  /**
   * Pairwise Modulus division operation<br>
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]<br>
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param y Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray floorMod(INDArray x, INDArray y) {
    NDValidation.validateNumerical("floorMod", "x", x);
    NDValidation.validateNumerical("floorMod", "y", y);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.FloorModOp(x, y))[0];
  }

  /**
   * Scalar floor modulus operation<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param value Scalar value for op
   * @return output Output variable (NUMERIC type)
   */
  public INDArray floorMod(INDArray x, double value) {
    NDValidation.validateNumerical("floorMod", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.scalar.ScalarFMod(x, value));
  }

  /**
   * Hamming distance reduction operation. The output contains the cosine distance for each<br>
   * tensor/subset along the specified dimensions:<br>
   * out = count( x[i] != y[i] )<br>
   *
   * @param x Input variable x (NUMERIC type)
   * @param y Input variable y (NUMERIC type)
   * @param dimensions Dimensions to calculate hammingDistance over (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray hammingDistance(INDArray x, INDArray y, int... dimensions) {
    NDValidation.validateNumerical("hammingDistance", "x", x);
    NDValidation.validateNumerical("hammingDistance", "y", y);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce3.HammingDistance(x, y, dimensions));
  }

  /**
   * Index of the max absolute value: argmax(abs(in))<br>
   * see argmax(String, INDArray, boolean, int...)<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=1))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray iamax(INDArray in, int... dimensions) {
    NDValidation.validateNumerical("iamax", "in", in);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.indexaccum.custom.ArgMax(in, false, dimensions))[0];
  }

  /**
   * Index of the max absolute value: argmax(abs(in))<br>
   * see argmax(String, INDArray, boolean, int...)<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param keepDims If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=1))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray iamax(INDArray in, boolean keepDims, int... dimensions) {
    NDValidation.validateNumerical("iamax", "in", in);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.indexaccum.custom.ArgMax(in, keepDims, dimensions))[0];
  }

  /**
   * Index of the min absolute value: argmin(abs(in))<br>
   * see argmin(String, INDArray, boolean, int...)<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=1))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray iamin(INDArray in, int... dimensions) {
    NDValidation.validateNumerical("iamin", "in", in);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.indexaccum.custom.ArgMin(in, false, dimensions))[0];
  }

  /**
   * Index of the min absolute value: argmin(abs(in))<br>
   * see argmin(String, INDArray, boolean, int...)<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param keepDims If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=1))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray iamin(INDArray in, boolean keepDims, int... dimensions) {
    NDValidation.validateNumerical("iamin", "in", in);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.indexaccum.custom.ArgMin(in, keepDims, dimensions))[0];
  }

  /**
   * Is finite operation: elementwise isFinite(x)<br>
   * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or<br>
   * value 0 otherwise<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray isFinite(INDArray x) {
    NDValidation.validateNumerical("isFinite", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.bool.IsFinite(x));
  }

  /**
   * Is infinite operation: elementwise isInfinite(x)<br>
   * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or<br>
   * value 0 otherwise<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray isInfinite(INDArray x) {
    NDValidation.validateNumerical("isInfinite", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.bool.IsInf(x));
  }

  /**
   * Is maximum operation: elementwise x == max(x)<br>
   * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or<br>
   * value 0 otherwise<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray isMax(INDArray x) {
    NDValidation.validateNumerical("isMax", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.any.IsMax(x))[0];
  }

  /**
   * Is Not a Number operation: elementwise isNaN(x)<br>
   * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or<br>
   * value 0 otherwise<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray isNaN(INDArray x) {
    NDValidation.validateNumerical("isNaN", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.bool.IsNaN(x));
  }

  /**
   * Is the array non decreasing?<br>
   * An array is non-decreasing if for every valid i, x[i] <= x[i+1]. For Rank 2+ arrays, values are compared<br>
   * in 'c' (row major) order<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Scalar variable with value 1 if non-decreasing, or 0 otherwise (NUMERIC type)
   */
  public INDArray isNonDecreasing(INDArray x) {
    NDValidation.validateNumerical("isNonDecreasing", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.IsNonDecreasing(x))[0];
  }

  /**
   * Is the array strictly increasing?<br>
   * An array is strictly increasing if for every valid i, x[i] < x[i+1]. For Rank 2+ arrays, values are compared<br>
   * in 'c' (row major) order<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Scalar variable with value 1 if strictly increasing, or 0 otherwise (NUMERIC type)
   */
  public INDArray isStrictlyIncreasing(INDArray x) {
    NDValidation.validateNumerical("isStrictlyIncreasing", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.IsStrictlyIncreasing(x))[0];
  }

  /**
   * Jaccard similarity reduction operation. The output contains the Jaccard distance for each<br>
   *                 tensor along the specified dimensions.<br>
   *
   * @param x Input variable x (NUMERIC type)
   * @param y Input variable y (NUMERIC type)
   * @param dimensions Dimensions to calculate jaccardDistance over (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray jaccardDistance(INDArray x, INDArray y, int... dimensions) {
    NDValidation.validateNumerical("jaccardDistance", "x", x);
    NDValidation.validateNumerical("jaccardDistance", "y", y);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce3.JaccardDistance(x, y, dimensions));
  }

  /**
   * Last index reduction operation.<br>
   * Returns a variable that contains the index of the last element that matches the specified condition (for each<br>
   * slice along the specified dimensions)<br>
   * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
   * the mean along a dimension).<br>
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
   * keepDims = true: [a,1,c]<br>
   * keepDims = false: [a,c]<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param condition Condition to check on input variable
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=1))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray lastIndex(INDArray in, Condition condition, int... dimensions) {
    NDValidation.validateNumerical("lastIndex", "in", in);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.indexaccum.LastIndex(in, false, condition, dimensions));
  }

  /**
   * Last index reduction operation.<br>
   * Returns a variable that contains the index of the last element that matches the specified condition (for each<br>
   * slice along the specified dimensions)<br>
   * Note that if keepDims = true, the output variable has the same rank as the input variable,<br>
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting<br>
   * the mean along a dimension).<br>
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:<br>
   * keepDims = true: [a,1,c]<br>
   * keepDims = false: [a,c]<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param condition Condition to check on input variable
   * @param keepDims If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=1))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray lastIndex(INDArray in, Condition condition, boolean keepDims, int... dimensions) {
    NDValidation.validateNumerical("lastIndex", "in", in);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.indexaccum.LastIndex(in, keepDims, condition, dimensions));
  }

  /**
   * Calculates difference between inputs X and Y.<br>
   *
   * @param x Input variable X (NUMERIC type)
   * @param y Input variable Y (NUMERIC type)
   */
  public INDArray[] listDiff(INDArray x, INDArray y) {
    NDValidation.validateNumerical("listDiff", "x", x);
    NDValidation.validateNumerical("listDiff", "y", y);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.ListDiff(x, y));
  }

  /**
   * Element-wise logarithm function (base e - natural logarithm): out = log(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray log(INDArray x) {
    NDValidation.validateNumerical("log", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.Log(x));
  }

  /**
   * Element-wise logarithm function (with specified base): out = log_{base}(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param base Logarithm base
   * @return output Output variable (NUMERIC type)
   */
  public INDArray log(INDArray x, double base) {
    NDValidation.validateNumerical("log", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.scalar.LogX(x, base));
  }

  /**
   * Elementwise natural logarithm function: out = log_e (1 + x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray log1p(INDArray x) {
    NDValidation.validateNumerical("log1p", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.Log1p(x));
  }

  /**
   * Log entropy reduction: log(-sum(x * log(x)))<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray logEntropy(INDArray in, int... dimensions) {
    NDValidation.validateNumerical("logEntropy", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.floating.LogEntropy(in, dimensions));
  }

  /**
   * Log-sum-exp reduction (optionally along dimension).<br>
   * Computes log(sum(exp(x))<br>
   *
   * @param input Input variable (NUMERIC type)
   * @param dimensions Optional dimensions to reduce along (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray logSumExp(INDArray input, int... dimensions) {
    NDValidation.validateNumerical("logSumExp", "input", input);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.custom.LogSumExp(input, dimensions))[0];
  }

  /**
   * Manhattan distance (l1 norm, l1 distance) reduction operation. The output contains the Manhattan distance for each<br>
   * tensor/subset along the specified dimensions:<br>
   * out = sum_i abs(x[i]-y[i])<br>
   *
   * @param x Input variable x (NUMERIC type)
   * @param y Input variable y (NUMERIC type)
   * @param dimensions Dimensions to calculate manhattanDistance over (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray manhattanDistance(INDArray x, INDArray y, int... dimensions) {
    NDValidation.validateNumerical("manhattanDistance", "x", x);
    NDValidation.validateNumerical("manhattanDistance", "y", y);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce3.ManhattanDistance(x, y, dimensions));
  }

  /**
   * Matrix determinant op. For 2D input, this returns the standard matrix determinant.<br>
   * For higher dimensional input with shape [..., m, m] the matrix determinant is returned for each <br>
   * shape [m,m] sub-matrix.<br>
   *
   * @param in Input (NUMERIC type)
   * @return output Matrix determinant variable (NUMERIC type)
   */
  public INDArray matrixDeterminant(INDArray in) {
    NDValidation.validateNumerical("matrixDeterminant", "in", in);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.MatrixDeterminant(in))[0];
  }

  /**
   * Matrix inverse op. For 2D input, this returns the standard matrix inverse.<br>
   * For higher dimensional input with shape [..., m, m] the matrix inverse is returned for each<br>
   * shape [m,m] sub-matrix.<br>
   *
   * @param in Input (NUMERIC type)
   * @return output Matrix inverse variable (NUMERIC type)
   */
  public INDArray matrixInverse(INDArray in) {
    NDValidation.validateNumerical("matrixInverse", "in", in);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.MatrixInverse(in))[0];
  }

  /**
   * Pairwise max operation, out = max(x, y)<br>
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]<br>
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html<br>
   *
   * @param x First input variable, x (NUMERIC type)
   * @param y Second input variable, y (NUMERIC type)
   * @return out Output (NUMERIC type)
   */
  public INDArray max(INDArray x, INDArray y) {
    NDValidation.validateNumerical("max", "x", x);
    NDValidation.validateNumerical("max", "y", y);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.Max(x, y))[0];
  }

  /**
   * Merge add function: merges an arbitrary number of equal shaped arrays using element-wise addition:<br>
   * out = sum_i in[i]<br>
   *
   * @param inputs Input variables (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray mergeAdd(INDArray... inputs) {
    NDValidation.validateNumerical("mergeAdd", "inputs", inputs);
    Preconditions.checkArgument(inputs.length >= 1, "inputs has incorrect size/length. Expected: inputs.length >= 1, got %s", inputs.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MergeAddOp(inputs))[0];
  }

  /**
   * Merge average function: merges an arbitrary number of equal shaped arrays using element-wise mean operation:<br>
   * out = mean_i in[i]<br>
   *
   * @param inputs Input variables (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray mergeAvg(INDArray... inputs) {
    NDValidation.validateNumerical("mergeAvg", "inputs", inputs);
    Preconditions.checkArgument(inputs.length >= 1, "inputs has incorrect size/length. Expected: inputs.length >= 1, got %s", inputs.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.MergeAvg(inputs))[0];
  }

  /**
   * Merge max function: merges an arbitrary number of equal shaped arrays using element-wise maximum operation:<br>
   * out = max_i in[i]<br>
   *
   * @param inputs Input variables (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray mergeMax(INDArray... inputs) {
    NDValidation.validateNumerical("mergeMax", "inputs", inputs);
    Preconditions.checkArgument(inputs.length >= 1, "inputs has incorrect size/length. Expected: inputs.length >= 1, got %s", inputs.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.MergeMax(inputs))[0];
  }

  /**
   * Broadcasts parameters for evaluation on an N-D grid.<br>
   *
   * @param inputs  (NUMERIC type)
   * @param cartesian 
   */
  public INDArray[] meshgrid(INDArray[] inputs, boolean cartesian) {
    NDValidation.validateNumerical("meshgrid", "inputs", inputs);
    Preconditions.checkArgument(inputs.length >= 0, "inputs has incorrect size/length. Expected: inputs.length >= 0, got %s", inputs.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.MeshGrid(inputs, cartesian));
  }

  /**
   * Pairwise max operation, out = min(x, y)<br>
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]<br>
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html<br>
   *
   * @param x First input variable, x (NUMERIC type)
   * @param y Second input variable, y (NUMERIC type)
   * @return out Output (NUMERIC type)
   */
  public INDArray min(INDArray x, INDArray y) {
    NDValidation.validateNumerical("min", "x", x);
    NDValidation.validateNumerical("min", "y", y);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.Min(x, y))[0];
  }

  /**
   * Pairwise modulus (remainder) operation, out = x % y<br>
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]<br>
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param y Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray mod(INDArray x, INDArray y) {
    NDValidation.validateNumerical("mod", "x", x);
    NDValidation.validateNumerical("mod", "y", y);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.ModOp(x, y))[0];
  }

  /**
   * Calculate the mean and (population) variance for the input variable, for the specified axis<br>
   *
   * @param input Input to calculate moments for (NUMERIC type)
   * @param axes Dimensions to perform calculation over (Size: AtLeast(min=0))
   */
  public INDArray[] moments(INDArray input, int... axes) {
    NDValidation.validateNumerical("moments", "input", input);
    Preconditions.checkArgument(axes.length >= 0, "axes has incorrect size/length. Expected: axes.length >= 0, got %s", axes.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.Moments(input, axes));
  }

  /**
   * Pairwise multiplication operation, out = x * y<br>
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]<br>
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param y Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray mul(INDArray x, INDArray y) {
    NDValidation.validateNumerical("mul", "x", x);
    NDValidation.validateNumerical("mul", "y", y);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MulOp(x, y))[0];
  }

  /**
   * Scalar multiplication operation, out = in * scalar<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param value Scalar value for op
   * @return output Output variable (NUMERIC type)
   */
  public INDArray mul(INDArray x, double value) {
    NDValidation.validateNumerical("mul", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.scalar.ScalarMultiplication(x, value));
  }

  /**
   * Elementwise negative operation: out = -x<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray neg(INDArray x) {
    NDValidation.validateNumerical("neg", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.same.Negative(x));
  }

  /**
   * Calculate the mean and variance from the sufficient statistics<br>
   *
   * @param counts Rank 0 (scalar) value with the total number of values used to calculate the sufficient statistics (NUMERIC type)
   * @param means Mean-value sufficient statistics: this is the SUM of all data values (NUMERIC type)
   * @param variances Variaance sufficient statistics: this is the squared sum of all data values (NUMERIC type)
   * @param shift Shift value, possibly 0, used when calculating the sufficient statistics (for numerical stability)
   */
  public INDArray[] normalizeMoments(INDArray counts, INDArray means, INDArray variances,
      double shift) {
    NDValidation.validateNumerical("normalizeMoments", "counts", counts);
    NDValidation.validateNumerical("normalizeMoments", "means", means);
    NDValidation.validateNumerical("normalizeMoments", "variances", variances);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.NormalizeMoments(counts, means, variances, shift));
  }

  /**
   * Boolean OR operation: elementwise (x != 0) || (y != 0)<br>
   * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
   * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.<br>
   *
   * @param x Input 1 (BOOL type)
   * @param y Input 2 (BOOL type)
   * @return output INDArray with values 0 and 1 based on where the condition is satisfied (BOOL type)
   */
  public INDArray or(INDArray x, INDArray y) {
    NDValidation.validateBool("or", "x", x);
    NDValidation.validateBool("or", "y", y);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool.Or(x, y));
  }

  /**
   * Element-wise power function: out = x^value<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param value Scalar value for op
   * @return output Output variable (NUMERIC type)
   */
  public INDArray pow(INDArray x, double value) {
    NDValidation.validateNumerical("pow", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.scalar.Pow(x, value));
  }

  /**
   * Element-wise (broadcastable) power function: out = x[i]^y[i]<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param y Power (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray pow(INDArray x, INDArray y) {
    NDValidation.validateNumerical("pow", "x", x);
    NDValidation.validateNumerical("pow", "y", y);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.Pow(x, y))[0];
  }

  /**
   * Rational Tanh Approximation elementwise function, as described in the paper:<br>
   * Compact Convolutional Neural Network Cascade for Face Detection<br>
   * This is a faster Tanh approximation<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray rationalTanh(INDArray x) {
    NDValidation.validateNumerical("rationalTanh", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.RationalTanh(x));
  }

  /**
   * Pairwise reverse division operation, out = y / x<br>
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]<br>
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param y Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray rdiv(INDArray x, INDArray y) {
    NDValidation.validateNumerical("rdiv", "x", x);
    NDValidation.validateNumerical("rdiv", "y", y);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.RDivOp(x, y))[0];
  }

  /**
   * Scalar reverse division operation, out = scalar / in<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param value Scalar value for op
   * @return output Output variable (NUMERIC type)
   */
  public INDArray rdiv(INDArray x, double value) {
    NDValidation.validateNumerical("rdiv", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.scalar.ScalarReverseDivision(x, value));
  }

  /**
   * Element-wise reciprocal (inverse) function: out[i] = 1 / in[i]<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray reciprocal(INDArray x) {
    NDValidation.validateNumerical("reciprocal", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.same.Reciprocal(x));
  }

  /**
   * Rectified tanh operation: max(0, tanh(in))<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray rectifiedTanh(INDArray x) {
    NDValidation.validateNumerical("rectifiedTanh", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.RectifiedTanh(x));
  }

  /**
   * Element-wise round function: out = round(x).<br>
   * Rounds (up or down depending on value) to the nearest integer value.<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray round(INDArray x) {
    NDValidation.validateNumerical("round", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.same.Round(x));
  }

  /**
   * Element-wise reciprocal (inverse) of square root: out = 1.0 / sqrt(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray rsqrt(INDArray x) {
    NDValidation.validateNumerical("rsqrt", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.floating.RSqrt(x));
  }

  /**
   * Pairwise reverse subtraction operation, out = y - x<br>
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]<br>
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param y Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray rsub(INDArray x, INDArray y) {
    NDValidation.validateNumerical("rsub", "x", x);
    NDValidation.validateNumerical("rsub", "y", y);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.RSubOp(x, y))[0];
  }

  /**
   * Scalar reverse subtraction operation, out = scalar - in<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param value Scalar value for op
   * @return output Output variable (NUMERIC type)
   */
  public INDArray rsub(INDArray x, double value) {
    NDValidation.validateNumerical("rsub", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.scalar.ScalarReverseSubtraction(x, value));
  }

  /**
   * Set the diagonal value to the specified values<br>
   * If input is<br>
   * [ a, b, c]<br>
   * [ d, e, f]<br>
   * [ g, h, i]<br>
   * and diag = [ 1, 2, 3] then output is<br>
   * [ 1, b, c]<br>
   * [ d, 2, f]<br>
   * [ g, h, 3]<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param diag Diagonal (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray setDiag(INDArray in, INDArray diag) {
    NDValidation.validateNumerical("setDiag", "in", in);
    NDValidation.validateNumerical("setDiag", "diag", diag);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.MatrixSetDiag(in, diag))[0];
  }

  /**
   * Shannon Entropy reduction: -sum(x * log2(x))<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray shannonEntropy(INDArray in, int... dimensions) {
    NDValidation.validateNumerical("shannonEntropy", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.floating.ShannonEntropy(in, dimensions));
  }

  /**
   * Element-wise sign (signum) function:<br>
   * out = -1 if in < 0<br>
   * out = 0 if in = 0<br>
   * out = 1 if in > 0<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray sign(INDArray x) {
    NDValidation.validateNumerical("sign", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.same.Sign(x));
  }

  /**
   * Elementwise sine operation: out = sin(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray sin(INDArray x) {
    NDValidation.validateNumerical("sin", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.Sin(x));
  }

  /**
   * Elementwise sinh (hyperbolic sine) operation: out = sinh(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray sinh(INDArray x) {
    NDValidation.validateNumerical("sinh", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.Sinh(x));
  }

  /**
   * Element-wise square root function: out = sqrt(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray sqrt(INDArray x) {
    NDValidation.validateNumerical("sqrt", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.floating.Sqrt(x));
  }

  /**
   * Element-wise square function: out = x^2<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray square(INDArray x) {
    NDValidation.validateNumerical("square", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.same.Square(x));
  }

  /**
   * Pairwise squared difference operation.<br>
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]<br>
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param y Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray squaredDifference(INDArray x, INDArray y) {
    NDValidation.validateNumerical("squaredDifference", "x", x);
    NDValidation.validateNumerical("squaredDifference", "y", y);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.SquaredDifferenceOp(x, y))[0];
  }

  /**
   * Standardize input variable along given axis<br>
   * <p><br>
   * out = (x - mean) / stdev<br>
   * <p><br>
   * with mean and stdev being calculated along the given dimension.<br>
   * <p><br>
   * For example: given x as a mini batch of the shape [numExamples, exampleLength]:<br>
   * <ul> <br>
   * <li>use dimension 1 too use the statistics (mean, stdev) for each example</li><br>
   * <li>use dimension 0 if you want to use the statistics for each column across all examples</li><br>
   * <li>use dimensions 0,1 if you want to use the statistics across all columns and examples</li><br>
   * </ul><br>
   *
   * @param x Input variable (NUMERIC type)
   * @param dimensions  (Size: AtLeast(min=1))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray standardize(INDArray x, int... dimensions) {
    NDValidation.validateNumerical("standardize", "x", x);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.Standardize(x, dimensions))[0];
  }

  /**
   * Elementwise step function:<br>
   * out(x) = 1 if x >= cutoff<br>
   * out(x) = 0 otherwise<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param value Scalar value for op
   * @return output Output variable (NUMERIC type)
   */
  public INDArray step(INDArray x, double value) {
    NDValidation.validateNumerical("step", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.scalar.Step(x, value));
  }

  /**
   * Pairwise subtraction operation, out = x - y<br>
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]<br>
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param y Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray sub(INDArray x, INDArray y) {
    NDValidation.validateNumerical("sub", "x", x);
    NDValidation.validateNumerical("sub", "y", y);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.SubOp(x, y))[0];
  }

  /**
   * Scalar subtraction operation, out = in - scalar<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param value Scalar value for op
   * @return output Output variable (NUMERIC type)
   */
  public INDArray sub(INDArray x, double value) {
    NDValidation.validateNumerical("sub", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.scalar.ScalarSubtraction(x, value));
  }

  /**
   * Elementwise tangent operation: out = tan(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray tan(INDArray x) {
    NDValidation.validateNumerical("tan", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.Tan(x));
  }

  /**
   * Elementwise tanh (hyperbolic tangent) operation: out = tanh(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray tanh(INDArray x) {
    NDValidation.validateNumerical("tanh", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.Tanh(x));
  }

  /**
   * Matrix trace operation<br>
   * For rank 2 matrices, the output is a scalar vith the trace - i.e., sum of the main diagonal.<br>
   * For higher rank inputs, output[a,b,c] = trace(in[a,b,c,:,:])<br>
   *
   * @param in Input variable (NUMERIC type)
   * @return output Trace (NUMERIC type)
   */
  public INDArray trace(INDArray in) {
    NDValidation.validateNumerical("trace", "in", in);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.Trace(in))[0];
  }

  /**
   * Boolean XOR (exclusive OR) operation: elementwise (x != 0) XOR (y != 0)<br>
   * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
   * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.<br>
   *
   * @param x Input 1 (BOOL type)
   * @param y Input 2 (BOOL type)
   * @return output INDArray with values 0 and 1 based on where the condition is satisfied (BOOL type)
   */
  public INDArray xor(INDArray x, INDArray y) {
    NDValidation.validateBool("xor", "x", x);
    NDValidation.validateBool("xor", "y", y);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool.Xor(x, y));
  }

  /**
   * Full array zero fraction array reduction operation, optionally along specified dimensions: out = (count(x == 0) / length(x))<br>
   *
   * @param input Input variable (NUMERIC type)
   * @return output Reduced array of rank 0 (scalar) (NUMERIC type)
   */
  public INDArray zeroFraction(INDArray input) {
    NDValidation.validateNumerical("zeroFraction", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.ZeroFraction(input))[0];
  }
}

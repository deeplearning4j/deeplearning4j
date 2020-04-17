/*******************************************************************************
 * Copyright (c) 2019-2020 Konduit K.K.
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

//================== GENERATED CODE - DO NOT MODIFY THIS FILE ==================

package org.nd4j.autodiff.samediff.ops;

import static org.nd4j.autodiff.samediff.ops.SDValidation.isSameType;

import java.lang.String;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.enums.PartitionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.indexing.conditions.Condition;

public class SDMath extends SDOps {
  public SDMath(SameDiff sameDiff) {
    super(sameDiff);
  }

  /**
   * Clips tensor values to a maximum average L2-norm.<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param clipValue Value for clipping
   * @param dimensions Dimensions to reduce over (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable clipByAvgNorm(SDVariable x, double clipValue, int... dimensions) {
    SDValidation.validateNumerical("ClipByAvgNorm", "x", x);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return new org.nd4j.linalg.api.ops.impl.transforms.clip.ClipByAvgNorm(sd,x, clipValue, dimensions).outputVariable();
  }

  /**
   * Clips tensor values to a maximum average L2-norm.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @param clipValue Value for clipping
   * @param dimensions Dimensions to reduce over (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable clipByAvgNorm(String name, SDVariable x, double clipValue, int... dimensions) {
    SDValidation.validateNumerical("ClipByAvgNorm", "x", x);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.clip.ClipByAvgNorm(sd,x, clipValue, dimensions).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Looks up ids in a list of embedding tensors.<br>
   *
   * @param x Input tensor (NUMERIC type)
   * @param indices A Tensor containing the ids to be looked up. (INT type)
   * @param PartitionMode partition_mode == 0 - i.e. 'mod' , 1 - 'div'
   * @return output Shifted output (NUMERIC type)
   */
  public SDVariable embeddingLookup(SDVariable x, SDVariable indices, PartitionMode PartitionMode) {
    SDValidation.validateNumerical("EmbeddingLookup", "x", x);
    SDValidation.validateInteger("EmbeddingLookup", "indices", indices);
    return new org.nd4j.linalg.api.ops.impl.shape.tensorops.EmbeddingLookup(sd,x, indices, PartitionMode).outputVariable();
  }

  /**
   * Looks up ids in a list of embedding tensors.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input tensor (NUMERIC type)
   * @param indices A Tensor containing the ids to be looked up. (INT type)
   * @param PartitionMode partition_mode == 0 - i.e. 'mod' , 1 - 'div'
   * @return output Shifted output (NUMERIC type)
   */
  public SDVariable embeddingLookup(String name, SDVariable x, SDVariable indices,
      PartitionMode PartitionMode) {
    SDValidation.validateNumerical("EmbeddingLookup", "x", x);
    SDValidation.validateInteger("EmbeddingLookup", "indices", indices);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.tensorops.EmbeddingLookup(sd,x, indices, PartitionMode).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Elementwise absolute value operation: out = abs(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable abs(SDVariable x) {
    SDValidation.validateNumerical("abs", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.same.Abs(sd,x).outputVariable();
  }

  /**
   * Elementwise absolute value operation: out = abs(x)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable abs(String name, SDVariable x) {
    SDValidation.validateNumerical("abs", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.same.Abs(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Elementwise acos (arccosine, inverse cosine) operation: out = arccos(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable acos(SDVariable x) {
    SDValidation.validateNumerical("acos", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.strict.ACos(sd,x).outputVariable();
  }

  /**
   * Elementwise acos (arccosine, inverse cosine) operation: out = arccos(x)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable acos(String name, SDVariable x) {
    SDValidation.validateNumerical("acos", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.strict.ACos(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Elementwise acosh (inverse hyperbolic cosine) function: out = acosh(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable acosh(SDVariable x) {
    SDValidation.validateNumerical("acosh", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.strict.ACosh(sd,x).outputVariable();
  }

  /**
   * Elementwise acosh (inverse hyperbolic cosine) function: out = acosh(x)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable acosh(String name, SDVariable x) {
    SDValidation.validateNumerical("acosh", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.strict.ACosh(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Absolute max array reduction operation, optionally along specified dimensions: out = max(abs(x))<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public SDVariable amax(SDVariable in, int... dimensions) {
    SDValidation.validateNumerical("amax", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return new org.nd4j.linalg.api.ops.impl.reduce.same.AMax(sd,in, dimensions).outputVariable();
  }

  /**
   * Absolute max array reduction operation, optionally along specified dimensions: out = max(abs(x))<br>
   *
   * @param name name May be null. Name for the output variable
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public SDVariable amax(String name, SDVariable in, int... dimensions) {
    SDValidation.validateNumerical("amax", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.same.AMax(sd,in, dimensions).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Absolute mean array reduction operation, optionally along specified dimensions: out = mean(abs(x))<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public SDVariable amean(SDVariable in, int... dimensions) {
    SDValidation.validateNumerical("amean", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return new org.nd4j.linalg.api.ops.impl.reduce.floating.AMean(sd,in, dimensions).outputVariable();
  }

  /**
   * Absolute mean array reduction operation, optionally along specified dimensions: out = mean(abs(x))<br>
   *
   * @param name name May be null. Name for the output variable
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public SDVariable amean(String name, SDVariable in, int... dimensions) {
    SDValidation.validateNumerical("amean", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.floating.AMean(sd,in, dimensions).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Absolute min array reduction operation, optionally along specified dimensions: out = min(abs(x))<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public SDVariable amin(SDVariable in, int... dimensions) {
    SDValidation.validateNumerical("amin", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return new org.nd4j.linalg.api.ops.impl.reduce.same.AMin(sd,in, dimensions).outputVariable();
  }

  /**
   * Absolute min array reduction operation, optionally along specified dimensions: out = min(abs(x))<br>
   *
   * @param name name May be null. Name for the output variable
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public SDVariable amin(String name, SDVariable in, int... dimensions) {
    SDValidation.validateNumerical("amin", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.same.AMin(sd,in, dimensions).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable and(SDVariable x, SDVariable y) {
    SDValidation.validateBool("and", "x", x);
    SDValidation.validateBool("and", "y", y);
    return new org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool.And(sd,x, y).outputVariable();
  }

  /**
   * Boolean AND operation: elementwise (x != 0) && (y != 0)<br>
   * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
   * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input 1 (BOOL type)
   * @param y Input 2 (BOOL type)
   * @return output INDArray with values 0 and 1 based on where the condition is satisfied (BOOL type)
   */
  public SDVariable and(String name, SDVariable x, SDVariable y) {
    SDValidation.validateBool("and", "x", x);
    SDValidation.validateBool("and", "y", y);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool.And(sd,x, y).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Elementwise asin (arcsin, inverse sine) operation: out = arcsin(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable asin(SDVariable x) {
    SDValidation.validateNumerical("asin", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.strict.ASin(sd,x).outputVariable();
  }

  /**
   * Elementwise asin (arcsin, inverse sine) operation: out = arcsin(x)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable asin(String name, SDVariable x) {
    SDValidation.validateNumerical("asin", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.strict.ASin(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Elementwise asinh (inverse hyperbolic sine) function: out = asinh(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable asinh(SDVariable x) {
    SDValidation.validateNumerical("asinh", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.strict.ASinh(sd,x).outputVariable();
  }

  /**
   * Elementwise asinh (inverse hyperbolic sine) function: out = asinh(x)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable asinh(String name, SDVariable x) {
    SDValidation.validateNumerical("asinh", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.strict.ASinh(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Absolute sum array reduction operation, optionally along specified dimensions: out = sum(abs(x))<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public SDVariable asum(SDVariable in, int... dimensions) {
    SDValidation.validateNumerical("asum", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return new org.nd4j.linalg.api.ops.impl.reduce.same.ASum(sd,in, dimensions).outputVariable();
  }

  /**
   * Absolute sum array reduction operation, optionally along specified dimensions: out = sum(abs(x))<br>
   *
   * @param name name May be null. Name for the output variable
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public SDVariable asum(String name, SDVariable in, int... dimensions) {
    SDValidation.validateNumerical("asum", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.same.ASum(sd,in, dimensions).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Elementwise atan (arctangent, inverse tangent) operation: out = arctangent(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable atan(SDVariable x) {
    SDValidation.validateNumerical("atan", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.strict.ATan(sd,x).outputVariable();
  }

  /**
   * Elementwise atan (arctangent, inverse tangent) operation: out = arctangent(x)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable atan(String name, SDVariable x) {
    SDValidation.validateNumerical("atan", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.strict.ATan(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Elementwise atan (arctangent, inverse tangent) operation: out = atan2(x,y).<br>
   * Similar to atan(y/x) but sigts of x and y are used to determine the location of the result<br>
   *
   * @param y Input Y variable (NUMERIC type)
   * @param x Input X variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable atan2(SDVariable y, SDVariable x) {
    SDValidation.validateNumerical("atan2", "y", y);
    SDValidation.validateNumerical("atan2", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.ATan2(sd,y, x).outputVariable();
  }

  /**
   * Elementwise atan (arctangent, inverse tangent) operation: out = atan2(x,y).<br>
   * Similar to atan(y/x) but sigts of x and y are used to determine the location of the result<br>
   *
   * @param name name May be null. Name for the output variable
   * @param y Input Y variable (NUMERIC type)
   * @param x Input X variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable atan2(String name, SDVariable y, SDVariable x) {
    SDValidation.validateNumerical("atan2", "y", y);
    SDValidation.validateNumerical("atan2", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.ATan2(sd,y, x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Elementwise atanh (inverse hyperbolic tangent) function: out = atanh(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable atanh(SDVariable x) {
    SDValidation.validateNumerical("atanh", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.strict.ATanh(sd,x).outputVariable();
  }

  /**
   * Elementwise atanh (inverse hyperbolic tangent) function: out = atanh(x)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable atanh(String name, SDVariable x) {
    SDValidation.validateNumerical("atanh", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.strict.ATanh(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Bit shift operation<br>
   *
   * @param x input (NUMERIC type)
   * @param shift shift value (NUMERIC type)
   * @return output shifted output (NUMERIC type)
   */
  public SDVariable bitShift(SDVariable x, SDVariable shift) {
    SDValidation.validateNumerical("bitShift", "x", x);
    SDValidation.validateNumerical("bitShift", "shift", shift);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.ShiftBits(sd,x, shift).outputVariable();
  }

  /**
   * Bit shift operation<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x input (NUMERIC type)
   * @param shift shift value (NUMERIC type)
   * @return output shifted output (NUMERIC type)
   */
  public SDVariable bitShift(String name, SDVariable x, SDVariable shift) {
    SDValidation.validateNumerical("bitShift", "x", x);
    SDValidation.validateNumerical("bitShift", "shift", shift);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.ShiftBits(sd,x, shift).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Right bit shift operation<br>
   *
   * @param x Input tensor (NUMERIC type)
   * @param shift shift argument (NUMERIC type)
   * @return output shifted output (NUMERIC type)
   */
  public SDVariable bitShiftRight(SDVariable x, SDVariable shift) {
    SDValidation.validateNumerical("bitShiftRight", "x", x);
    SDValidation.validateNumerical("bitShiftRight", "shift", shift);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.RShiftBits(sd,x, shift).outputVariable();
  }

  /**
   * Right bit shift operation<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input tensor (NUMERIC type)
   * @param shift shift argument (NUMERIC type)
   * @return output shifted output (NUMERIC type)
   */
  public SDVariable bitShiftRight(String name, SDVariable x, SDVariable shift) {
    SDValidation.validateNumerical("bitShiftRight", "x", x);
    SDValidation.validateNumerical("bitShiftRight", "shift", shift);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.RShiftBits(sd,x, shift).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Cyclic bit shift operation<br>
   *
   * @param x Input tensor (NUMERIC type)
   * @param shift shift argy=ument (NUMERIC type)
   * @return output shifted output (NUMERIC type)
   */
  public SDVariable bitShiftRotl(SDVariable x, SDVariable shift) {
    SDValidation.validateNumerical("bitShiftRotl", "x", x);
    SDValidation.validateNumerical("bitShiftRotl", "shift", shift);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.CyclicShiftBits(sd,x, shift).outputVariable();
  }

  /**
   * Cyclic bit shift operation<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input tensor (NUMERIC type)
   * @param shift shift argy=ument (NUMERIC type)
   * @return output shifted output (NUMERIC type)
   */
  public SDVariable bitShiftRotl(String name, SDVariable x, SDVariable shift) {
    SDValidation.validateNumerical("bitShiftRotl", "x", x);
    SDValidation.validateNumerical("bitShiftRotl", "shift", shift);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.CyclicShiftBits(sd,x, shift).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Cyclic right shift operation<br>
   *
   * @param x Input tensor (NUMERIC type)
   * @param shift Shift argument (NUMERIC type)
   * @return output Shifted output (NUMERIC type)
   */
  public SDVariable bitShiftRotr(SDVariable x, SDVariable shift) {
    SDValidation.validateNumerical("bitShiftRotr", "x", x);
    SDValidation.validateNumerical("bitShiftRotr", "shift", shift);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.CyclicRShiftBits(sd,x, shift).outputVariable();
  }

  /**
   * Cyclic right shift operation<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input tensor (NUMERIC type)
   * @param shift Shift argument (NUMERIC type)
   * @return output Shifted output (NUMERIC type)
   */
  public SDVariable bitShiftRotr(String name, SDVariable x, SDVariable shift) {
    SDValidation.validateNumerical("bitShiftRotr", "x", x);
    SDValidation.validateNumerical("bitShiftRotr", "shift", shift);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.CyclicRShiftBits(sd,x, shift).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Element-wise ceiling function: out = ceil(x).<br>
   * Rounds each value up to the nearest integer value (if not already an integer)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable ceil(SDVariable x) {
    SDValidation.validateNumerical("ceil", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.same.Ceil(sd,x).outputVariable();
  }

  /**
   * Element-wise ceiling function: out = ceil(x).<br>
   * Rounds each value up to the nearest integer value (if not already an integer)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable ceil(String name, SDVariable x) {
    SDValidation.validateNumerical("ceil", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.same.Ceil(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable clipByNorm(SDVariable x, double clipValue, int... dimensions) {
    SDValidation.validateNumerical("clipByNorm", "x", x);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return new org.nd4j.linalg.api.ops.impl.transforms.clip.ClipByNorm(sd,x, clipValue, dimensions).outputVariable();
  }

  /**
   * Clipping by L2 norm, optionally along dimension(s)<br>
   * if l2Norm(x,dimension) < clipValue, then input is returned unmodifed<br>
   * Otherwise, out[i] = in[i] * clipValue / l2Norm(in, dimensions) where each value is clipped according<br>
   * to the corresponding l2Norm along the specified dimensions<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @param clipValue Clipping value (maximum l2 norm)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable clipByNorm(String name, SDVariable x, double clipValue, int... dimensions) {
    SDValidation.validateNumerical("clipByNorm", "x", x);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.clip.ClipByNorm(sd,x, clipValue, dimensions).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable clipByValue(SDVariable x, double clipValueMin, double clipValueMax) {
    SDValidation.validateNumerical("clipByValue", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.clip.ClipByValue(sd,x, clipValueMin, clipValueMax).outputVariable();
  }

  /**
   * Element-wise clipping function:<br>
   * out[i] = in[i] if in[i] >= clipValueMin and in[i] <= clipValueMax<br>
   * out[i] = clipValueMin if in[i] < clipValueMin<br>
   * out[i] = clipValueMax if in[i] > clipValueMax<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @param clipValueMin Minimum value for clipping
   * @param clipValueMax Maximum value for clipping
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable clipByValue(String name, SDVariable x, double clipValueMin,
      double clipValueMax) {
    SDValidation.validateNumerical("clipByValue", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.clip.ClipByValue(sd,x, clipValueMin, clipValueMax).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable confusionMatrix(SDVariable labels, SDVariable pred, DataType dataType) {
    SDValidation.validateNumerical("confusionMatrix", "labels", labels);
    SDValidation.validateNumerical("confusionMatrix", "pred", pred);
    return new org.nd4j.linalg.api.ops.impl.shape.ConfusionMatrix(sd,labels, pred, dataType).outputVariable();
  }

  /**
   * Compute the 2d confusion matrix of size [numClasses, numClasses] from a pair of labels and predictions, both of<br>
   * which are represented as integer values. This version assumes the number of classes is 1 + max(max(labels), max(pred))<br>
   * For example, if labels = [0, 1, 1] and predicted = [0, 2, 1] then output is:<br>
   * [1, 0, 0]<br>
   * [0, 1, 1]<br>
   * [0, 0, 0]<br>
   *
   * @param name name May be null. Name for the output variable
   * @param labels Labels - 1D array of integer values representing label values (NUMERIC type)
   * @param pred Predictions - 1D array of integer values representing predictions. Same length as labels (NUMERIC type)
   * @param dataType Data type
   * @return output variable (2D, shape [numClasses, numClasses}) (NUMERIC type)
   */
  public SDVariable confusionMatrix(String name, SDVariable labels, SDVariable pred,
      DataType dataType) {
    SDValidation.validateNumerical("confusionMatrix", "labels", labels);
    SDValidation.validateNumerical("confusionMatrix", "pred", pred);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.ConfusionMatrix(sd,labels, pred, dataType).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable confusionMatrix(SDVariable labels, SDVariable pred, int numClasses) {
    SDValidation.validateNumerical("confusionMatrix", "labels", labels);
    SDValidation.validateNumerical("confusionMatrix", "pred", pred);
    return new org.nd4j.linalg.api.ops.impl.shape.ConfusionMatrix(sd,labels, pred, numClasses).outputVariable();
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
   * @param name name May be null. Name for the output variable
   * @param labels Labels - 1D array of integer values representing label values (NUMERIC type)
   * @param pred Predictions - 1D array of integer values representing predictions. Same length as labels (NUMERIC type)
   * @param numClasses Number of classes
   * @return output variable (2D, shape [numClasses, numClasses}) (NUMERIC type)
   */
  public SDVariable confusionMatrix(String name, SDVariable labels, SDVariable pred,
      int numClasses) {
    SDValidation.validateNumerical("confusionMatrix", "labels", labels);
    SDValidation.validateNumerical("confusionMatrix", "pred", pred);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.ConfusionMatrix(sd,labels, pred, numClasses).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable confusionMatrix(SDVariable labels, SDVariable pred, SDVariable weights) {
    SDValidation.validateNumerical("confusionMatrix", "labels", labels);
    SDValidation.validateNumerical("confusionMatrix", "pred", pred);
    SDValidation.validateNumerical("confusionMatrix", "weights", weights);
    return new org.nd4j.linalg.api.ops.impl.shape.ConfusionMatrix(sd,labels, pred, weights).outputVariable();
  }

  /**
   * Compute the 2d confusion matrix of size [numClasses, numClasses] from a pair of labels and predictions, both of<br>
   * which are represented as integer values. This version assumes the number of classes is 1 + max(max(labels), max(pred))<br>
   * For example, if labels = [0, 1, 1], predicted = [0, 2, 1] and weights = [1, 2, 3]<br>
   * [1, 0, 0]<br>
   * [0, 3, 2]<br>
   * [0, 0, 0]<br>
   *
   * @param name name May be null. Name for the output variable
   * @param labels Labels - 1D array of integer values representing label values (NUMERIC type)
   * @param pred Predictions - 1D array of integer values representing predictions. Same length as labels (NUMERIC type)
   * @param weights Weights - 1D array of values (may be real/decimal) representing the weight/contribution of each prediction. Must be same length as both labels and predictions arrays (NUMERIC type)
   * @return output variable (2D, shape [numClasses, numClasses}) (NUMERIC type)
   */
  public SDVariable confusionMatrix(String name, SDVariable labels, SDVariable pred,
      SDVariable weights) {
    SDValidation.validateNumerical("confusionMatrix", "labels", labels);
    SDValidation.validateNumerical("confusionMatrix", "pred", pred);
    SDValidation.validateNumerical("confusionMatrix", "weights", weights);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.ConfusionMatrix(sd,labels, pred, weights).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable confusionMatrix(SDVariable labels, SDVariable pred, SDVariable weights,
      int numClasses) {
    SDValidation.validateNumerical("confusionMatrix", "labels", labels);
    SDValidation.validateNumerical("confusionMatrix", "pred", pred);
    SDValidation.validateNumerical("confusionMatrix", "weights", weights);
    return new org.nd4j.linalg.api.ops.impl.shape.ConfusionMatrix(sd,labels, pred, weights, numClasses).outputVariable();
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
   * @param name name May be null. Name for the output variable
   * @param labels Labels - 1D array of integer values representing label values (NUMERIC type)
   * @param pred Predictions - 1D array of integer values representing predictions. Same length as labels (NUMERIC type)
   * @param weights Weights - 1D array of values (may be real/decimal) representing the weight/contribution of each prediction. Must be same length as both labels and predictions arrays (NUMERIC type)
   * @param numClasses 
   * @return output Output variable (2D, shape [numClasses, numClasses}) (NUMERIC type)
   */
  public SDVariable confusionMatrix(String name, SDVariable labels, SDVariable pred,
      SDVariable weights, int numClasses) {
    SDValidation.validateNumerical("confusionMatrix", "labels", labels);
    SDValidation.validateNumerical("confusionMatrix", "pred", pred);
    SDValidation.validateNumerical("confusionMatrix", "weights", weights);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.ConfusionMatrix(sd,labels, pred, weights, numClasses).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Elementwise cosine operation: out = cos(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable cos(SDVariable x) {
    SDValidation.validateNumerical("cos", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.strict.Cos(sd,x).outputVariable();
  }

  /**
   * Elementwise cosine operation: out = cos(x)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable cos(String name, SDVariable x) {
    SDValidation.validateNumerical("cos", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.strict.Cos(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Elementwise cosh (hyperbolic cosine) operation: out = cosh(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable cosh(SDVariable x) {
    SDValidation.validateNumerical("cosh", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.strict.Cosh(sd,x).outputVariable();
  }

  /**
   * Elementwise cosh (hyperbolic cosine) operation: out = cosh(x)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable cosh(String name, SDVariable x) {
    SDValidation.validateNumerical("cosh", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.strict.Cosh(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable cosineDistance(SDVariable x, SDVariable y, int... dimensions) {
    SDValidation.validateNumerical("cosineDistance", "x", x);
    SDValidation.validateNumerical("cosineDistance", "y", y);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return new org.nd4j.linalg.api.ops.impl.reduce3.CosineDistance(sd,x, y, dimensions).outputVariable();
  }

  /**
   * Cosine distance reduction operation. The output contains the cosine distance for each<br>
   * tensor/subset along the specified dimensions:<br>
   * out = 1.0 - cosineSimilarity(x,y)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable x (NUMERIC type)
   * @param y Input variable y (NUMERIC type)
   * @param dimensions Dimensions to calculate cosineDistance over (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable cosineDistance(String name, SDVariable x, SDVariable y, int... dimensions) {
    SDValidation.validateNumerical("cosineDistance", "x", x);
    SDValidation.validateNumerical("cosineDistance", "y", y);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce3.CosineDistance(sd,x, y, dimensions).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable cosineSimilarity(SDVariable x, SDVariable y, int... dimensions) {
    SDValidation.validateNumerical("cosineSimilarity", "x", x);
    SDValidation.validateNumerical("cosineSimilarity", "y", y);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return new org.nd4j.linalg.api.ops.impl.reduce3.CosineSimilarity(sd,x, y, dimensions).outputVariable();
  }

  /**
   * Cosine similarity pairwise reduction operation. The output contains the cosine similarity for each tensor/subset<br>
   * along the specified dimensions:<br>
   * out = (sum_i x[i] * y[i]) / ( sqrt(sum_i x[i]^2) * sqrt(sum_i y[i]^2)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable x (NUMERIC type)
   * @param y Input variable y (NUMERIC type)
   * @param dimensions Dimensions to calculate cosineSimilarity over (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable cosineSimilarity(String name, SDVariable x, SDVariable y, int... dimensions) {
    SDValidation.validateNumerical("cosineSimilarity", "x", x);
    SDValidation.validateNumerical("cosineSimilarity", "y", y);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce3.CosineSimilarity(sd,x, y, dimensions).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Count non zero array reduction operation, optionally along specified dimensions: out = count(x != 0)<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public SDVariable countNonZero(SDVariable in, int... dimensions) {
    SDValidation.validateNumerical("countNonZero", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return new org.nd4j.linalg.api.ops.impl.reduce.longer.CountNonZero(sd,in, dimensions).outputVariable();
  }

  /**
   * Count non zero array reduction operation, optionally along specified dimensions: out = count(x != 0)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public SDVariable countNonZero(String name, SDVariable in, int... dimensions) {
    SDValidation.validateNumerical("countNonZero", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.longer.CountNonZero(sd,in, dimensions).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Count zero array reduction operation, optionally along specified dimensions: out = count(x == 0)<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public SDVariable countZero(SDVariable in, int... dimensions) {
    SDValidation.validateNumerical("countZero", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return new org.nd4j.linalg.api.ops.impl.reduce.longer.CountZero(sd,in, dimensions).outputVariable();
  }

  /**
   * Count zero array reduction operation, optionally along specified dimensions: out = count(x == 0)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public SDVariable countZero(String name, SDVariable in, int... dimensions) {
    SDValidation.validateNumerical("countZero", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.longer.CountZero(sd,in, dimensions).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Returns the pair-wise cross product of equal size arrays a and b: a x b = ||a||x||b|| sin(theta).<br>
   * Can take rank 1 or above inputs (of equal shapes), but note that the last dimension must have dimension 3<br>
   *
   * @param a First input (NUMERIC type)
   * @param b Second input (NUMERIC type)
   * @return output Element-wise cross product (NUMERIC type)
   */
  public SDVariable cross(SDVariable a, SDVariable b) {
    SDValidation.validateNumerical("cross", "a", a);
    SDValidation.validateNumerical("cross", "b", b);
    return new org.nd4j.linalg.api.ops.impl.shape.Cross(sd,a, b).outputVariable();
  }

  /**
   * Returns the pair-wise cross product of equal size arrays a and b: a x b = ||a||x||b|| sin(theta).<br>
   * Can take rank 1 or above inputs (of equal shapes), but note that the last dimension must have dimension 3<br>
   *
   * @param name name May be null. Name for the output variable
   * @param a First input (NUMERIC type)
   * @param b Second input (NUMERIC type)
   * @return output Element-wise cross product (NUMERIC type)
   */
  public SDVariable cross(String name, SDVariable a, SDVariable b) {
    SDValidation.validateNumerical("cross", "a", a);
    SDValidation.validateNumerical("cross", "b", b);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.Cross(sd,a, b).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Element-wise cube function: out = x^3<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable cube(SDVariable x) {
    SDValidation.validateNumerical("cube", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.same.Cube(sd,x).outputVariable();
  }

  /**
   * Element-wise cube function: out = x^3<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable cube(String name, SDVariable x) {
    SDValidation.validateNumerical("cube", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.same.Cube(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable diag(SDVariable x) {
    SDValidation.validateNumerical("diag", "x", x);
    return new org.nd4j.linalg.api.ops.impl.shape.Diag(sd,x).outputVariable();
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
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable diag(String name, SDVariable x) {
    SDValidation.validateNumerical("diag", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.Diag(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable diagPart(SDVariable x) {
    SDValidation.validateNumerical("diagPart", "x", x);
    return new org.nd4j.linalg.api.ops.impl.shape.DiagPart(sd,x).outputVariable();
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
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Diagonal part of the input (NUMERIC type)
   */
  public SDVariable diagPart(String name, SDVariable x) {
    SDValidation.validateNumerical("diagPart", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.DiagPart(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Entropy reduction: -sum(x * log(x))<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public SDVariable entropy(SDVariable in, int... dimensions) {
    SDValidation.validateNumerical("entropy", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return new org.nd4j.linalg.api.ops.impl.reduce.floating.Entropy(sd,in, dimensions).outputVariable();
  }

  /**
   * Entropy reduction: -sum(x * log(x))<br>
   *
   * @param name name May be null. Name for the output variable
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public SDVariable entropy(String name, SDVariable in, int... dimensions) {
    SDValidation.validateNumerical("entropy", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.floating.Entropy(sd,in, dimensions).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Element-wise Gaussian error function - out = erf(in)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable erf(SDVariable x) {
    SDValidation.validateNumerical("erf", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.strict.Erf(sd,x).outputVariable();
  }

  /**
   * Element-wise Gaussian error function - out = erf(in)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable erf(String name, SDVariable x) {
    SDValidation.validateNumerical("erf", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.strict.Erf(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Element-wise complementary Gaussian error function - out = erfc(in) = 1 - erf(in)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable erfc(SDVariable x) {
    SDValidation.validateNumerical("erfc", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.strict.Erfc(sd,x).outputVariable();
  }

  /**
   * Element-wise complementary Gaussian error function - out = erfc(in) = 1 - erf(in)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable erfc(String name, SDVariable x) {
    SDValidation.validateNumerical("erfc", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.strict.Erfc(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable euclideanDistance(SDVariable x, SDVariable y, int... dimensions) {
    SDValidation.validateNumerical("euclideanDistance", "x", x);
    SDValidation.validateNumerical("euclideanDistance", "y", y);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return new org.nd4j.linalg.api.ops.impl.reduce3.EuclideanDistance(sd,x, y, dimensions).outputVariable();
  }

  /**
   * Euclidean distance (l2 norm, l2 distance) reduction operation. The output contains the Euclidean distance for each<br>
   * tensor/subset along the specified dimensions:<br>
   * out = sqrt( sum_i (x[i] - y[i])^2 )<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable x (NUMERIC type)
   * @param y Input variable y (NUMERIC type)
   * @param dimensions Dimensions to calculate euclideanDistance over (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable euclideanDistance(String name, SDVariable x, SDVariable y, int... dimensions) {
    SDValidation.validateNumerical("euclideanDistance", "x", x);
    SDValidation.validateNumerical("euclideanDistance", "y", y);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce3.EuclideanDistance(sd,x, y, dimensions).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Elementwise exponent function: out = exp(x) = 2.71828...^x<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable exp(SDVariable x) {
    SDValidation.validateNumerical("exp", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.strict.Exp(sd,x).outputVariable();
  }

  /**
   * Elementwise exponent function: out = exp(x) = 2.71828...^x<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable exp(String name, SDVariable x) {
    SDValidation.validateNumerical("exp", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.strict.Exp(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Elementwise 1.0 - exponent function: out = 1.0 - exp(x) = 1.0 - 2.71828...^x<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable expm1(SDVariable x) {
    SDValidation.validateNumerical("expm1", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.strict.Expm1(sd,x).outputVariable();
  }

  /**
   * Elementwise 1.0 - exponent function: out = 1.0 - exp(x) = 1.0 - 2.71828...^x<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable expm1(String name, SDVariable x) {
    SDValidation.validateNumerical("expm1", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.strict.Expm1(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Generate an identity matrix with the specified number of rows and columns.<br>
   *
   * @param rows Number of rows
   * @return output Identity matrix (NUMERIC type)
   */
  public SDVariable eye(int rows) {
    return new org.nd4j.linalg.api.ops.impl.shape.Eye(sd,rows).outputVariable();
  }

  /**
   * Generate an identity matrix with the specified number of rows and columns.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param rows Number of rows
   * @return output Identity matrix (NUMERIC type)
   */
  public SDVariable eye(String name, int rows) {
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.Eye(sd,rows).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * As per eye(String, int, int, DataType) but with the default datatype, Eye.DEFAULT_DTYPE<br>
   *
   * @param rows Number of rows
   * @param cols Number of columns
   * @return output  (NUMERIC type)
   */
  public SDVariable eye(int rows, int cols) {
    return new org.nd4j.linalg.api.ops.impl.shape.Eye(sd,rows, cols).outputVariable();
  }

  /**
   * As per eye(String, int, int, DataType) but with the default datatype, Eye.DEFAULT_DTYPE<br>
   *
   * @param name name May be null. Name for the output variable
   * @param rows Number of rows
   * @param cols Number of columns
   * @return output  (NUMERIC type)
   */
  public SDVariable eye(String name, int rows, int cols) {
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.Eye(sd,rows, cols).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable eye(int rows, int cols, DataType dataType, int... dimensions) {
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return new org.nd4j.linalg.api.ops.impl.shape.Eye(sd,rows, cols, dataType, dimensions).outputVariable();
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
   * @param name name May be null. Name for the output variable
   * @param rows Number of rows
   * @param cols Number of columns
   * @param dataType Data type
   * @param dimensions  (Size: AtLeast(min=0))
   * @return output Identity matrix (NUMERIC type)
   */
  public SDVariable eye(String name, int rows, int cols, DataType dataType, int... dimensions) {
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.Eye(sd,rows, cols, dataType, dimensions).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * As per eye(int, int) bit with the number of rows/columns specified as scalar INDArrays<br>
   *
   * @param rows Number of rows (INT type)
   * @param cols Number of columns (INT type)
   * @return output Identity matrix (NUMERIC type)
   */
  public SDVariable eye(SDVariable rows, SDVariable cols) {
    SDValidation.validateInteger("eye", "rows", rows);
    SDValidation.validateInteger("eye", "cols", cols);
    return new org.nd4j.linalg.api.ops.impl.shape.Eye(sd,rows, cols).outputVariable();
  }

  /**
   * As per eye(int, int) bit with the number of rows/columns specified as scalar INDArrays<br>
   *
   * @param name name May be null. Name for the output variable
   * @param rows Number of rows (INT type)
   * @param cols Number of columns (INT type)
   * @return output Identity matrix (NUMERIC type)
   */
  public SDVariable eye(String name, SDVariable rows, SDVariable cols) {
    SDValidation.validateInteger("eye", "rows", rows);
    SDValidation.validateInteger("eye", "cols", cols);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.Eye(sd,rows, cols).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * As per eye(String, int) but with the number of rows specified as a scalar INDArray<br>
   *
   * @param rows Number of rows (INT type)
   * @return output SDVaribable identity matrix (NUMERIC type)
   */
  public SDVariable eye(SDVariable rows) {
    SDValidation.validateInteger("eye", "rows", rows);
    return new org.nd4j.linalg.api.ops.impl.shape.Eye(sd,rows).outputVariable();
  }

  /**
   * As per eye(String, int) but with the number of rows specified as a scalar INDArray<br>
   *
   * @param name name May be null. Name for the output variable
   * @param rows Number of rows (INT type)
   * @return output SDVaribable identity matrix (NUMERIC type)
   */
  public SDVariable eye(String name, SDVariable rows) {
    SDValidation.validateInteger("eye", "rows", rows);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.Eye(sd,rows).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable firstIndex(SDVariable in, Condition condition, int... dimensions) {
    SDValidation.validateNumerical("firstIndex", "in", in);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    return new org.nd4j.linalg.api.ops.impl.indexaccum.FirstIndex(sd,in, false, condition, dimensions).outputVariable();
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
   * @param name name May be null. Name for the output variable
   * @param in Input variable (NUMERIC type)
   * @param condition Condition to check on input variable
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=1))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public SDVariable firstIndex(String name, SDVariable in, Condition condition, int... dimensions) {
    SDValidation.validateNumerical("firstIndex", "in", in);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.indexaccum.FirstIndex(sd,in, false, condition, dimensions).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable firstIndex(SDVariable in, Condition condition, boolean keepDims,
      int... dimensions) {
    SDValidation.validateNumerical("firstIndex", "in", in);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    return new org.nd4j.linalg.api.ops.impl.indexaccum.FirstIndex(sd,in, keepDims, condition, dimensions).outputVariable();
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
   * @param name name May be null. Name for the output variable
   * @param in Input variable (NUMERIC type)
   * @param condition Condition to check on input variable
   * @param keepDims If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=1))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public SDVariable firstIndex(String name, SDVariable in, Condition condition, boolean keepDims,
      int... dimensions) {
    SDValidation.validateNumerical("firstIndex", "in", in);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.indexaccum.FirstIndex(sd,in, keepDims, condition, dimensions).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Element-wise floor function: out = floor(x).<br>
   * Rounds each value down to the nearest integer value (if not already an integer)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable floor(SDVariable x) {
    SDValidation.validateNumerical("floor", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.same.Floor(sd,x).outputVariable();
  }

  /**
   * Element-wise floor function: out = floor(x).<br>
   * Rounds each value down to the nearest integer value (if not already an integer)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable floor(String name, SDVariable x) {
    SDValidation.validateNumerical("floor", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.same.Floor(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable hammingDistance(SDVariable x, SDVariable y, int... dimensions) {
    SDValidation.validateNumerical("hammingDistance", "x", x);
    SDValidation.validateNumerical("hammingDistance", "y", y);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return new org.nd4j.linalg.api.ops.impl.reduce3.HammingDistance(sd,x, y, dimensions).outputVariable();
  }

  /**
   * Hamming distance reduction operation. The output contains the cosine distance for each<br>
   * tensor/subset along the specified dimensions:<br>
   * out = count( x[i] != y[i] )<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable x (NUMERIC type)
   * @param y Input variable y (NUMERIC type)
   * @param dimensions Dimensions to calculate hammingDistance over (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable hammingDistance(String name, SDVariable x, SDVariable y, int... dimensions) {
    SDValidation.validateNumerical("hammingDistance", "x", x);
    SDValidation.validateNumerical("hammingDistance", "y", y);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce3.HammingDistance(sd,x, y, dimensions).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Index of the max absolute value: argmax(abs(in))<br>
   * see argmax(String, INDArray, boolean, int...)<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=1))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public SDVariable iamax(SDVariable in, int... dimensions) {
    SDValidation.validateNumerical("iamax", "in", in);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    return new org.nd4j.linalg.api.ops.impl.indexaccum.IAMax(sd,in, false, dimensions).outputVariable();
  }

  /**
   * Index of the max absolute value: argmax(abs(in))<br>
   * see argmax(String, INDArray, boolean, int...)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=1))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public SDVariable iamax(String name, SDVariable in, int... dimensions) {
    SDValidation.validateNumerical("iamax", "in", in);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.indexaccum.IAMax(sd,in, false, dimensions).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable iamax(SDVariable in, boolean keepDims, int... dimensions) {
    SDValidation.validateNumerical("iamax", "in", in);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    return new org.nd4j.linalg.api.ops.impl.indexaccum.IAMax(sd,in, keepDims, dimensions).outputVariable();
  }

  /**
   * Index of the max absolute value: argmax(abs(in))<br>
   * see argmax(String, INDArray, boolean, int...)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param in Input variable (NUMERIC type)
   * @param keepDims If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=1))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public SDVariable iamax(String name, SDVariable in, boolean keepDims, int... dimensions) {
    SDValidation.validateNumerical("iamax", "in", in);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.indexaccum.IAMax(sd,in, keepDims, dimensions).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Index of the min absolute value: argmin(abs(in))<br>
   * see argmin(String, INDArray, boolean, int...)<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=1))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public SDVariable iamin(SDVariable in, int... dimensions) {
    SDValidation.validateNumerical("iamin", "in", in);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    return new org.nd4j.linalg.api.ops.impl.indexaccum.IAMin(sd,in, false, dimensions).outputVariable();
  }

  /**
   * Index of the min absolute value: argmin(abs(in))<br>
   * see argmin(String, INDArray, boolean, int...)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=1))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public SDVariable iamin(String name, SDVariable in, int... dimensions) {
    SDValidation.validateNumerical("iamin", "in", in);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.indexaccum.IAMin(sd,in, false, dimensions).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable iamin(SDVariable in, boolean keepDims, int... dimensions) {
    SDValidation.validateNumerical("iamin", "in", in);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    return new org.nd4j.linalg.api.ops.impl.indexaccum.IAMin(sd,in, keepDims, dimensions).outputVariable();
  }

  /**
   * Index of the min absolute value: argmin(abs(in))<br>
   * see argmin(String, INDArray, boolean, int...)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param in Input variable (NUMERIC type)
   * @param keepDims If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=1))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public SDVariable iamin(String name, SDVariable in, boolean keepDims, int... dimensions) {
    SDValidation.validateNumerical("iamin", "in", in);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.indexaccum.IAMin(sd,in, keepDims, dimensions).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Is finite operation: elementwise isFinite(x)<br>
   * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or<br>
   * value 0 otherwise<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable isFinite(SDVariable x) {
    SDValidation.validateNumerical("isFinite", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.bool.IsFinite(sd,x).outputVariable();
  }

  /**
   * Is finite operation: elementwise isFinite(x)<br>
   * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or<br>
   * value 0 otherwise<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable isFinite(String name, SDVariable x) {
    SDValidation.validateNumerical("isFinite", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.bool.IsFinite(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Is infinite operation: elementwise isInfinite(x)<br>
   * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or<br>
   * value 0 otherwise<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable isInfinite(SDVariable x) {
    SDValidation.validateNumerical("isInfinite", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.bool.IsInf(sd,x).outputVariable();
  }

  /**
   * Is infinite operation: elementwise isInfinite(x)<br>
   * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or<br>
   * value 0 otherwise<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable isInfinite(String name, SDVariable x) {
    SDValidation.validateNumerical("isInfinite", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.bool.IsInf(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Is maximum operation: elementwise x == max(x)<br>
   * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or<br>
   * value 0 otherwise<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable isMax(SDVariable x) {
    SDValidation.validateNumerical("isMax", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.any.IsMax(sd,x).outputVariable();
  }

  /**
   * Is maximum operation: elementwise x == max(x)<br>
   * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or<br>
   * value 0 otherwise<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable isMax(String name, SDVariable x) {
    SDValidation.validateNumerical("isMax", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.any.IsMax(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Is Not a Number operation: elementwise isNaN(x)<br>
   * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or<br>
   * value 0 otherwise<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable isNaN(SDVariable x) {
    SDValidation.validateNumerical("isNaN", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.bool.IsNaN(sd,x).outputVariable();
  }

  /**
   * Is Not a Number operation: elementwise isNaN(x)<br>
   * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or<br>
   * value 0 otherwise<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable isNaN(String name, SDVariable x) {
    SDValidation.validateNumerical("isNaN", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.bool.IsNaN(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Is the array non decreasing?<br>
   * An array is non-decreasing if for every valid i, x[i] <= x[i+1]. For Rank 2+ arrays, values are compared<br>
   * in 'c' (row major) order<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Scalar variable with value 1 if non-decreasing, or 0 otherwise (NUMERIC type)
   */
  public SDVariable isNonDecreasing(SDVariable x) {
    SDValidation.validateNumerical("isNonDecreasing", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.IsNonDecreasing(sd,x).outputVariable();
  }

  /**
   * Is the array non decreasing?<br>
   * An array is non-decreasing if for every valid i, x[i] <= x[i+1]. For Rank 2+ arrays, values are compared<br>
   * in 'c' (row major) order<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Scalar variable with value 1 if non-decreasing, or 0 otherwise (NUMERIC type)
   */
  public SDVariable isNonDecreasing(String name, SDVariable x) {
    SDValidation.validateNumerical("isNonDecreasing", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.IsNonDecreasing(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Is the array strictly increasing?<br>
   * An array is strictly increasing if for every valid i, x[i] < x[i+1]. For Rank 2+ arrays, values are compared<br>
   * in 'c' (row major) order<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Scalar variable with value 1 if strictly increasing, or 0 otherwise (NUMERIC type)
   */
  public SDVariable isStrictlyIncreasing(SDVariable x) {
    SDValidation.validateNumerical("isStrictlyIncreasing", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.IsStrictlyIncreasing(sd,x).outputVariable();
  }

  /**
   * Is the array strictly increasing?<br>
   * An array is strictly increasing if for every valid i, x[i] < x[i+1]. For Rank 2+ arrays, values are compared<br>
   * in 'c' (row major) order<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Scalar variable with value 1 if strictly increasing, or 0 otherwise (NUMERIC type)
   */
  public SDVariable isStrictlyIncreasing(String name, SDVariable x) {
    SDValidation.validateNumerical("isStrictlyIncreasing", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.IsStrictlyIncreasing(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable jaccardDistance(SDVariable x, SDVariable y, int... dimensions) {
    SDValidation.validateNumerical("jaccardDistance", "x", x);
    SDValidation.validateNumerical("jaccardDistance", "y", y);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return new org.nd4j.linalg.api.ops.impl.reduce3.JaccardDistance(sd,x, y, dimensions).outputVariable();
  }

  /**
   * Jaccard similarity reduction operation. The output contains the Jaccard distance for each<br>
   *                 tensor along the specified dimensions.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable x (NUMERIC type)
   * @param y Input variable y (NUMERIC type)
   * @param dimensions Dimensions to calculate jaccardDistance over (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable jaccardDistance(String name, SDVariable x, SDVariable y, int... dimensions) {
    SDValidation.validateNumerical("jaccardDistance", "x", x);
    SDValidation.validateNumerical("jaccardDistance", "y", y);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce3.JaccardDistance(sd,x, y, dimensions).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable lastIndex(SDVariable in, Condition condition, int... dimensions) {
    SDValidation.validateNumerical("lastIndex", "in", in);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    return new org.nd4j.linalg.api.ops.impl.indexaccum.LastIndex(sd,in, false, condition, dimensions).outputVariable();
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
   * @param name name May be null. Name for the output variable
   * @param in Input variable (NUMERIC type)
   * @param condition Condition to check on input variable
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=1))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public SDVariable lastIndex(String name, SDVariable in, Condition condition, int... dimensions) {
    SDValidation.validateNumerical("lastIndex", "in", in);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.indexaccum.LastIndex(sd,in, false, condition, dimensions).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable lastIndex(SDVariable in, Condition condition, boolean keepDims,
      int... dimensions) {
    SDValidation.validateNumerical("lastIndex", "in", in);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    return new org.nd4j.linalg.api.ops.impl.indexaccum.LastIndex(sd,in, keepDims, condition, dimensions).outputVariable();
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
   * @param name name May be null. Name for the output variable
   * @param in Input variable (NUMERIC type)
   * @param condition Condition to check on input variable
   * @param keepDims If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=1))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public SDVariable lastIndex(String name, SDVariable in, Condition condition, boolean keepDims,
      int... dimensions) {
    SDValidation.validateNumerical("lastIndex", "in", in);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.indexaccum.LastIndex(sd,in, keepDims, condition, dimensions).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Calculates difference between inputs X and Y.<br>
   *
   * @param x Input variable X (NUMERIC type)
   * @param y Input variable Y (NUMERIC type)
   */
  public SDVariable[] listDiff(SDVariable x, SDVariable y) {
    SDValidation.validateNumerical("listDiff", "x", x);
    SDValidation.validateNumerical("listDiff", "y", y);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.ListDiff(sd,x, y).outputVariables();
  }

  /**
   * Calculates difference between inputs X and Y.<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param x Input variable X (NUMERIC type)
   * @param y Input variable Y (NUMERIC type)
   */
  public SDVariable[] listDiff(String[] names, SDVariable x, SDVariable y) {
    SDValidation.validateNumerical("listDiff", "x", x);
    SDValidation.validateNumerical("listDiff", "y", y);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.ListDiff(sd,x, y).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
  }

  /**
   * Element-wise logarithm function (base e - natural logarithm): out = log(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable log(SDVariable x) {
    SDValidation.validateNumerical("log", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.strict.Log(sd,x).outputVariable();
  }

  /**
   * Element-wise logarithm function (base e - natural logarithm): out = log(x)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable log(String name, SDVariable x) {
    SDValidation.validateNumerical("log", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.strict.Log(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Element-wise logarithm function (with specified base): out = log_{base}(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param base Logarithm base
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable log(SDVariable x, double base) {
    SDValidation.validateNumerical("log", "x", x);
    return new org.nd4j.linalg.api.ops.impl.scalar.LogX(sd,x, base).outputVariable();
  }

  /**
   * Element-wise logarithm function (with specified base): out = log_{base}(x)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @param base Logarithm base
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable log(String name, SDVariable x, double base) {
    SDValidation.validateNumerical("log", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.scalar.LogX(sd,x, base).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Elementwise natural logarithm function: out = log_e (1 + x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable log1p(SDVariable x) {
    SDValidation.validateNumerical("log1p", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.strict.Log1p(sd,x).outputVariable();
  }

  /**
   * Elementwise natural logarithm function: out = log_e (1 + x)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable log1p(String name, SDVariable x) {
    SDValidation.validateNumerical("log1p", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.strict.Log1p(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Log entropy reduction: log(-sum(x * log(x)))<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public SDVariable logEntropy(SDVariable in, int... dimensions) {
    SDValidation.validateNumerical("logEntropy", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return new org.nd4j.linalg.api.ops.impl.reduce.floating.LogEntropy(sd,in, dimensions).outputVariable();
  }

  /**
   * Log entropy reduction: log(-sum(x * log(x)))<br>
   *
   * @param name name May be null. Name for the output variable
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public SDVariable logEntropy(String name, SDVariable in, int... dimensions) {
    SDValidation.validateNumerical("logEntropy", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.floating.LogEntropy(sd,in, dimensions).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Log-sum-exp reduction (optionally along dimension).<br>
   * Computes log(sum(exp(x))<br>
   *
   * @param input Input variable (NUMERIC type)
   * @param dimensions Optional dimensions to reduce along (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable logSumExp(SDVariable input, int... dimensions) {
    SDValidation.validateNumerical("logSumExp", "input", input);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return new org.nd4j.linalg.api.ops.impl.reduce.custom.LogSumExp(sd,input, dimensions).outputVariable();
  }

  /**
   * Log-sum-exp reduction (optionally along dimension).<br>
   * Computes log(sum(exp(x))<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input variable (NUMERIC type)
   * @param dimensions Optional dimensions to reduce along (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable logSumExp(String name, SDVariable input, int... dimensions) {
    SDValidation.validateNumerical("logSumExp", "input", input);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.custom.LogSumExp(sd,input, dimensions).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable manhattanDistance(SDVariable x, SDVariable y, int... dimensions) {
    SDValidation.validateNumerical("manhattanDistance", "x", x);
    SDValidation.validateNumerical("manhattanDistance", "y", y);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return new org.nd4j.linalg.api.ops.impl.reduce3.ManhattanDistance(sd,x, y, dimensions).outputVariable();
  }

  /**
   * Manhattan distance (l1 norm, l1 distance) reduction operation. The output contains the Manhattan distance for each<br>
   * tensor/subset along the specified dimensions:<br>
   * out = sum_i abs(x[i]-y[i])<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable x (NUMERIC type)
   * @param y Input variable y (NUMERIC type)
   * @param dimensions Dimensions to calculate manhattanDistance over (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable manhattanDistance(String name, SDVariable x, SDVariable y, int... dimensions) {
    SDValidation.validateNumerical("manhattanDistance", "x", x);
    SDValidation.validateNumerical("manhattanDistance", "y", y);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce3.ManhattanDistance(sd,x, y, dimensions).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Matrix determinant op. For 2D input, this returns the standard matrix determinant.<br>
   * For higher dimensional input with shape [..., m, m] the matrix determinant is returned for each <br>
   * shape [m,m] sub-matrix.<br>
   *
   * @param in Input (NUMERIC type)
   * @return output Matrix determinant variable (NUMERIC type)
   */
  public SDVariable matrixDeterminant(SDVariable in) {
    SDValidation.validateNumerical("matrixDeterminant", "in", in);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.MatrixDeterminant(sd,in).outputVariable();
  }

  /**
   * Matrix determinant op. For 2D input, this returns the standard matrix determinant.<br>
   * For higher dimensional input with shape [..., m, m] the matrix determinant is returned for each <br>
   * shape [m,m] sub-matrix.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param in Input (NUMERIC type)
   * @return output Matrix determinant variable (NUMERIC type)
   */
  public SDVariable matrixDeterminant(String name, SDVariable in) {
    SDValidation.validateNumerical("matrixDeterminant", "in", in);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.MatrixDeterminant(sd,in).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Matrix inverse op. For 2D input, this returns the standard matrix inverse.<br>
   * For higher dimensional input with shape [..., m, m] the matrix inverse is returned for each<br>
   * shape [m,m] sub-matrix.<br>
   *
   * @param in Input (NUMERIC type)
   * @return output Matrix inverse variable (NUMERIC type)
   */
  public SDVariable matrixInverse(SDVariable in) {
    SDValidation.validateNumerical("matrixInverse", "in", in);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.MatrixInverse(sd,in).outputVariable();
  }

  /**
   * Matrix inverse op. For 2D input, this returns the standard matrix inverse.<br>
   * For higher dimensional input with shape [..., m, m] the matrix inverse is returned for each<br>
   * shape [m,m] sub-matrix.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param in Input (NUMERIC type)
   * @return output Matrix inverse variable (NUMERIC type)
   */
  public SDVariable matrixInverse(String name, SDVariable in) {
    SDValidation.validateNumerical("matrixInverse", "in", in);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.MatrixInverse(sd,in).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Merge add function: merges an arbitrary number of equal shaped arrays using element-wise addition:<br>
   * out = sum_i in[i]<br>
   *
   * @param inputs Input variables (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable mergeAdd(SDVariable... inputs) {
    SDValidation.validateNumerical("mergeAdd", "inputs", inputs);
    Preconditions.checkArgument(inputs.length >= 1, "inputs has incorrect size/length. Expected: inputs.length >= 1, got %s", inputs.length);
    return new org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MergeAddOp(sd,inputs).outputVariable();
  }

  /**
   * Merge add function: merges an arbitrary number of equal shaped arrays using element-wise addition:<br>
   * out = sum_i in[i]<br>
   *
   * @param name name May be null. Name for the output variable
   * @param inputs Input variables (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable mergeAdd(String name, SDVariable... inputs) {
    SDValidation.validateNumerical("mergeAdd", "inputs", inputs);
    Preconditions.checkArgument(inputs.length >= 1, "inputs has incorrect size/length. Expected: inputs.length >= 1, got %s", inputs.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MergeAddOp(sd,inputs).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Merge average function: merges an arbitrary number of equal shaped arrays using element-wise mean operation:<br>
   * out = mean_i in[i]<br>
   *
   * @param inputs Input variables (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable mergeAvg(SDVariable... inputs) {
    SDValidation.validateNumerical("mergeAvg", "inputs", inputs);
    Preconditions.checkArgument(inputs.length >= 1, "inputs has incorrect size/length. Expected: inputs.length >= 1, got %s", inputs.length);
    return new org.nd4j.linalg.api.ops.impl.shape.MergeAvg(sd,inputs).outputVariable();
  }

  /**
   * Merge average function: merges an arbitrary number of equal shaped arrays using element-wise mean operation:<br>
   * out = mean_i in[i]<br>
   *
   * @param name name May be null. Name for the output variable
   * @param inputs Input variables (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable mergeAvg(String name, SDVariable... inputs) {
    SDValidation.validateNumerical("mergeAvg", "inputs", inputs);
    Preconditions.checkArgument(inputs.length >= 1, "inputs has incorrect size/length. Expected: inputs.length >= 1, got %s", inputs.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.MergeAvg(sd,inputs).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Merge max function: merges an arbitrary number of equal shaped arrays using element-wise maximum operation:<br>
   * out = max_i in[i]<br>
   *
   * @param inputs Input variables (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable mergeMax(SDVariable... inputs) {
    SDValidation.validateNumerical("mergeMax", "inputs", inputs);
    Preconditions.checkArgument(inputs.length >= 1, "inputs has incorrect size/length. Expected: inputs.length >= 1, got %s", inputs.length);
    return new org.nd4j.linalg.api.ops.impl.shape.MergeMax(sd,inputs).outputVariable();
  }

  /**
   * Merge max function: merges an arbitrary number of equal shaped arrays using element-wise maximum operation:<br>
   * out = max_i in[i]<br>
   *
   * @param name name May be null. Name for the output variable
   * @param inputs Input variables (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable mergeMax(String name, SDVariable... inputs) {
    SDValidation.validateNumerical("mergeMax", "inputs", inputs);
    Preconditions.checkArgument(inputs.length >= 1, "inputs has incorrect size/length. Expected: inputs.length >= 1, got %s", inputs.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.MergeMax(sd,inputs).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Broadcasts parameters for evaluation on an N-D grid.<br>
   *
   * @param inputs  (NUMERIC type)
   * @param cartesian 
   */
  public SDVariable[] meshgrid(SDVariable[] inputs, boolean cartesian) {
    SDValidation.validateNumerical("meshgrid", "inputs", inputs);
    Preconditions.checkArgument(inputs.length >= 0, "inputs has incorrect size/length. Expected: inputs.length >= 0, got %s", inputs.length);
    return new org.nd4j.linalg.api.ops.impl.shape.MeshGrid(sd,inputs, cartesian).outputVariables();
  }

  /**
   * Broadcasts parameters for evaluation on an N-D grid.<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param inputs  (NUMERIC type)
   * @param cartesian 
   */
  public SDVariable[] meshgrid(String[] names, SDVariable[] inputs, boolean cartesian) {
    SDValidation.validateNumerical("meshgrid", "inputs", inputs);
    Preconditions.checkArgument(inputs.length >= 0, "inputs has incorrect size/length. Expected: inputs.length >= 0, got %s", inputs.length);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.shape.MeshGrid(sd,inputs, cartesian).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
  }

  /**
   * Calculate the mean and (population) variance for the input variable, for the specified axis<br>
   *
   * @param input Input to calculate moments for (NUMERIC type)
   * @param axes Dimensions to perform calculation over (Size: AtLeast(min=0))
   */
  public SDVariable[] moments(SDVariable input, int... axes) {
    SDValidation.validateNumerical("moments", "input", input);
    Preconditions.checkArgument(axes.length >= 0, "axes has incorrect size/length. Expected: axes.length >= 0, got %s", axes.length);
    return new org.nd4j.linalg.api.ops.impl.reduce.Moments(sd,input, axes).outputVariables();
  }

  /**
   * Calculate the mean and (population) variance for the input variable, for the specified axis<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param input Input to calculate moments for (NUMERIC type)
   * @param axes Dimensions to perform calculation over (Size: AtLeast(min=0))
   */
  public SDVariable[] moments(String[] names, SDVariable input, int... axes) {
    SDValidation.validateNumerical("moments", "input", input);
    Preconditions.checkArgument(axes.length >= 0, "axes has incorrect size/length. Expected: axes.length >= 0, got %s", axes.length);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.reduce.Moments(sd,input, axes).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
  }

  /**
   * Elementwise negative operation: out = -x<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable neg(SDVariable x) {
    SDValidation.validateNumerical("neg", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.same.Negative(sd,x).outputVariable();
  }

  /**
   * Elementwise negative operation: out = -x<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable neg(String name, SDVariable x) {
    SDValidation.validateNumerical("neg", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.same.Negative(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Calculate the mean and variance from the sufficient statistics<br>
   *
   * @param counts Rank 0 (scalar) value with the total number of values used to calculate the sufficient statistics (NUMERIC type)
   * @param means Mean-value sufficient statistics: this is the SUM of all data values (NUMERIC type)
   * @param variances Variaance sufficient statistics: this is the squared sum of all data values (NUMERIC type)
   * @param shift Shift value, possibly 0, used when calculating the sufficient statistics (for numerical stability)
   */
  public SDVariable[] normalizeMoments(SDVariable counts, SDVariable means, SDVariable variances,
      double shift) {
    SDValidation.validateNumerical("normalizeMoments", "counts", counts);
    SDValidation.validateNumerical("normalizeMoments", "means", means);
    SDValidation.validateNumerical("normalizeMoments", "variances", variances);
    return new org.nd4j.linalg.api.ops.impl.reduce.NormalizeMoments(sd,counts, means, variances, shift).outputVariables();
  }

  /**
   * Calculate the mean and variance from the sufficient statistics<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param counts Rank 0 (scalar) value with the total number of values used to calculate the sufficient statistics (NUMERIC type)
   * @param means Mean-value sufficient statistics: this is the SUM of all data values (NUMERIC type)
   * @param variances Variaance sufficient statistics: this is the squared sum of all data values (NUMERIC type)
   * @param shift Shift value, possibly 0, used when calculating the sufficient statistics (for numerical stability)
   */
  public SDVariable[] normalizeMoments(String[] names, SDVariable counts, SDVariable means,
      SDVariable variances, double shift) {
    SDValidation.validateNumerical("normalizeMoments", "counts", counts);
    SDValidation.validateNumerical("normalizeMoments", "means", means);
    SDValidation.validateNumerical("normalizeMoments", "variances", variances);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.reduce.NormalizeMoments(sd,counts, means, variances, shift).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
  public SDVariable or(SDVariable x, SDVariable y) {
    SDValidation.validateBool("or", "x", x);
    SDValidation.validateBool("or", "y", y);
    return new org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool.Or(sd,x, y).outputVariable();
  }

  /**
   * Boolean OR operation: elementwise (x != 0) || (y != 0)<br>
   * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
   * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input 1 (BOOL type)
   * @param y Input 2 (BOOL type)
   * @return output INDArray with values 0 and 1 based on where the condition is satisfied (BOOL type)
   */
  public SDVariable or(String name, SDVariable x, SDVariable y) {
    SDValidation.validateBool("or", "x", x);
    SDValidation.validateBool("or", "y", y);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool.Or(sd,x, y).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Element-wise power function: out = x^value<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param value Scalar value for op
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable pow(SDVariable x, double value) {
    SDValidation.validateNumerical("pow", "x", x);
    return new org.nd4j.linalg.api.ops.impl.scalar.Pow(sd,x, value).outputVariable();
  }

  /**
   * Element-wise power function: out = x^value<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @param value Scalar value for op
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable pow(String name, SDVariable x, double value) {
    SDValidation.validateNumerical("pow", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.scalar.Pow(sd,x, value).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Element-wise (broadcastable) power function: out = x[i]^y[i]<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param y Power (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable pow(SDVariable x, SDVariable y) {
    SDValidation.validateNumerical("pow", "x", x);
    SDValidation.validateNumerical("pow", "y", y);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.Pow(sd,x, y).outputVariable();
  }

  /**
   * Element-wise (broadcastable) power function: out = x[i]^y[i]<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @param y Power (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable pow(String name, SDVariable x, SDVariable y) {
    SDValidation.validateNumerical("pow", "x", x);
    SDValidation.validateNumerical("pow", "y", y);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.Pow(sd,x, y).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Element-wise reciprocal (inverse) function: out[i] = 1 / in[i]<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable reciprocal(SDVariable x) {
    SDValidation.validateNumerical("reciprocal", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.same.Reciprocal(sd,x).outputVariable();
  }

  /**
   * Element-wise reciprocal (inverse) function: out[i] = 1 / in[i]<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable reciprocal(String name, SDVariable x) {
    SDValidation.validateNumerical("reciprocal", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.same.Reciprocal(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Element-wise round function: out = round(x).<br>
   * Rounds (up or down depending on value) to the nearest integer value.<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable round(SDVariable x) {
    SDValidation.validateNumerical("round", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.same.Round(sd,x).outputVariable();
  }

  /**
   * Element-wise round function: out = round(x).<br>
   * Rounds (up or down depending on value) to the nearest integer value.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable round(String name, SDVariable x) {
    SDValidation.validateNumerical("round", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.same.Round(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Element-wise reciprocal (inverse) of square root: out = 1.0 / sqrt(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable rsqrt(SDVariable x) {
    SDValidation.validateNumerical("rsqrt", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.floating.RSqrt(sd,x).outputVariable();
  }

  /**
   * Element-wise reciprocal (inverse) of square root: out = 1.0 / sqrt(x)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable rsqrt(String name, SDVariable x) {
    SDValidation.validateNumerical("rsqrt", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.floating.RSqrt(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable setDiag(SDVariable in, SDVariable diag) {
    SDValidation.validateNumerical("setDiag", "in", in);
    SDValidation.validateNumerical("setDiag", "diag", diag);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.MatrixSetDiag(sd,in, diag).outputVariable();
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
   * @param name name May be null. Name for the output variable
   * @param in Input variable (NUMERIC type)
   * @param diag Diagonal (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable setDiag(String name, SDVariable in, SDVariable diag) {
    SDValidation.validateNumerical("setDiag", "in", in);
    SDValidation.validateNumerical("setDiag", "diag", diag);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.MatrixSetDiag(sd,in, diag).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Shannon Entropy reduction: -sum(x * log2(x))<br>
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public SDVariable shannonEntropy(SDVariable in, int... dimensions) {
    SDValidation.validateNumerical("shannonEntropy", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return new org.nd4j.linalg.api.ops.impl.reduce.floating.ShannonEntropy(sd,in, dimensions).outputVariable();
  }

  /**
   * Shannon Entropy reduction: -sum(x * log2(x))<br>
   *
   * @param name name May be null. Name for the output variable
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public SDVariable shannonEntropy(String name, SDVariable in, int... dimensions) {
    SDValidation.validateNumerical("shannonEntropy", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.floating.ShannonEntropy(sd,in, dimensions).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable sign(SDVariable x) {
    SDValidation.validateNumerical("sign", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.same.Sign(sd,x).outputVariable();
  }

  /**
   * Element-wise sign (signum) function:<br>
   * out = -1 if in < 0<br>
   * out = 0 if in = 0<br>
   * out = 1 if in > 0<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable sign(String name, SDVariable x) {
    SDValidation.validateNumerical("sign", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.same.Sign(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Elementwise sine operation: out = sin(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable sin(SDVariable x) {
    SDValidation.validateNumerical("sin", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.strict.Sin(sd,x).outputVariable();
  }

  /**
   * Elementwise sine operation: out = sin(x)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable sin(String name, SDVariable x) {
    SDValidation.validateNumerical("sin", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.strict.Sin(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Elementwise sinh (hyperbolic sine) operation: out = sinh(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable sinh(SDVariable x) {
    SDValidation.validateNumerical("sinh", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.strict.Sinh(sd,x).outputVariable();
  }

  /**
   * Elementwise sinh (hyperbolic sine) operation: out = sinh(x)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable sinh(String name, SDVariable x) {
    SDValidation.validateNumerical("sinh", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.strict.Sinh(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Element-wise square root function: out = sqrt(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable sqrt(SDVariable x) {
    SDValidation.validateNumerical("sqrt", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.floating.Sqrt(sd,x).outputVariable();
  }

  /**
   * Element-wise square root function: out = sqrt(x)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable sqrt(String name, SDVariable x) {
    SDValidation.validateNumerical("sqrt", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.floating.Sqrt(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Element-wise square function: out = x^2<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable square(SDVariable x) {
    SDValidation.validateNumerical("square", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.same.Square(sd,x).outputVariable();
  }

  /**
   * Element-wise square function: out = x^2<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable square(String name, SDVariable x) {
    SDValidation.validateNumerical("square", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.same.Square(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable standardize(SDVariable x, int... dimensions) {
    SDValidation.validateNumerical("standardize", "x", x);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.Standardize(sd,x, dimensions).outputVariable();
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
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @param dimensions  (Size: AtLeast(min=1))
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable standardize(String name, SDVariable x, int... dimensions) {
    SDValidation.validateNumerical("standardize", "x", x);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.Standardize(sd,x, dimensions).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable step(SDVariable x, double value) {
    SDValidation.validateNumerical("step", "x", x);
    return new org.nd4j.linalg.api.ops.impl.scalar.Step(sd,x, value).outputVariable();
  }

  /**
   * Elementwise step function:<br>
   * out(x) = 1 if x >= cutoff<br>
   * out(x) = 0 otherwise<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @param value Scalar value for op
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable step(String name, SDVariable x, double value) {
    SDValidation.validateNumerical("step", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.scalar.Step(sd,x, value).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Elementwise tangent operation: out = tan(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable tan(SDVariable x) {
    SDValidation.validateNumerical("tan", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.strict.Tan(sd,x).outputVariable();
  }

  /**
   * Elementwise tangent operation: out = tan(x)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable tan(String name, SDVariable x) {
    SDValidation.validateNumerical("tan", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.strict.Tan(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Elementwise tanh (hyperbolic tangent) operation: out = tanh(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable tanh(SDVariable x) {
    SDValidation.validateNumerical("tanh", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.strict.Tanh(sd,x).outputVariable();
  }

  /**
   * Elementwise tanh (hyperbolic tangent) operation: out = tanh(x)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable tanh(String name, SDVariable x) {
    SDValidation.validateNumerical("tanh", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.strict.Tanh(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Matrix trace operation<br>
   * For rank 2 matrices, the output is a scalar vith the trace - i.e., sum of the main diagonal.<br>
   * For higher rank inputs, output[a,b,c] = trace(in[a,b,c,:,:])<br>
   *
   * @param in Input variable (NUMERIC type)
   * @return output Trace (NUMERIC type)
   */
  public SDVariable trace(SDVariable in) {
    SDValidation.validateNumerical("trace", "in", in);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.Trace(sd,in).outputVariable();
  }

  /**
   * Matrix trace operation<br>
   * For rank 2 matrices, the output is a scalar vith the trace - i.e., sum of the main diagonal.<br>
   * For higher rank inputs, output[a,b,c] = trace(in[a,b,c,:,:])<br>
   *
   * @param name name May be null. Name for the output variable
   * @param in Input variable (NUMERIC type)
   * @return output Trace (NUMERIC type)
   */
  public SDVariable trace(String name, SDVariable in) {
    SDValidation.validateNumerical("trace", "in", in);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.Trace(sd,in).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable xor(SDVariable x, SDVariable y) {
    SDValidation.validateBool("xor", "x", x);
    SDValidation.validateBool("xor", "y", y);
    return new org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool.Xor(sd,x, y).outputVariable();
  }

  /**
   * Boolean XOR (exclusive OR) operation: elementwise (x != 0) XOR (y != 0)<br>
   * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
   * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input 1 (BOOL type)
   * @param y Input 2 (BOOL type)
   * @return output INDArray with values 0 and 1 based on where the condition is satisfied (BOOL type)
   */
  public SDVariable xor(String name, SDVariable x, SDVariable y) {
    SDValidation.validateBool("xor", "x", x);
    SDValidation.validateBool("xor", "y", y);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool.Xor(sd,x, y).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Full array zero fraction array reduction operation, optionally along specified dimensions: out = (count(x == 0) / length(x))<br>
   *
   * @param input Input variable (NUMERIC type)
   * @return output Reduced array of rank 0 (scalar) (NUMERIC type)
   */
  public SDVariable zeroFraction(SDVariable input) {
    SDValidation.validateNumerical("zeroFraction", "input", input);
    return new org.nd4j.linalg.api.ops.impl.reduce.ZeroFraction(sd,input).outputVariable();
  }

  /**
   * Full array zero fraction array reduction operation, optionally along specified dimensions: out = (count(x == 0) / length(x))<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input variable (NUMERIC type)
   * @return output Reduced array of rank 0 (scalar) (NUMERIC type)
   */
  public SDVariable zeroFraction(String name, SDVariable input) {
    SDValidation.validateNumerical("zeroFraction", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.ZeroFraction(sd,input).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }
}

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

//================== GENERATED CODE - DO NOT MODIFY THIS FILE ==================

package org.nd4j.linalg.factory.ops;

import static org.nd4j.linalg.factory.NDValidation.isSameType;

import org.nd4j.common.base.Preconditions;
import org.nd4j.enums.PartitionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.FirstIndex;
import org.nd4j.linalg.api.ops.impl.indexaccum.LastIndex;
import org.nd4j.linalg.api.ops.impl.indexaccum.custom.ArgMax;
import org.nd4j.linalg.api.ops.impl.indexaccum.custom.ArgMin;
import org.nd4j.linalg.api.ops.impl.reduce.Moments;
import org.nd4j.linalg.api.ops.impl.reduce.NormalizeMoments;
import org.nd4j.linalg.api.ops.impl.reduce.ZeroFraction;
import org.nd4j.linalg.api.ops.impl.reduce.custom.LogSumExp;
import org.nd4j.linalg.api.ops.impl.reduce.floating.AMean;
import org.nd4j.linalg.api.ops.impl.reduce.floating.Entropy;
import org.nd4j.linalg.api.ops.impl.reduce.floating.LogEntropy;
import org.nd4j.linalg.api.ops.impl.reduce.floating.Mean;
import org.nd4j.linalg.api.ops.impl.reduce.floating.Norm1;
import org.nd4j.linalg.api.ops.impl.reduce.floating.Norm2;
import org.nd4j.linalg.api.ops.impl.reduce.floating.NormMax;
import org.nd4j.linalg.api.ops.impl.reduce.floating.ShannonEntropy;
import org.nd4j.linalg.api.ops.impl.reduce.floating.SquaredNorm;
import org.nd4j.linalg.api.ops.impl.reduce.longer.CountNonZero;
import org.nd4j.linalg.api.ops.impl.reduce.longer.CountZero;
import org.nd4j.linalg.api.ops.impl.reduce.same.AMax;
import org.nd4j.linalg.api.ops.impl.reduce.same.AMin;
import org.nd4j.linalg.api.ops.impl.reduce.same.ASum;
import org.nd4j.linalg.api.ops.impl.reduce.same.Prod;
import org.nd4j.linalg.api.ops.impl.reduce.same.Sum;
import org.nd4j.linalg.api.ops.impl.reduce3.CosineDistance;
import org.nd4j.linalg.api.ops.impl.reduce3.CosineSimilarity;
import org.nd4j.linalg.api.ops.impl.reduce3.EuclideanDistance;
import org.nd4j.linalg.api.ops.impl.reduce3.HammingDistance;
import org.nd4j.linalg.api.ops.impl.reduce3.JaccardDistance;
import org.nd4j.linalg.api.ops.impl.reduce3.ManhattanDistance;
import org.nd4j.linalg.api.ops.impl.scalar.LogX;
import org.nd4j.linalg.api.ops.impl.scalar.Pow;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarAdd;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarDivision;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarFMod;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarMultiplication;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarReverseDivision;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarReverseSubtraction;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarSubtraction;
import org.nd4j.linalg.api.ops.impl.scalar.Step;
import org.nd4j.linalg.api.ops.impl.shape.ConfusionMatrix;
import org.nd4j.linalg.api.ops.impl.shape.Cross;
import org.nd4j.linalg.api.ops.impl.shape.Diag;
import org.nd4j.linalg.api.ops.impl.shape.DiagPart;
import org.nd4j.linalg.api.ops.impl.shape.Eye;
import org.nd4j.linalg.api.ops.impl.shape.MergeAvg;
import org.nd4j.linalg.api.ops.impl.shape.MergeMax;
import org.nd4j.linalg.api.ops.impl.shape.MergeMaxIndex;
import org.nd4j.linalg.api.ops.impl.shape.MeshGrid;
import org.nd4j.linalg.api.ops.impl.shape.tensorops.EmbeddingLookup;
import org.nd4j.linalg.api.ops.impl.transforms.any.IsMax;
import org.nd4j.linalg.api.ops.impl.transforms.bool.IsFinite;
import org.nd4j.linalg.api.ops.impl.transforms.bool.IsInf;
import org.nd4j.linalg.api.ops.impl.transforms.bool.IsNaN;
import org.nd4j.linalg.api.ops.impl.transforms.clip.ClipByAvgNorm;
import org.nd4j.linalg.api.ops.impl.transforms.clip.ClipByNorm;
import org.nd4j.linalg.api.ops.impl.transforms.clip.ClipByValue;
import org.nd4j.linalg.api.ops.impl.transforms.custom.ATan2;
import org.nd4j.linalg.api.ops.impl.transforms.custom.CyclicRShiftBits;
import org.nd4j.linalg.api.ops.impl.transforms.custom.CyclicShiftBits;
import org.nd4j.linalg.api.ops.impl.transforms.custom.IsNonDecreasing;
import org.nd4j.linalg.api.ops.impl.transforms.custom.IsStrictlyIncreasing;
import org.nd4j.linalg.api.ops.impl.transforms.custom.ListDiff;
import org.nd4j.linalg.api.ops.impl.transforms.custom.LogicalAnd;
import org.nd4j.linalg.api.ops.impl.transforms.custom.LogicalOr;
import org.nd4j.linalg.api.ops.impl.transforms.custom.LogicalXor;
import org.nd4j.linalg.api.ops.impl.transforms.custom.MatrixDeterminant;
import org.nd4j.linalg.api.ops.impl.transforms.custom.MatrixInverse;
import org.nd4j.linalg.api.ops.impl.transforms.custom.MatrixSetDiag;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Max;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Min;
import org.nd4j.linalg.api.ops.impl.transforms.custom.RShiftBits;
import org.nd4j.linalg.api.ops.impl.transforms.custom.ShiftBits;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Standardize;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Trace;
import org.nd4j.linalg.api.ops.impl.transforms.floating.RSqrt;
import org.nd4j.linalg.api.ops.impl.transforms.floating.Sqrt;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.AddOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.DivOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.FloorDivOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.FloorModOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MergeAddOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.ModOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MulOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.RDivOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.RSubOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.SquaredDifferenceOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.SubOp;
import org.nd4j.linalg.api.ops.impl.transforms.same.Abs;
import org.nd4j.linalg.api.ops.impl.transforms.same.Ceil;
import org.nd4j.linalg.api.ops.impl.transforms.same.Cube;
import org.nd4j.linalg.api.ops.impl.transforms.same.Floor;
import org.nd4j.linalg.api.ops.impl.transforms.same.Negative;
import org.nd4j.linalg.api.ops.impl.transforms.same.Reciprocal;
import org.nd4j.linalg.api.ops.impl.transforms.same.Round;
import org.nd4j.linalg.api.ops.impl.transforms.same.Sign;
import org.nd4j.linalg.api.ops.impl.transforms.same.Square;
import org.nd4j.linalg.api.ops.impl.transforms.strict.ACos;
import org.nd4j.linalg.api.ops.impl.transforms.strict.ACosh;
import org.nd4j.linalg.api.ops.impl.transforms.strict.ASin;
import org.nd4j.linalg.api.ops.impl.transforms.strict.ASinh;
import org.nd4j.linalg.api.ops.impl.transforms.strict.ATan;
import org.nd4j.linalg.api.ops.impl.transforms.strict.ATanh;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Cos;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Cosh;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Erf;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Erfc;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Exp;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Expm1;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Log;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Log1p;
import org.nd4j.linalg.api.ops.impl.transforms.strict.RationalTanh;
import org.nd4j.linalg.api.ops.impl.transforms.strict.RectifiedTanh;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Sin;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Sinh;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Tan;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Tanh;
import org.nd4j.linalg.factory.NDValidation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Condition;

public class NDMath {
  public NDMath() {
  }

  /**
   * Clips tensor values to a maximum average L2-norm.
   *
   * @param x Input variable (NUMERIC type)
   * @param clipValue Value for clipping
   * @param dimensions Dimensions to reduce over (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray clipByAvgNorm(INDArray x, double clipValue, long... dimensions) {
    NDValidation.validateNumerical("ClipByAvgNorm", "x", x);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    INDArray[] __tmp = Nd4j.exec(new ClipByAvgNorm(x, clipValue, dimensions));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Looks up ids in a list of embedding tensors.
   *
   *
   * @param x Input tensor (NUMERIC type)
   * @param indices A Tensor containing the ids to be looked up. (INT type)
   * @param PartitionMode partition_mode == 0 - i.e. 'mod' , 1 - 'div'
   * @return output Shifted output (NUMERIC type)
   */
  public INDArray embeddingLookup(INDArray x, INDArray[] indices, PartitionMode PartitionMode) {
    NDValidation.validateNumerical("EmbeddingLookup", "x", x);
    NDValidation.validateInteger("EmbeddingLookup", "indices", indices);
    Preconditions.checkArgument(indices.length >= 1, "indices has incorrect size/length. Expected: indices.length >= 1, got %s", indices.length);
    INDArray[] __tmp = Nd4j.exec(new EmbeddingLookup(x, indices, PartitionMode));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Return array of max elements indices with along tensor dimensions
   *
   * @param x Input tensor (NUMERIC type)
   * @param dataType Data type
   * @return output Array max elements indices with along dimensions. (INT type)
   */
  public INDArray mergeMaxIndex(INDArray[] x, DataType dataType) {
    NDValidation.validateNumerical("MergeMaxIndex", "x", x);
    Preconditions.checkArgument(x.length >= 1, "x has incorrect size/length. Expected: x.length >= 1, got %s", x.length);
    INDArray[] __tmp = Nd4j.exec(new MergeMaxIndex(x, dataType));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Return array of max elements indices with along tensor dimensions
   *
   * @param x Input tensor (NUMERIC type)
   * @return output Array max elements indices with along dimensions. (INT type)
   */
  public INDArray mergeMaxIndex(INDArray... x) {
    NDValidation.validateNumerical("MergeMaxIndex", "x", x);
    Preconditions.checkArgument(x.length >= 1, "x has incorrect size/length. Expected: x.length >= 1, got %s", x.length);
    INDArray[] __tmp = Nd4j.exec(new MergeMaxIndex(x, DataType.INT));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Elementwise absolute value operation: out = abs(x)
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray abs(INDArray x) {
    NDValidation.validateNumerical("abs", "x", x);
    return Nd4j.exec(new Abs(x));
  }

  /**
   * Elementwise acos (arccosine, inverse cosine) operation: out = arccos(x)
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray acos(INDArray x) {
    NDValidation.validateNumerical("acos", "x", x);
    return Nd4j.exec(new ACos(x));
  }

  /**
   * Elementwise acosh (inverse hyperbolic cosine) function: out = acosh(x)
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray acosh(INDArray x) {
    NDValidation.validateNumerical("acosh", "x", x);
    return Nd4j.exec(new ACosh(x));
  }

  /**
   * Pairwise addition operation, out = x + y
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
   *
   * @param x Input variable (NUMERIC type)
   * @param y Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray add(INDArray x, INDArray y) {
    NDValidation.validateNumerical("add", "x", x);
    NDValidation.validateNumerical("add", "y", y);
    INDArray[] __tmp = Nd4j.exec(new AddOp(x, y));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Scalar add operation, out = in + scalar
   *
   * @param x Input variable (NUMERIC type)
   * @param value Scalar value for op
   * @return output Output variable (NUMERIC type)
   */
  public INDArray add(INDArray x, double value) {
    NDValidation.validateNumerical("add", "x", x);
    return Nd4j.exec(new ScalarAdd(x, value));
  }

  /**
   * Boolean AND operation: elementwise (x != 0) &amp;&amp; (y != 0)
   * If x and y arrays have equal shape, the output shape is the same as these inputs.
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.
   * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
   *
   * @param x Input 1 (BOOL type)
   * @param y Input 2 (BOOL type)
   * @return output INDArray with values 0 and 1 based on where the condition is satisfied (BOOL type)
   */
  public INDArray and(INDArray x, INDArray y) {
    NDValidation.validateBool("and", "x", x);
    NDValidation.validateBool("and", "y", y);
    INDArray[] __tmp = Nd4j.exec(new LogicalAnd(x, y));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Elementwise asin (arcsin, inverse sine) operation: out = arcsin(x)
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray asin(INDArray x) {
    NDValidation.validateNumerical("asin", "x", x);
    return Nd4j.exec(new ASin(x));
  }

  /**
   * Elementwise asinh (inverse hyperbolic sine) function: out = asinh(x)
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray asinh(INDArray x) {
    NDValidation.validateNumerical("asinh", "x", x);
    return Nd4j.exec(new ASinh(x));
  }

  /**
   * Absolute sum array reduction operation, optionally along specified dimensions: out = sum(abs(x))
   *
   * @param in Input variable (NUMERIC type)
   * @param keepDims Whether to keep the original  dimensions or produce a shrunk array with less dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray asum(INDArray in, boolean keepDims, long... dimensions) {
    NDValidation.validateNumerical("asum", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new ASum(in, keepDims, dimensions));
  }

  /**
   * Absolute sum array reduction operation, optionally along specified dimensions: out = sum(abs(x))
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray asum(INDArray in, long... dimensions) {
    NDValidation.validateNumerical("asum", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new ASum(in, false, dimensions));
  }

  /**
   * Elementwise atan (arctangent, inverse tangent) operation: out = arctangent(x)
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray atan(INDArray x) {
    NDValidation.validateNumerical("atan", "x", x);
    return Nd4j.exec(new ATan(x));
  }

  /**
   * Elementwise atan (arctangent, inverse tangent) operation: out = atan2(x,y).
   * Similar to atan(y/x) but sigts of x and y are used to determine the location of the result
   *
   * @param y Input Y variable (NUMERIC type)
   * @param x Input X variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray atan2(INDArray y, INDArray x) {
    NDValidation.validateNumerical("atan2", "y", y);
    NDValidation.validateNumerical("atan2", "x", x);
    INDArray[] __tmp = Nd4j.exec(new ATan2(y, x));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Elementwise atanh (inverse hyperbolic tangent) function: out = atanh(x)
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray atanh(INDArray x) {
    NDValidation.validateNumerical("atanh", "x", x);
    return Nd4j.exec(new ATanh(x));
  }

  /**
   * Bit shift operation
   *
   * @param x input (NUMERIC type)
   * @param shift shift value (NUMERIC type)
   * @return output shifted output (NUMERIC type)
   */
  public INDArray bitShift(INDArray x, INDArray shift) {
    NDValidation.validateNumerical("bitShift", "x", x);
    NDValidation.validateNumerical("bitShift", "shift", shift);
    INDArray[] __tmp = Nd4j.exec(new ShiftBits(x, shift));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Right bit shift operation
   *
   * @param x Input tensor (NUMERIC type)
   * @param shift shift argument (NUMERIC type)
   * @return output shifted output (NUMERIC type)
   */
  public INDArray bitShiftRight(INDArray x, INDArray shift) {
    NDValidation.validateNumerical("bitShiftRight", "x", x);
    NDValidation.validateNumerical("bitShiftRight", "shift", shift);
    INDArray[] __tmp = Nd4j.exec(new RShiftBits(x, shift));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Cyclic bit shift operation
   *
   * @param x Input tensor (NUMERIC type)
   * @param shift shift argy=ument (NUMERIC type)
   * @return output shifted output (NUMERIC type)
   */
  public INDArray bitShiftRotl(INDArray x, INDArray shift) {
    NDValidation.validateNumerical("bitShiftRotl", "x", x);
    NDValidation.validateNumerical("bitShiftRotl", "shift", shift);
    INDArray[] __tmp = Nd4j.exec(new CyclicShiftBits(x, shift));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Cyclic right shift operation
   *
   * @param x Input tensor (NUMERIC type)
   * @param shift Shift argument (NUMERIC type)
   * @return output Shifted output (NUMERIC type)
   */
  public INDArray bitShiftRotr(INDArray x, INDArray shift) {
    NDValidation.validateNumerical("bitShiftRotr", "x", x);
    NDValidation.validateNumerical("bitShiftRotr", "shift", shift);
    INDArray[] __tmp = Nd4j.exec(new CyclicRShiftBits(x, shift));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Element-wise ceiling function: out = ceil(x).
   * Rounds each value up to the nearest integer value (if not already an integer)
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray ceil(INDArray x) {
    NDValidation.validateNumerical("ceil", "x", x);
    return Nd4j.exec(new Ceil(x));
  }

  /**
   * Clipping by L2 norm, optionally along dimension(s)
   * if l2Norm(x,dimension) &lt; clipValue, then input is returned unmodified
   * Otherwise, out[i] = in[i] * clipValue / l2Norm(in, dimensions) where each value is clipped according
   * to the corresponding l2Norm along the specified dimensions
   *
   * @param x Input variable (NUMERIC type)
   * @param clipValue Clipping value (maximum l2 norm)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray clipByNorm(INDArray x, double clipValue, long... dimensions) {
    NDValidation.validateNumerical("clipByNorm", "x", x);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    INDArray[] __tmp = Nd4j.exec(new ClipByNorm(x, clipValue, dimensions));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Element-wise clipping function:
   * out[i] = in[i] if in[i] >= clipValueMin and in[i] &lt;= clipValueMax
   * out[i] = clipValueMin if in[i] &lt; clipValueMin
   * out[i] = clipValueMax if in[i] > clipValueMax
   *
   * @param x Input variable (NUMERIC type)
   * @param clipValueMin Minimum value for clipping
   * @param clipValueMax Maximum value for clipping
   * @return output Output variable (NUMERIC type)
   */
  public INDArray clipByValue(INDArray x, double clipValueMin, double clipValueMax) {
    NDValidation.validateNumerical("clipByValue", "x", x);
    INDArray[] __tmp = Nd4j.exec(new ClipByValue(x, clipValueMin, clipValueMax));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Compute the 2d confusion matrix of size [numClasses, numClasses] from a pair of labels and predictions, both of
   * which are represented as integer values. This version assumes the number of classes is 1 + max(max(labels), max(pred))
   * For example, if labels = [0, 1, 1] and predicted = [0, 2, 1] then output is:
   * [1, 0, 0]
   * [0, 1, 1]
   * [0, 0, 0]
   *
   * @param labels Labels - 1D array of integer values representing label values (NUMERIC type)
   * @param pred Predictions - 1D array of integer values representing predictions. Same length as labels (NUMERIC type)
   * @param dataType Data type
   * @return output variable (2D, shape [numClasses, numClasses}) (NUMERIC type)
   */
  public INDArray confusionMatrix(INDArray labels, INDArray pred, DataType dataType) {
    NDValidation.validateNumerical("confusionMatrix", "labels", labels);
    NDValidation.validateNumerical("confusionMatrix", "pred", pred);
    INDArray[] __tmp = Nd4j.exec(new ConfusionMatrix(labels, pred, dataType));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Compute the 2d confusion matrix of size [numClasses, numClasses] from a pair of labels and predictions, both of
   * which are represented as integer values.
   * For example, if labels = [0, 1, 1], predicted = [0, 2, 1], and numClasses=4 then output is:
   * [1, 0, 0, 0]
   * [0, 1, 1, 0]
   * [0, 0, 0, 0]
   * [0, 0, 0, 0]
   *
   * @param labels Labels - 1D array of integer values representing label values (NUMERIC type)
   * @param pred Predictions - 1D array of integer values representing predictions. Same length as labels (NUMERIC type)
   * @param numClasses Number of classes
   * @return output variable (2D, shape [numClasses, numClasses}) (NUMERIC type)
   */
  public INDArray confusionMatrix(INDArray labels, INDArray pred, int numClasses) {
    NDValidation.validateNumerical("confusionMatrix", "labels", labels);
    NDValidation.validateNumerical("confusionMatrix", "pred", pred);
    INDArray[] __tmp = Nd4j.exec(new ConfusionMatrix(labels, pred, numClasses));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Compute the 2d confusion matrix of size [numClasses, numClasses] from a pair of labels and predictions, both of
   * which are represented as integer values. This version assumes the number of classes is 1 + max(max(labels), max(pred))
   * For example, if labels = [0, 1, 1], predicted = [0, 2, 1] and weights = [1, 2, 3]
   * [1, 0, 0]
   * [0, 3, 2]
   * [0, 0, 0]
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
    INDArray[] __tmp = Nd4j.exec(new ConfusionMatrix(labels, pred, weights));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Compute the 2d confusion matrix of size [numClasses, numClasses] from a pair of labels and predictions, both of
   * which are represented as integer values.
   * For example, if labels = [0, 1, 1], predicted = [0, 2, 1], numClasses = 4, and weights = [1, 2, 3]
   * [1, 0, 0, 0]
   * [0, 3, 2, 0]
   * [0, 0, 0, 0]
   * [0, 0, 0, 0]
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
    INDArray[] __tmp = Nd4j.exec(new ConfusionMatrix(labels, pred, weights, numClasses));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Elementwise cosine operation: out = cos(x)
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray cos(INDArray x) {
    NDValidation.validateNumerical("cos", "x", x);
    return Nd4j.exec(new Cos(x));
  }

  /**
   * Elementwise cosh (hyperbolic cosine) operation: out = cosh(x)
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray cosh(INDArray x) {
    NDValidation.validateNumerical("cosh", "x", x);
    return Nd4j.exec(new Cosh(x));
  }

  /**
   * Cosine distance reduction operation. The output contains the cosine distance for each
   * tensor/subset along the specified dimensions:
   * out = 1.0 - cosineSimilarity(x,y)
   *
   * @param x Input variable x (NUMERIC type)
   * @param y Input variable y (NUMERIC type)
   * @param keepDims Whether to preserve original dimensions or not
   * @param isComplex Depending on the implementation, such as distance calculations, this can determine whether all distance calculations for all points should be done.
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray cosineDistance(INDArray x, INDArray y, boolean keepDims, boolean isComplex,
      long... dimensions) {
    NDValidation.validateNumerical("cosineDistance", "x", x);
    NDValidation.validateNumerical("cosineDistance", "y", y);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new CosineDistance(x, y, keepDims, isComplex, dimensions));
  }

  /**
   * Cosine distance reduction operation. The output contains the cosine distance for each
   * tensor/subset along the specified dimensions:
   * out = 1.0 - cosineSimilarity(x,y)
   *
   * @param x Input variable x (NUMERIC type)
   * @param y Input variable y (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray cosineDistance(INDArray x, INDArray y, long... dimensions) {
    NDValidation.validateNumerical("cosineDistance", "x", x);
    NDValidation.validateNumerical("cosineDistance", "y", y);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new CosineDistance(x, y, false, false, dimensions));
  }

  /**
   * Cosine similarity pairwise reduction operation. The output contains the cosine similarity for each tensor/subset
   * along the specified dimensions:
   * out = (sum_i x[i] * y[i]) / ( sqrt(sum_i x[i]^2) * sqrt(sum_i y[i]^2)
   *
   * @param x Input variable x (NUMERIC type)
   * @param y Input variable y (NUMERIC type)
   * @param keepDims Whether to preserve original dimensions or not
   * @param isComplex Depending on the implementation, such as distance calculations, this can determine whether all distance calculations for all points should be done.
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray cosineSimilarity(INDArray x, INDArray y, boolean keepDims, boolean isComplex,
      long... dimensions) {
    NDValidation.validateNumerical("cosineSimilarity", "x", x);
    NDValidation.validateNumerical("cosineSimilarity", "y", y);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new CosineSimilarity(x, y, keepDims, isComplex, dimensions));
  }

  /**
   * Cosine similarity pairwise reduction operation. The output contains the cosine similarity for each tensor/subset
   * along the specified dimensions:
   * out = (sum_i x[i] * y[i]) / ( sqrt(sum_i x[i]^2) * sqrt(sum_i y[i]^2)
   *
   * @param x Input variable x (NUMERIC type)
   * @param y Input variable y (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray cosineSimilarity(INDArray x, INDArray y, long... dimensions) {
    NDValidation.validateNumerical("cosineSimilarity", "x", x);
    NDValidation.validateNumerical("cosineSimilarity", "y", y);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new CosineSimilarity(x, y, false, false, dimensions));
  }

  /**
   * Count non zero array reduction operation, optionally along specified dimensions: out = count(x != 0)
   *
   * @param in Input variable (NUMERIC type)
   * @param keepDims Whether to keep the original  dimensions or produce a shrunk array with less dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray countNonZero(INDArray in, boolean keepDims, long... dimensions) {
    NDValidation.validateNumerical("countNonZero", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new CountNonZero(in, keepDims, dimensions));
  }

  /**
   * Count non zero array reduction operation, optionally along specified dimensions: out = count(x != 0)
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray countNonZero(INDArray in, long... dimensions) {
    NDValidation.validateNumerical("countNonZero", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new CountNonZero(in, false, dimensions));
  }

  /**
   * Count zero array reduction operation, optionally along specified dimensions: out = count(x == 0)
   *
   * @param in Input variable (NUMERIC type)
   * @param keepDims Whether to keep the original  dimensions or produce a shrunk array with less dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray countZero(INDArray in, boolean keepDims, long... dimensions) {
    NDValidation.validateNumerical("countZero", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new CountZero(in, keepDims, dimensions));
  }

  /**
   * Count zero array reduction operation, optionally along specified dimensions: out = count(x == 0)
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray countZero(INDArray in, long... dimensions) {
    NDValidation.validateNumerical("countZero", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new CountZero(in, false, dimensions));
  }

  /**
   * Returns the pair-wise cross product of equal size arrays a and b: a x b = ||a||x||b|| sin(theta).
   * Can take rank 1 or above inputs (of equal shapes), but note that the last dimension must have dimension 3
   *
   * @param a First input (NUMERIC type)
   * @param b Second input (NUMERIC type)
   * @return output Element-wise cross product (NUMERIC type)
   */
  public INDArray cross(INDArray a, INDArray b) {
    NDValidation.validateNumerical("cross", "a", a);
    NDValidation.validateNumerical("cross", "b", b);
    INDArray[] __tmp = Nd4j.exec(new Cross(a, b));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Element-wise cube function: out = x^3
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray cube(INDArray x) {
    NDValidation.validateNumerical("cube", "x", x);
    return Nd4j.exec(new Cube(x));
  }

  /**
   * Returns an output variable with diagonal values equal to the specified values; off-diagonal values will be set to 0
   * For example, if input = [1,2,3], then output is given by:
   * [ 1, 0, 0]
   * [ 0, 2, 0]
   * [ 0, 0, 3]
   *
   * Higher input ranks are also supported: if input has shape [a,...,R-1] then output[i,...,k,i,...,k] = input[i,...,k].
   * i.e., for input rank R, output has rank 2R
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray diag(INDArray x) {
    NDValidation.validateNumerical("diag", "x", x);
    INDArray[] __tmp = Nd4j.exec(new Diag(x));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Extract the diagonal part from the input array.
   * If input is
   * [ 1, 0, 0]
   * [ 0, 2, 0]
   * [ 0, 0, 3]
   * then output is [1, 2, 3].
   * Supports higher dimensions: in general, out[i,...,k] = in[i,...,k,i,...,k]
   *
   * @param x Input variable (NUMERIC type)
   * @return output Diagonal part of the input (NUMERIC type)
   */
  public INDArray diagPart(INDArray x) {
    NDValidation.validateNumerical("diagPart", "x", x);
    INDArray[] __tmp = Nd4j.exec(new DiagPart(x));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Pairwise division operation, out = x / y
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
   *
   * @param x Input variable (NUMERIC type)
   * @param y Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray div(INDArray x, INDArray y) {
    NDValidation.validateNumerical("div", "x", x);
    NDValidation.validateNumerical("div", "y", y);
    INDArray[] __tmp = Nd4j.exec(new DivOp(x, y));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Scalar division operation, out = in / scalar
   *
   * @param x Input variable (NUMERIC type)
   * @param value Scalar value for op
   * @return output Output variable (NUMERIC type)
   */
  public INDArray div(INDArray x, double value) {
    NDValidation.validateNumerical("div", "x", x);
    return Nd4j.exec(new ScalarDivision(x, value));
  }

  /**
   * Entropy reduction: -sum(x * log(x))
   *
   * @param in Input variable (NUMERIC type)
   * @param keepDims Whether to keep the original  dimensions or produce a shrunk array with less dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray entropy(INDArray in, boolean keepDims, long... dimensions) {
    NDValidation.validateNumerical("entropy", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new Entropy(in, keepDims, dimensions));
  }

  /**
   * Entropy reduction: -sum(x * log(x))
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray entropy(INDArray in, long... dimensions) {
    NDValidation.validateNumerical("entropy", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new Entropy(in, false, dimensions));
  }

  /**
   * Element-wise Gaussian error function - out = erf(in)
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray erf(INDArray x) {
    NDValidation.validateNumerical("erf", "x", x);
    return Nd4j.exec(new Erf(x));
  }

  /**
   * Element-wise complementary Gaussian error function - out = erfc(in) = 1 - erf(in)
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray erfc(INDArray x) {
    NDValidation.validateNumerical("erfc", "x", x);
    return Nd4j.exec(new Erfc(x));
  }

  /**
   * Euclidean distance (l2 norm, l2 distance) reduction operation. The output contains the Euclidean distance for each
   * tensor/subset along the specified dimensions:
   * out = sqrt( sum_i (x[i] - y[i])^2 )
   *
   * @param x Input variable x (NUMERIC type)
   * @param y Input variable y (NUMERIC type)
   * @param keepDims Whether to preserve original dimensions or not
   * @param isComplex Depending on the implementation, such as distance calculations, this can determine whether all distance calculations for all points should be done.
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray euclideanDistance(INDArray x, INDArray y, boolean keepDims, boolean isComplex,
      long... dimensions) {
    NDValidation.validateNumerical("euclideanDistance", "x", x);
    NDValidation.validateNumerical("euclideanDistance", "y", y);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new EuclideanDistance(x, y, keepDims, isComplex, dimensions));
  }

  /**
   * Euclidean distance (l2 norm, l2 distance) reduction operation. The output contains the Euclidean distance for each
   * tensor/subset along the specified dimensions:
   * out = sqrt( sum_i (x[i] - y[i])^2 )
   *
   * @param x Input variable x (NUMERIC type)
   * @param y Input variable y (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray euclideanDistance(INDArray x, INDArray y, long... dimensions) {
    NDValidation.validateNumerical("euclideanDistance", "x", x);
    NDValidation.validateNumerical("euclideanDistance", "y", y);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new EuclideanDistance(x, y, false, false, dimensions));
  }

  /**
   * Elementwise exponent function: out = exp(x) = 2.71828...^x
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray exp(INDArray x) {
    NDValidation.validateNumerical("exp", "x", x);
    return Nd4j.exec(new Exp(x));
  }

  /**
   * Elementwise 1.0 - exponent function: out = 1.0 - exp(x) = 1.0 - 2.71828...^x
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray expm1(INDArray x) {
    NDValidation.validateNumerical("expm1", "x", x);
    return Nd4j.exec(new Expm1(x));
  }

  /**
   * Generate an identity matrix with the specified number of rows and columns.
   *
   * @param rows Number of rows
   * @return output Identity matrix (NUMERIC type)
   */
  public INDArray eye(int rows) {
    INDArray[] __tmp = Nd4j.exec(new Eye(rows));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * As per eye(String, int, int, DataType) but with the default datatype, Eye.DEFAULT_DTYPE
   *
   * @param rows Number of rows
   * @param cols Number of columns
   * @return output  (NUMERIC type)
   */
  public INDArray eye(int rows, int cols) {
    INDArray[] __tmp = Nd4j.exec(new Eye(rows, cols));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Generate an identity matrix with the specified number of rows and columns
   * Example:
   * <pre>
   * {@code INDArray eye = eye(3,2)
   * eye:
   * [ 1, 0]
   * [ 0, 1]
   * [ 0, 0]}
   * </pre>
   *
   * @param rows Number of rows
   * @param cols Number of columns
   * @param dataType Data type
   * @param dimensions  (Size: AtLeast(min=0))
   * @return output Identity matrix (NUMERIC type)
   */
  public INDArray eye(int rows, int cols, DataType dataType, long... dimensions) {
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    INDArray[] __tmp = Nd4j.exec(new Eye(rows, cols, dataType, dimensions));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * As per eye(int, int) bit with the number of rows/columns specified as scalar INDArrays
   *
   * @param rows Number of rows (INT type)
   * @param cols Number of columns (INT type)
   * @return output Identity matrix (NUMERIC type)
   */
  public INDArray eye(INDArray rows, INDArray cols) {
    NDValidation.validateInteger("eye", "rows", rows);
    NDValidation.validateInteger("eye", "cols", cols);
    INDArray[] __tmp = Nd4j.exec(new Eye(rows, cols));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * As per eye(String, int) but with the number of rows specified as a scalar INDArray
   *
   * @param rows Number of rows (INT type)
   * @return output SDVaribable identity matrix (NUMERIC type)
   */
  public INDArray eye(INDArray rows) {
    NDValidation.validateInteger("eye", "rows", rows);
    INDArray[] __tmp = Nd4j.exec(new Eye(rows));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * First index reduction operation.
   * Returns a variable that contains the index of the first element that matches the specified condition (for each
   * slice along the specified dimensions)
   * Note that if keepDims = true, the output variable has the same rank as the input variable,
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
   * the mean along a dimension).
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
   * keepDims = true: [a,1,c]
   * keepDims = false: [a,c]
   *
   * @param in Input variable (NUMERIC type)
   * @param condition Condition to check on input variable
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=1))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray firstIndex(INDArray in, Condition condition, long... dimensions) {
    NDValidation.validateNumerical("firstIndex", "in", in);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    return Nd4j.exec(new FirstIndex(in, false, condition, dimensions));
  }

  /**
   * First index reduction operation.
   * Returns a variable that contains the index of the first element that matches the specified condition (for each
   * slice along the specified dimensions)
   * Note that if keepDims = true, the output variable has the same rank as the input variable,
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
   * the mean along a dimension).
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
   * keepDims = true: [a,1,c]
   * keepDims = false: [a,c]
   *
   * @param in Input variable (NUMERIC type)
   * @param condition Condition to check on input variable
   * @param keepDims If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=1))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray firstIndex(INDArray in, Condition condition, boolean keepDims,
      long... dimensions) {
    NDValidation.validateNumerical("firstIndex", "in", in);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    return Nd4j.exec(new FirstIndex(in, keepDims, condition, dimensions));
  }

  /**
   * Element-wise floor function: out = floor(x).
   * Rounds each value down to the nearest integer value (if not already an integer)
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray floor(INDArray x) {
    NDValidation.validateNumerical("floor", "x", x);
    return Nd4j.exec(new Floor(x));
  }

  /**
   * Pairwise floor division operation, out = floor(x / y)
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
   *
   * @param x Input variable (NUMERIC type)
   * @param y Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray floorDiv(INDArray x, INDArray y) {
    NDValidation.validateNumerical("floorDiv", "x", x);
    NDValidation.validateNumerical("floorDiv", "y", y);
    INDArray[] __tmp = Nd4j.exec(new FloorDivOp(x, y));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Pairwise Modulus division operation
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
   *
   * @param x Input variable (NUMERIC type)
   * @param y Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray floorMod(INDArray x, INDArray y) {
    NDValidation.validateNumerical("floorMod", "x", x);
    NDValidation.validateNumerical("floorMod", "y", y);
    INDArray[] __tmp = Nd4j.exec(new FloorModOp(x, y));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Scalar floor modulus operation
   *
   * @param x Input variable (NUMERIC type)
   * @param value Scalar value for op
   * @return output Output variable (NUMERIC type)
   */
  public INDArray floorMod(INDArray x, double value) {
    NDValidation.validateNumerical("floorMod", "x", x);
    return Nd4j.exec(new ScalarFMod(x, value));
  }

  /**
   * Hamming distance reduction operation. The output contains the cosine distance for each
   * tensor/subset along the specified dimensions:
   * out = count( x[i] != y[i] )
   *
   * @param x Input variable x (NUMERIC type)
   * @param y Input variable y (NUMERIC type)
   * @param keepDims Whether to preserve original dimensions or not
   * @param isComplex Depending on the implementation, such as distance calculations, this can determine whether all distance calculations for all points should be done.
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray hammingDistance(INDArray x, INDArray y, boolean keepDims, boolean isComplex,
      long... dimensions) {
    NDValidation.validateNumerical("hammingDistance", "x", x);
    NDValidation.validateNumerical("hammingDistance", "y", y);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new HammingDistance(x, y, keepDims, isComplex, dimensions));
  }

  /**
   * Hamming distance reduction operation. The output contains the cosine distance for each
   * tensor/subset along the specified dimensions:
   * out = count( x[i] != y[i] )
   *
   * @param x Input variable x (NUMERIC type)
   * @param y Input variable y (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray hammingDistance(INDArray x, INDArray y, long... dimensions) {
    NDValidation.validateNumerical("hammingDistance", "x", x);
    NDValidation.validateNumerical("hammingDistance", "y", y);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new HammingDistance(x, y, false, false, dimensions));
  }

  /**
   * Index of the max absolute value: argmax(abs(in))
   * see argmax(String, INDArray, boolean, int...)
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=1))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray iamax(INDArray in, long... dimensions) {
    NDValidation.validateNumerical("iamax", "in", in);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    INDArray[] __tmp = Nd4j.exec(new ArgMax(in, false, dimensions));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Index of the max absolute value: argmax(abs(in))
   * see argmax(String, INDArray, boolean, int...)
   *
   * @param in Input variable (NUMERIC type)
   * @param keepDims If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=1))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray iamax(INDArray in, boolean keepDims, long... dimensions) {
    NDValidation.validateNumerical("iamax", "in", in);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    INDArray[] __tmp = Nd4j.exec(new ArgMax(in, keepDims, dimensions));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Index of the min absolute value: argmin(abs(in))
   * see argmin(String, INDArray, boolean, int...)
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=1))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray iamin(INDArray in, long... dimensions) {
    NDValidation.validateNumerical("iamin", "in", in);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    INDArray[] __tmp = Nd4j.exec(new ArgMin(in, false, dimensions));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Index of the min absolute value: argmin(abs(in))
   * see argmin(String, INDArray, boolean, int...)
   *
   * @param in Input variable (NUMERIC type)
   * @param keepDims If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=1))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray iamin(INDArray in, boolean keepDims, long... dimensions) {
    NDValidation.validateNumerical("iamin", "in", in);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    INDArray[] __tmp = Nd4j.exec(new ArgMin(in, keepDims, dimensions));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Is finite operation: elementwise isFinite(x)
   * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
   * value 0 otherwise
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray isFinite(INDArray x) {
    NDValidation.validateNumerical("isFinite", "x", x);
    return Nd4j.exec(new IsFinite(x));
  }

  /**
   * Is infinite operation: elementwise isInfinite(x)
   * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
   * value 0 otherwise
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray isInfinite(INDArray x) {
    NDValidation.validateNumerical("isInfinite", "x", x);
    return Nd4j.exec(new IsInf(x));
  }

  /**
   * Is maximum operation: elementwise x == max(x)
   * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
   * value 0 otherwise
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray isMax(INDArray x) {
    NDValidation.validateNumerical("isMax", "x", x);
    INDArray[] __tmp = Nd4j.exec(new IsMax(x));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Is Not a Number operation: elementwise isNaN(x)
   * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
   * value 0 otherwise
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray isNaN(INDArray x) {
    NDValidation.validateNumerical("isNaN", "x", x);
    return Nd4j.exec(new IsNaN(x));
  }

  /**
   * Is the array non decreasing?
   * An array is non-decreasing if for every valid i, x[i] &lt;= x[i+1]. For Rank 2+ arrays, values are compared
   * in 'c' (row major) order
   *
   * @param x Input variable (NUMERIC type)
   * @return output Scalar variable with value 1 if non-decreasing, or 0 otherwise (NUMERIC type)
   */
  public INDArray isNonDecreasing(INDArray x) {
    NDValidation.validateNumerical("isNonDecreasing", "x", x);
    INDArray[] __tmp = Nd4j.exec(new IsNonDecreasing(x));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Is the array strictly increasing?
   * An array is strictly increasing if for every valid i, x[i] &lt; x[i+1]. For Rank 2+ arrays, values are compared
   * in 'c' (row major) order
   *
   * @param x Input variable (NUMERIC type)
   * @return output Scalar variable with value 1 if strictly increasing, or 0 otherwise (NUMERIC type)
   */
  public INDArray isStrictlyIncreasing(INDArray x) {
    NDValidation.validateNumerical("isStrictlyIncreasing", "x", x);
    INDArray[] __tmp = Nd4j.exec(new IsStrictlyIncreasing(x));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Jaccard similarity reduction operation. The output contains the Jaccard distance for each
   *                 tensor along the specified dimensions.
   *
   * @param x Input variable x (NUMERIC type)
   * @param y Input variable y (NUMERIC type)
   * @param keepDims Whether to preserve original dimensions or not
   * @param isComplex Depending on the implementation, such as distance calculations, this can determine whether all distance calculations for all points should be done.
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray jaccardDistance(INDArray x, INDArray y, boolean keepDims, boolean isComplex,
      long... dimensions) {
    NDValidation.validateNumerical("jaccardDistance", "x", x);
    NDValidation.validateNumerical("jaccardDistance", "y", y);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new JaccardDistance(x, y, keepDims, isComplex, dimensions));
  }

  /**
   * Jaccard similarity reduction operation. The output contains the Jaccard distance for each
   *                 tensor along the specified dimensions.
   *
   * @param x Input variable x (NUMERIC type)
   * @param y Input variable y (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray jaccardDistance(INDArray x, INDArray y, long... dimensions) {
    NDValidation.validateNumerical("jaccardDistance", "x", x);
    NDValidation.validateNumerical("jaccardDistance", "y", y);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new JaccardDistance(x, y, false, false, dimensions));
  }

  /**
   * Last index reduction operation.
   * Returns a variable that contains the index of the last element that matches the specified condition (for each
   * slice along the specified dimensions)
   * Note that if keepDims = true, the output variable has the same rank as the input variable,
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
   * the mean along a dimension).
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
   * keepDims = true: [a,1,c]
   * keepDims = false: [a,c]
   *
   * @param in Input variable (NUMERIC type)
   * @param condition Condition to check on input variable
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=1))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray lastIndex(INDArray in, Condition condition, long... dimensions) {
    NDValidation.validateNumerical("lastIndex", "in", in);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    return Nd4j.exec(new LastIndex(in, false, condition, dimensions));
  }

  /**
   * Last index reduction operation.
   * Returns a variable that contains the index of the last element that matches the specified condition (for each
   * slice along the specified dimensions)
   * Note that if keepDims = true, the output variable has the same rank as the input variable,
   * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
   * the mean along a dimension).
   * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
   * keepDims = true: [a,1,c]
   * keepDims = false: [a,c]
   *
   * @param in Input variable (NUMERIC type)
   * @param condition Condition to check on input variable
   * @param keepDims If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=1))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray lastIndex(INDArray in, Condition condition, boolean keepDims,
      long... dimensions) {
    NDValidation.validateNumerical("lastIndex", "in", in);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    return Nd4j.exec(new LastIndex(in, keepDims, condition, dimensions));
  }

  /**
   * Calculates difference between inputs X and Y.
   *
   * @param x Input variable X (NUMERIC type)
   * @param y Input variable Y (NUMERIC type)
   * @return output1 Calculated difference between X and Y (NUMERIC type)
   * @return output2 Calculated difference between X and Y (NUMERIC type)
   */
  public INDArray[] listDiff(INDArray x, INDArray y) {
    NDValidation.validateNumerical("listDiff", "x", x);
    NDValidation.validateNumerical("listDiff", "y", y);
    return Nd4j.exec(new ListDiff(x, y));
  }

  /**
   * Element-wise logarithm function (base e - natural logarithm): out = log(x)
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray log(INDArray x) {
    NDValidation.validateNumerical("log", "x", x);
    return Nd4j.exec(new Log(x));
  }

  /**
   * Element-wise logarithm function (with specified base): out = log_{base}(x)
   *
   * @param x Input variable (NUMERIC type)
   * @param base Logarithm base
   * @return output Output variable (NUMERIC type)
   */
  public INDArray log(INDArray x, double base) {
    NDValidation.validateNumerical("log", "x", x);
    return Nd4j.exec(new LogX(x, base));
  }

  /**
   * Elementwise natural logarithm function: out = log_e (1 + x)
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray log1p(INDArray x) {
    NDValidation.validateNumerical("log1p", "x", x);
    return Nd4j.exec(new Log1p(x));
  }

  /**
   * Log entropy reduction: log(-sum(x * log(x)))
   *
   * @param in Input variable (NUMERIC type)
   * @param keepDims Whether to keep the original  dimensions or produce a shrunk array with less dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray logEntropy(INDArray in, boolean keepDims, long... dimensions) {
    NDValidation.validateNumerical("logEntropy", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new LogEntropy(in, keepDims, dimensions));
  }

  /**
   * Log entropy reduction: log(-sum(x * log(x)))
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray logEntropy(INDArray in, long... dimensions) {
    NDValidation.validateNumerical("logEntropy", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new LogEntropy(in, false, dimensions));
  }

  /**
   * Log-sum-exp reduction (optionally along dimension).
   * Computes log(sum(exp(x))
   *
   * @param input Input variable (NUMERIC type)
   * @param dimensions Optional dimensions to reduce along (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray logSumExp(INDArray input, long... dimensions) {
    NDValidation.validateNumerical("logSumExp", "input", input);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    INDArray[] __tmp = Nd4j.exec(new LogSumExp(input, dimensions));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Manhattan distance (l1 norm, l1 distance) reduction operation. The output contains the Manhattan distance for each
   * tensor/subset along the specified dimensions:
   * out = sum_i abs(x[i]-y[i])
   *
   * @param x Input variable x (NUMERIC type)
   * @param y Input variable y (NUMERIC type)
   * @param keepDims Whether to preserve original dimensions or not
   * @param isComplex Depending on the implementation, such as distance calculations, this can determine whether all distance calculations for all points should be done.
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray manhattanDistance(INDArray x, INDArray y, boolean keepDims, boolean isComplex,
      long... dimensions) {
    NDValidation.validateNumerical("manhattanDistance", "x", x);
    NDValidation.validateNumerical("manhattanDistance", "y", y);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new ManhattanDistance(x, y, keepDims, isComplex, dimensions));
  }

  /**
   * Manhattan distance (l1 norm, l1 distance) reduction operation. The output contains the Manhattan distance for each
   * tensor/subset along the specified dimensions:
   * out = sum_i abs(x[i]-y[i])
   *
   * @param x Input variable x (NUMERIC type)
   * @param y Input variable y (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray manhattanDistance(INDArray x, INDArray y, long... dimensions) {
    NDValidation.validateNumerical("manhattanDistance", "x", x);
    NDValidation.validateNumerical("manhattanDistance", "y", y);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new ManhattanDistance(x, y, false, false, dimensions));
  }

  /**
   * Matrix determinant op. For 2D input, this returns the standard matrix determinant.
   * For higher dimensional input with shape [..., m, m] the matrix determinant is returned for each
   * shape [m,m] sub-matrix.
   *
   * @param in Input (NUMERIC type)
   * @return output Matrix determinant variable (NUMERIC type)
   */
  public INDArray matrixDeterminant(INDArray in) {
    NDValidation.validateNumerical("matrixDeterminant", "in", in);
    INDArray[] __tmp = Nd4j.exec(new MatrixDeterminant(in));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Matrix inverse op. For 2D input, this returns the standard matrix inverse.
   * For higher dimensional input with shape [..., m, m] the matrix inverse is returned for each
   * shape [m,m] sub-matrix.
   *
   * @param in Input (NUMERIC type)
   * @return output Matrix inverse variable (NUMERIC type)
   */
  public INDArray matrixInverse(INDArray in) {
    NDValidation.validateNumerical("matrixInverse", "in", in);
    INDArray[] __tmp = Nd4j.exec(new MatrixInverse(in));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Pairwise max operation, out = max(x, y)
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
   *
   * @param x First input variable, x (NUMERIC type)
   * @param y Second input variable, y (NUMERIC type)
   * @return out Output (NUMERIC type)
   */
  public INDArray max(INDArray x, INDArray y) {
    NDValidation.validateNumerical("max", "x", x);
    NDValidation.validateNumerical("max", "y", y);
    INDArray[] __tmp = Nd4j.exec(new Max(x, y));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Mean array reduction operation, optionally along specified dimensions: out = mean(abs(x))
   *
   * @param in Input variable (NUMERIC type)
   * @param keepDims Whether to keep the original  dimensions or produce a shrunk array with less dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray mean(INDArray in, boolean keepDims, long... dimensions) {
    NDValidation.validateNumerical("mean", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new Mean(in, keepDims, dimensions));
  }

  /**
   * Mean array reduction operation, optionally along specified dimensions: out = mean(abs(x))
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray mean(INDArray in, long... dimensions) {
    NDValidation.validateNumerical("mean", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new Mean(in, false, dimensions));
  }

  /**
   * Mean array reduction operation, optionally along specified dimensions: out = mean(abs(x))
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce along (NUMERIC type)
   * @param keepDims Whether to keep the original  dimensions or produce a shrunk array with less dimensions
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray mean(INDArray in, INDArray dimensions, boolean keepDims) {
    NDValidation.validateNumerical("mean", "in", in);
    NDValidation.validateNumerical("mean", "dimensions", dimensions);
    return Nd4j.exec(new Mean(in, dimensions, keepDims));
  }

  /**
   * Mean array reduction operation, optionally along specified dimensions: out = mean(abs(x))
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce along (NUMERIC type)
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray mean(INDArray in, INDArray dimensions) {
    NDValidation.validateNumerical("mean", "in", in);
    NDValidation.validateNumerical("mean", "dimensions", dimensions);
    return Nd4j.exec(new Mean(in, dimensions, false));
  }

  /**
   * Merge add function: merges an arbitrary number of equal shaped arrays using element-wise addition:
   * out = sum_i in[i]
   *
   * @param inputs Input variables (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray mergeAdd(INDArray... inputs) {
    NDValidation.validateNumerical("mergeAdd", "inputs", inputs);
    Preconditions.checkArgument(inputs.length >= 1, "inputs has incorrect size/length. Expected: inputs.length >= 1, got %s", inputs.length);
    INDArray[] __tmp = Nd4j.exec(new MergeAddOp(inputs));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Merge average function: merges an arbitrary number of equal shaped arrays using element-wise mean operation:
   * out = mean_i in[i]
   *
   * @param inputs Input variables (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray mergeAvg(INDArray... inputs) {
    NDValidation.validateNumerical("mergeAvg", "inputs", inputs);
    Preconditions.checkArgument(inputs.length >= 1, "inputs has incorrect size/length. Expected: inputs.length >= 1, got %s", inputs.length);
    INDArray[] __tmp = Nd4j.exec(new MergeAvg(inputs));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Merge max function: merges an arbitrary number of equal shaped arrays using element-wise maximum operation:
   * out = max_i in[i]
   *
   * @param inputs Input variables (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray mergeMax(INDArray... inputs) {
    NDValidation.validateNumerical("mergeMax", "inputs", inputs);
    Preconditions.checkArgument(inputs.length >= 1, "inputs has incorrect size/length. Expected: inputs.length >= 1, got %s", inputs.length);
    INDArray[] __tmp = Nd4j.exec(new MergeMax(inputs));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Broadcasts parameters for evaluation on an N-D grid.
   *
   * @param inputs  (NUMERIC type)
   * @param cartesian
   * @return output1 Output array (NUMERIC type)
   * @return output2 Output array (NUMERIC type)
   */
  public INDArray[] meshgrid(INDArray[] inputs, boolean cartesian) {
    NDValidation.validateNumerical("meshgrid", "inputs", inputs);
    Preconditions.checkArgument(inputs.length >= 0, "inputs has incorrect size/length. Expected: inputs.length >= 0, got %s", inputs.length);
    return Nd4j.exec(new MeshGrid(inputs, cartesian));
  }

  /**
   * Pairwise max operation, out = min(x, y)
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
   *
   * @param x First input variable, x (NUMERIC type)
   * @param y Second input variable, y (NUMERIC type)
   * @return out Output (NUMERIC type)
   */
  public INDArray min(INDArray x, INDArray y) {
    NDValidation.validateNumerical("min", "x", x);
    NDValidation.validateNumerical("min", "y", y);
    INDArray[] __tmp = Nd4j.exec(new Min(x, y));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Pairwise modulus (remainder) operation, out = x % y
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
   *
   * @param x Input variable (NUMERIC type)
   * @param y Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray mod(INDArray x, INDArray y) {
    NDValidation.validateNumerical("mod", "x", x);
    NDValidation.validateNumerical("mod", "y", y);
    INDArray[] __tmp = Nd4j.exec(new ModOp(x, y));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Calculate the mean and (population) variance for the input variable, for the specified axis
   *
   * @param input Input to calculate moments for (NUMERIC type)
   * @param axes Dimensions to perform calculation over (Size: AtLeast(min=0))
   * @param keepDims Whether to keep dimensions during reduction or not.
   * @return output_mean Mean variable (NUMERIC type)
   * @return output_variance Variance variable (NUMERIC type)
   */
  public INDArray[] moments(INDArray input, long[] axes, boolean keepDims) {
    NDValidation.validateNumerical("moments", "input", input);
    Preconditions.checkArgument(axes.length >= 0, "axes has incorrect size/length. Expected: axes.length >= 0, got %s", axes.length);
    return Nd4j.exec(new Moments(input, axes, keepDims));
  }

  /**
   * Calculate the mean and (population) variance for the input variable, for the specified axis
   *
   * @param input Input to calculate moments for (NUMERIC type)
   * @param axes Dimensions to perform calculation over (NUMERIC type)
   * @param keepDims Whether to keep dimensions during reduction or not.
   * @return output_mean Mean variable (NUMERIC type)
   * @return output_variance Variance variable (NUMERIC type)
   */
  public INDArray[] moments(INDArray input, INDArray axes, boolean keepDims) {
    NDValidation.validateNumerical("moments", "input", input);
    NDValidation.validateNumerical("moments", "axes", axes);
    return Nd4j.exec(new Moments(input, axes, keepDims));
  }

  /**
   * Pairwise multiplication operation, out = x * y
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
   *
   * @param x Input variable (NUMERIC type)
   * @param y Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray mul(INDArray x, INDArray y) {
    NDValidation.validateNumerical("mul", "x", x);
    NDValidation.validateNumerical("mul", "y", y);
    INDArray[] __tmp = Nd4j.exec(new MulOp(x, y));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Scalar multiplication operation, out = in * scalar
   *
   * @param x Input variable (NUMERIC type)
   * @param value Scalar value for op
   * @return output Output variable (NUMERIC type)
   */
  public INDArray mul(INDArray x, double value) {
    NDValidation.validateNumerical("mul", "x", x);
    return Nd4j.exec(new ScalarMultiplication(x, value));
  }

  /**
   * Elementwise negative operation: out = -x
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray neg(INDArray x) {
    NDValidation.validateNumerical("neg", "x", x);
    return Nd4j.exec(new Negative(x));
  }

  /**
   * Mean array reduction operation, optionally along specified dimensions: out = mean(abs(x))
   *
   * @param in Input variable (NUMERIC type)
   * @param keepDims Whether to keep the original  dimensions or produce a shrunk array with less dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray norm1(INDArray in, boolean keepDims, long... dimensions) {
    NDValidation.validateNumerical("norm1", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new Norm1(in, keepDims, dimensions));
  }

  /**
   * Mean array reduction operation, optionally along specified dimensions: out = mean(abs(x))
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray norm1(INDArray in, long... dimensions) {
    NDValidation.validateNumerical("norm1", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new Norm1(in, false, dimensions));
  }

  /**
   * Sum of absolute differences.
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce along (NUMERIC type)
   * @param keepDims Whether to keep the original  dimensions or produce a shrunk array with less dimensions
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray norm1(INDArray in, INDArray dimensions, boolean keepDims) {
    NDValidation.validateNumerical("norm1", "in", in);
    NDValidation.validateNumerical("norm1", "dimensions", dimensions);
    return Nd4j.exec(new Norm1(in, dimensions, keepDims));
  }

  /**
   * Sum of absolute differences.
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce along (NUMERIC type)
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray norm1(INDArray in, INDArray dimensions) {
    NDValidation.validateNumerical("norm1", "in", in);
    NDValidation.validateNumerical("norm1", "dimensions", dimensions);
    return Nd4j.exec(new Norm1(in, dimensions, false));
  }

  /**
   * Euclidean norm: euclidean distance of a vector from the origin
   *
   * @param in Input variable (NUMERIC type)
   * @param keepDims Whether to keep the original  dimensions or produce a shrunk array with less dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray norm2(INDArray in, boolean keepDims, long... dimensions) {
    NDValidation.validateNumerical("norm2", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new Norm2(in, keepDims, dimensions));
  }

  /**
   * Euclidean norm: euclidean distance of a vector from the origin
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray norm2(INDArray in, long... dimensions) {
    NDValidation.validateNumerical("norm2", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new Norm2(in, false, dimensions));
  }

  /**
   * Euclidean norm: euclidean distance of a vector from the origin
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce along (NUMERIC type)
   * @param keepDims Whether to keep the original  dimensions or produce a shrunk array with less dimensions
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray norm2(INDArray in, INDArray dimensions, boolean keepDims) {
    NDValidation.validateNumerical("norm2", "in", in);
    NDValidation.validateNumerical("norm2", "dimensions", dimensions);
    return Nd4j.exec(new Norm2(in, dimensions, keepDims));
  }

  /**
   * Euclidean norm: euclidean distance of a vector from the origin
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce along (NUMERIC type)
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray norm2(INDArray in, INDArray dimensions) {
    NDValidation.validateNumerical("norm2", "in", in);
    NDValidation.validateNumerical("norm2", "dimensions", dimensions);
    return Nd4j.exec(new Norm2(in, dimensions, false));
  }

  /**
   * Differences between max absolute value
   *
   * @param in Input variable (NUMERIC type)
   * @param keepDims Whether to keep the original  dimensions or produce a shrunk array with less dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray normMax(INDArray in, boolean keepDims, long... dimensions) {
    NDValidation.validateNumerical("normMax", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new NormMax(in, keepDims, dimensions));
  }

  /**
   * Differences between max absolute value
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray normMax(INDArray in, long... dimensions) {
    NDValidation.validateNumerical("normMax", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new NormMax(in, false, dimensions));
  }

  /**
   * Differences between max absolute value
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce along (NUMERIC type)
   * @param keepDims Whether to keep the original  dimensions or produce a shrunk array with less dimensions
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray normMax(INDArray in, INDArray dimensions, boolean keepDims) {
    NDValidation.validateNumerical("normMax", "in", in);
    NDValidation.validateNumerical("normMax", "dimensions", dimensions);
    return Nd4j.exec(new NormMax(in, dimensions, keepDims));
  }

  /**
   * Differences between max absolute value
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce along (NUMERIC type)
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray normMax(INDArray in, INDArray dimensions) {
    NDValidation.validateNumerical("normMax", "in", in);
    NDValidation.validateNumerical("normMax", "dimensions", dimensions);
    return Nd4j.exec(new NormMax(in, dimensions, false));
  }

  /**
   * Calculate the mean and variance from the sufficient statistics
   *
   * @param counts Rank 0 (scalar) value with the total number of values used to calculate the sufficient statistics (NUMERIC type)
   * @param means Mean-value sufficient statistics: this is the SUM of all data values (NUMERIC type)
   * @param variances Variaance sufficient statistics: this is the squared sum of all data values (NUMERIC type)
   * @param shift Shift value, possibly 0, used when calculating the sufficient statistics (for numerical stability)
   * @return output_mean Mean variable (NUMERIC type)
   * @return output_population Population variable (NUMERIC type)
   */
  public INDArray[] normalizeMoments(INDArray counts, INDArray means, INDArray variances,
      double shift) {
    NDValidation.validateNumerical("normalizeMoments", "counts", counts);
    NDValidation.validateNumerical("normalizeMoments", "means", means);
    NDValidation.validateNumerical("normalizeMoments", "variances", variances);
    return Nd4j.exec(new NormalizeMoments(counts, means, variances, shift));
  }

  /**
   * Boolean OR operation: elementwise (x != 0) || (y != 0)
   * If x and y arrays have equal shape, the output shape is the same as these inputs.
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.
   * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
   *
   * @param x Input 1 (BOOL type)
   * @param y Input 2 (BOOL type)
   * @return output INDArray with values 0 and 1 based on where the condition is satisfied (BOOL type)
   */
  public INDArray or(INDArray x, INDArray y) {
    NDValidation.validateBool("or", "x", x);
    NDValidation.validateBool("or", "y", y);
    INDArray[] __tmp = Nd4j.exec(new LogicalOr(x, y));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Element-wise power function: out = x^value
   *
   * @param x Input variable (NUMERIC type)
   * @param value Scalar value for op
   * @return output Output variable (NUMERIC type)
   */
  public INDArray pow(INDArray x, double value) {
    NDValidation.validateNumerical("pow", "x", x);
    return Nd4j.exec(new Pow(x, value));
  }

  /**
   * Element-wise (broadcastable) power function: out = x[i]^y[i]
   *
   * @param x Input variable (NUMERIC type)
   * @param y Power (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray pow(INDArray x, INDArray y) {
    NDValidation.validateNumerical("pow", "x", x);
    NDValidation.validateNumerical("pow", "y", y);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.Pow(x, y));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * The max of an array along each dimension
   *
   * @param in Input variable (NUMERIC type)
   * @param keepDims Whether to keep the original  dimensions or produce a shrunk array with less dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray prod(INDArray in, boolean keepDims, long... dimensions) {
    NDValidation.validateNumerical("prod", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new Prod(in, keepDims, dimensions));
  }

  /**
   * The max of an array along each dimension
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray prod(INDArray in, long... dimensions) {
    NDValidation.validateNumerical("prod", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new Prod(in, false, dimensions));
  }

  /**
   * The product of an array long each dimension
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce along (NUMERIC type)
   * @param keepDims Whether to keep the original  dimensions or produce a shrunk array with less dimensions
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray prod(INDArray in, INDArray dimensions, boolean keepDims) {
    NDValidation.validateNumerical("prod", "in", in);
    NDValidation.validateNumerical("prod", "dimensions", dimensions);
    return Nd4j.exec(new Prod(in, dimensions, keepDims));
  }

  /**
   * The product of an array long each dimension
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce along (NUMERIC type)
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray prod(INDArray in, INDArray dimensions) {
    NDValidation.validateNumerical("prod", "in", in);
    NDValidation.validateNumerical("prod", "dimensions", dimensions);
    return Nd4j.exec(new Prod(in, dimensions, false));
  }

  /**
   * Rational Tanh Approximation elementwise function, as described in the paper:
   * Compact Convolutional Neural Network Cascade for Face Detection
   * This is a faster Tanh approximation
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray rationalTanh(INDArray x) {
    NDValidation.validateNumerical("rationalTanh", "x", x);
    return Nd4j.exec(new RationalTanh(x));
  }

  /**
   * Pairwise reverse division operation, out = y / x
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
   *
   * @param x Input variable (NUMERIC type)
   * @param y Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray rdiv(INDArray x, INDArray y) {
    NDValidation.validateNumerical("rdiv", "x", x);
    NDValidation.validateNumerical("rdiv", "y", y);
    INDArray[] __tmp = Nd4j.exec(new RDivOp(x, y));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Scalar reverse division operation, out = scalar / in
   *
   * @param x Input variable (NUMERIC type)
   * @param value Scalar value for op
   * @return output Output variable (NUMERIC type)
   */
  public INDArray rdiv(INDArray x, double value) {
    NDValidation.validateNumerical("rdiv", "x", x);
    return Nd4j.exec(new ScalarReverseDivision(x, value));
  }

  /**
   * Element-wise reciprocal (inverse) function: out[i] = 1 / in[i]
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray reciprocal(INDArray x) {
    NDValidation.validateNumerical("reciprocal", "x", x);
    return Nd4j.exec(new Reciprocal(x));
  }

  /**
   * Rectified tanh operation: max(0, tanh(in))
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray rectifiedTanh(INDArray x) {
    NDValidation.validateNumerical("rectifiedTanh", "x", x);
    return Nd4j.exec(new RectifiedTanh(x));
  }

  /**
   * Absolute max array reduction operation, optionally along specified dimensions: out = max(abs(x))
   *
   * @param in Input variable (NUMERIC type)
   * @param keepDims Whether to keep the original  dimensions or produce a shrunk array with less dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray reduceAMax(INDArray in, boolean keepDims, long... dimensions) {
    NDValidation.validateNumerical("reduceAMax", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new AMax(in, keepDims, dimensions));
  }

  /**
   * Absolute max array reduction operation, optionally along specified dimensions: out = max(abs(x))
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray reduceAMax(INDArray in, long... dimensions) {
    NDValidation.validateNumerical("reduceAMax", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new AMax(in, false, dimensions));
  }

  /**
   * Absolute max array reduction operation, optionally along specified dimensions: out = max(abs(x))
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce along (NUMERIC type)
   * @param keepDims Whether to keep the original  dimensions or produce a shrunk array with less dimensions
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray reduceAMax(INDArray in, INDArray dimensions, boolean keepDims) {
    NDValidation.validateNumerical("reduceAMax", "in", in);
    NDValidation.validateNumerical("reduceAMax", "dimensions", dimensions);
    return Nd4j.exec(new AMax(in, dimensions, keepDims));
  }

  /**
   * Absolute max array reduction operation, optionally along specified dimensions: out = max(abs(x))
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce along (NUMERIC type)
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray reduceAMax(INDArray in, INDArray dimensions) {
    NDValidation.validateNumerical("reduceAMax", "in", in);
    NDValidation.validateNumerical("reduceAMax", "dimensions", dimensions);
    return Nd4j.exec(new AMax(in, dimensions, false));
  }

  /**
   * Absolute mean array reduction operation, optionally along specified dimensions: out = mean(abs(x))
   *
   * @param in Input variable (NUMERIC type)
   * @param keepDims Whether to keep the original  dimensions or produce a shrunk array with less dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray reduceAmean(INDArray in, boolean keepDims, long... dimensions) {
    NDValidation.validateNumerical("reduceAmean", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new AMean(in, keepDims, dimensions));
  }

  /**
   * Absolute mean array reduction operation, optionally along specified dimensions: out = mean(abs(x))
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray reduceAmean(INDArray in, long... dimensions) {
    NDValidation.validateNumerical("reduceAmean", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new AMean(in, false, dimensions));
  }

  /**
   * Absolute mean array reduction operation, optionally along specified dimensions: out = mean(abs(x))
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce along (NUMERIC type)
   * @param keepDims Whether to keep the original  dimensions or produce a shrunk array with less dimensions
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray reduceAmean(INDArray in, INDArray dimensions, boolean keepDims) {
    NDValidation.validateNumerical("reduceAmean", "in", in);
    NDValidation.validateNumerical("reduceAmean", "dimensions", dimensions);
    return Nd4j.exec(new AMean(in, dimensions, keepDims));
  }

  /**
   * Absolute mean array reduction operation, optionally along specified dimensions: out = mean(abs(x))
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce along (NUMERIC type)
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray reduceAmean(INDArray in, INDArray dimensions) {
    NDValidation.validateNumerical("reduceAmean", "in", in);
    NDValidation.validateNumerical("reduceAmean", "dimensions", dimensions);
    return Nd4j.exec(new AMean(in, dimensions, false));
  }

  /**
   * Absolute min array reduction operation, optionally along specified dimensions: out = min(abs(x))
   *
   * @param in Input variable (NUMERIC type)
   * @param keepDims Whether to keep the original  dimensions or produce a shrunk array with less dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray reduceAmin(INDArray in, boolean keepDims, long... dimensions) {
    NDValidation.validateNumerical("reduceAmin", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new AMin(in, keepDims, dimensions));
  }

  /**
   * Absolute min array reduction operation, optionally along specified dimensions: out = min(abs(x))
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray reduceAmin(INDArray in, long... dimensions) {
    NDValidation.validateNumerical("reduceAmin", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new AMin(in, false, dimensions));
  }

  /**
   * Absolute min array reduction operation, optionally along specified dimensions: out = min(abs(x))
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce along (NUMERIC type)
   * @param keepDims Whether to keep the original  dimensions or produce a shrunk array with less dimensions
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray reduceAmin(INDArray in, INDArray dimensions, boolean keepDims) {
    NDValidation.validateNumerical("reduceAmin", "in", in);
    NDValidation.validateNumerical("reduceAmin", "dimensions", dimensions);
    return Nd4j.exec(new AMin(in, dimensions, keepDims));
  }

  /**
   * Absolute min array reduction operation, optionally along specified dimensions: out = min(abs(x))
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce along (NUMERIC type)
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray reduceAmin(INDArray in, INDArray dimensions) {
    NDValidation.validateNumerical("reduceAmin", "in", in);
    NDValidation.validateNumerical("reduceAmin", "dimensions", dimensions);
    return Nd4j.exec(new AMin(in, dimensions, false));
  }

  /**
   * The max of an array along each dimension
   *
   * @param in Input variable (NUMERIC type)
   * @param keepDims Whether to keep the original  dimensions or produce a shrunk array with less dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray reduceMax(INDArray in, boolean keepDims, long... dimensions) {
    NDValidation.validateNumerical("reduceMax", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.same.Max(in, keepDims, dimensions));
  }

  /**
   * The max of an array along each dimension
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray reduceMax(INDArray in, long... dimensions) {
    NDValidation.validateNumerical("reduceMax", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.same.Max(in, false, dimensions));
  }

  /**
   * The max of an array long each dimension
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce along (NUMERIC type)
   * @param keepDims Whether to keep the original  dimensions or produce a shrunk array with less dimensions
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray reduceMax(INDArray in, INDArray dimensions, boolean keepDims) {
    NDValidation.validateNumerical("reduceMax", "in", in);
    NDValidation.validateNumerical("reduceMax", "dimensions", dimensions);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.same.Max(in, dimensions, keepDims));
  }

  /**
   * The max of an array long each dimension
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce along (NUMERIC type)
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray reduceMax(INDArray in, INDArray dimensions) {
    NDValidation.validateNumerical("reduceMax", "in", in);
    NDValidation.validateNumerical("reduceMax", "dimensions", dimensions);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.same.Max(in, dimensions, false));
  }

  /**
   * The minimum of an array along each dimension
   *
   * @param in Input variable (NUMERIC type)
   * @param keepDims Whether to keep the original  dimensions or produce a shrunk array with less dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray reduceMin(INDArray in, boolean keepDims, long... dimensions) {
    NDValidation.validateNumerical("reduceMin", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.same.Min(in, keepDims, dimensions));
  }

  /**
   * The minimum of an array along each dimension
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray reduceMin(INDArray in, long... dimensions) {
    NDValidation.validateNumerical("reduceMin", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.same.Min(in, false, dimensions));
  }

  /**
   * The minimum of an array long each dimension
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce along (NUMERIC type)
   * @param keepDims Whether to keep the original  dimensions or produce a shrunk array with less dimensions
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray reduceMin(INDArray in, INDArray dimensions, boolean keepDims) {
    NDValidation.validateNumerical("reduceMin", "in", in);
    NDValidation.validateNumerical("reduceMin", "dimensions", dimensions);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.same.Min(in, dimensions, keepDims));
  }

  /**
   * The minimum of an array long each dimension
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce along (NUMERIC type)
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray reduceMin(INDArray in, INDArray dimensions) {
    NDValidation.validateNumerical("reduceMin", "in", in);
    NDValidation.validateNumerical("reduceMin", "dimensions", dimensions);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.same.Min(in, dimensions, false));
  }

  /**
   * Element-wise round function: out = round(x).
   * Rounds (up or down depending on value) to the nearest integer value.
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray round(INDArray x) {
    NDValidation.validateNumerical("round", "x", x);
    return Nd4j.exec(new Round(x));
  }

  /**
   * Element-wise reciprocal (inverse) of square root: out = 1.0 / sqrt(x)
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray rsqrt(INDArray x) {
    NDValidation.validateNumerical("rsqrt", "x", x);
    return Nd4j.exec(new RSqrt(x));
  }

  /**
   * Pairwise reverse subtraction operation, out = y - x
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
   *
   * @param x Input variable (NUMERIC type)
   * @param y Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray rsub(INDArray x, INDArray y) {
    NDValidation.validateNumerical("rsub", "x", x);
    NDValidation.validateNumerical("rsub", "y", y);
    INDArray[] __tmp = Nd4j.exec(new RSubOp(x, y));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Scalar reverse subtraction operation, out = scalar - in
   *
   * @param x Input variable (NUMERIC type)
   * @param value Scalar value for op
   * @return output Output variable (NUMERIC type)
   */
  public INDArray rsub(INDArray x, double value) {
    NDValidation.validateNumerical("rsub", "x", x);
    return Nd4j.exec(new ScalarReverseSubtraction(x, value));
  }

  /**
   * Set the diagonal value to the specified values
   * If input is
   * [ a, b, c]
   * [ d, e, f]
   * [ g, h, i]
   * and diag = [ 1, 2, 3] then output is
   * [ 1, b, c]
   * [ d, 2, f]
   * [ g, h, 3]
   *
   * @param in Input variable (NUMERIC type)
   * @param diag Diagonal (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray setDiag(INDArray in, INDArray diag) {
    NDValidation.validateNumerical("setDiag", "in", in);
    NDValidation.validateNumerical("setDiag", "diag", diag);
    INDArray[] __tmp = Nd4j.exec(new MatrixSetDiag(in, diag));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Shannon Entropy reduction: -sum(x * log2(x))
   *
   * @param in Input variable (NUMERIC type)
   * @param keepDims Whether to keep the original  dimensions or produce a shrunk array with less dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray shannonEntropy(INDArray in, boolean keepDims, long... dimensions) {
    NDValidation.validateNumerical("shannonEntropy", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new ShannonEntropy(in, keepDims, dimensions));
  }

  /**
   * Shannon Entropy reduction: -sum(x * log2(x))
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray shannonEntropy(INDArray in, long... dimensions) {
    NDValidation.validateNumerical("shannonEntropy", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new ShannonEntropy(in, false, dimensions));
  }

  /**
   * Shannon Entropy reduction: -sum(x * log2(x))
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce along (NUMERIC type)
   * @param keepDims Whether to keep the original  dimensions or produce a shrunk array with less dimensions
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray shannonEntropy(INDArray in, INDArray dimensions, boolean keepDims) {
    NDValidation.validateNumerical("shannonEntropy", "in", in);
    NDValidation.validateNumerical("shannonEntropy", "dimensions", dimensions);
    return Nd4j.exec(new ShannonEntropy(in, dimensions, keepDims));
  }

  /**
   * Shannon Entropy reduction: -sum(x * log2(x))
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce along (NUMERIC type)
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray shannonEntropy(INDArray in, INDArray dimensions) {
    NDValidation.validateNumerical("shannonEntropy", "in", in);
    NDValidation.validateNumerical("shannonEntropy", "dimensions", dimensions);
    return Nd4j.exec(new ShannonEntropy(in, dimensions, false));
  }

  /**
   * Element-wise sign (signum) function:
   * out = -1 if in &lt; 0
   * out = 0 if in = 0
   * out = 1 if in > 0
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray sign(INDArray x) {
    NDValidation.validateNumerical("sign", "x", x);
    return Nd4j.exec(new Sign(x));
  }

  /**
   * Elementwise sine operation: out = sin(x)
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray sin(INDArray x) {
    NDValidation.validateNumerical("sin", "x", x);
    return Nd4j.exec(new Sin(x));
  }

  /**
   * Elementwise sinh (hyperbolic sine) operation: out = sinh(x)
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray sinh(INDArray x) {
    NDValidation.validateNumerical("sinh", "x", x);
    return Nd4j.exec(new Sinh(x));
  }

  /**
   * Element-wise square root function: out = sqrt(x)
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray sqrt(INDArray x) {
    NDValidation.validateNumerical("sqrt", "x", x);
    return Nd4j.exec(new Sqrt(x));
  }

  /**
   * Element-wise square function: out = x^2
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray square(INDArray x) {
    NDValidation.validateNumerical("square", "x", x);
    return Nd4j.exec(new Square(x));
  }

  /**
   * Pairwise squared difference operation.
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
   *
   * @param x Input variable (NUMERIC type)
   * @param y Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray squaredDifference(INDArray x, INDArray y) {
    NDValidation.validateNumerical("squaredDifference", "x", x);
    NDValidation.validateNumerical("squaredDifference", "y", y);
    INDArray[] __tmp = Nd4j.exec(new SquaredDifferenceOp(x, y));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Sum of squared differences.
   *
   * @param in Input variable (NUMERIC type)
   * @param keepDims Whether to keep the original  dimensions or produce a shrunk array with less dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray squaredNorm(INDArray in, boolean keepDims, long... dimensions) {
    NDValidation.validateNumerical("squaredNorm", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new SquaredNorm(in, keepDims, dimensions));
  }

  /**
   * Sum of squared differences.
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray squaredNorm(INDArray in, long... dimensions) {
    NDValidation.validateNumerical("squaredNorm", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new SquaredNorm(in, false, dimensions));
  }

  /**
   * Sum of squared differences.
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce along (NUMERIC type)
   * @param keepDims Whether to keep the original  dimensions or produce a shrunk array with less dimensions
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray squaredNorm(INDArray in, INDArray dimensions, boolean keepDims) {
    NDValidation.validateNumerical("squaredNorm", "in", in);
    NDValidation.validateNumerical("squaredNorm", "dimensions", dimensions);
    return Nd4j.exec(new SquaredNorm(in, dimensions, keepDims));
  }

  /**
   * Sum of squared differences.
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce along (NUMERIC type)
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray squaredNorm(INDArray in, INDArray dimensions) {
    NDValidation.validateNumerical("squaredNorm", "in", in);
    NDValidation.validateNumerical("squaredNorm", "dimensions", dimensions);
    return Nd4j.exec(new SquaredNorm(in, dimensions, false));
  }

  /**
   * Standardize input variable along given axis
   * <p>
   * out = (x - mean) / stdev
   * <p>
   * with mean and stdev being calculated along the given dimension.
   * <p>
   * For example: given x as a mini batch of the shape [numExamples, exampleLength]:
   * <ul>
   * <li>use dimension 1 too use the statistics (mean, stdev) for each example</li>
   * <li>use dimension 0 if you want to use the statistics for each column across all examples</li>
   * <li>use dimensions 0,1 if you want to use the statistics across all columns and examples</li>
   * </ul>
   *
   * @param x Input variable (NUMERIC type)
   * @param dimensions  (Size: AtLeast(min=1))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray standardize(INDArray x, long... dimensions) {
    NDValidation.validateNumerical("standardize", "x", x);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    INDArray[] __tmp = Nd4j.exec(new Standardize(x, dimensions));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Elementwise step function:
   * out(x) = 1 if x >= cutoff
   * out(x) = 0 otherwise
   *
   * @param x Input variable (NUMERIC type)
   * @param value Scalar value for op
   * @return output Output variable (NUMERIC type)
   */
  public INDArray step(INDArray x, double value) {
    NDValidation.validateNumerical("step", "x", x);
    return Nd4j.exec(new Step(x, value));
  }

  /**
   * Pairwise subtraction operation, out = x - y
   *
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.
   * For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]
   * Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
   *
   * @param x Input variable (NUMERIC type)
   * @param y Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray sub(INDArray x, INDArray y) {
    NDValidation.validateNumerical("sub", "x", x);
    NDValidation.validateNumerical("sub", "y", y);
    INDArray[] __tmp = Nd4j.exec(new SubOp(x, y));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Scalar subtraction operation, out = in - scalar
   *
   * @param x Input variable (NUMERIC type)
   * @param value Scalar value for op
   * @return output Output variable (NUMERIC type)
   */
  public INDArray sub(INDArray x, double value) {
    NDValidation.validateNumerical("sub", "x", x);
    return Nd4j.exec(new ScalarSubtraction(x, value));
  }

  /**
   * Sum of an array, optionally along specified dimensions: out = sum(x))
   *
   * @param in Input variable (NUMERIC type)
   * @param keepDims Whether to keep the original  dimensions or produce a shrunk array with less dimensions
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray sum(INDArray in, boolean keepDims, long... dimensions) {
    NDValidation.validateNumerical("sum", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new Sum(in, keepDims, dimensions));
  }

  /**
   * Sum of an array, optionally along specified dimensions: out = sum(x))
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed (Size: AtLeast(min=0))
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray sum(INDArray in, long... dimensions) {
    NDValidation.validateNumerical("sum", "in", in);
    Preconditions.checkArgument(dimensions.length >= 0, "dimensions has incorrect size/length. Expected: dimensions.length >= 0, got %s", dimensions.length);
    return Nd4j.exec(new Sum(in, false, dimensions));
  }

  /**
   * Sum of an array, optionally along specified dimensions: out = sum(x))
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce along (NUMERIC type)
   * @param keepDims Whether to keep the original  dimensions or produce a shrunk array with less dimensions
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray sum(INDArray in, INDArray dimensions, boolean keepDims) {
    NDValidation.validateNumerical("sum", "in", in);
    NDValidation.validateNumerical("sum", "dimensions", dimensions);
    return Nd4j.exec(new Sum(in, dimensions, keepDims));
  }

  /**
   * Sum of an array, optionally along specified dimensions: out = sum(x))
   *
   * @param in Input variable (NUMERIC type)
   * @param dimensions Dimensions to reduce along (NUMERIC type)
   * @return output Reduced array of rank (input rank - num dimensions) (NUMERIC type)
   */
  public INDArray sum(INDArray in, INDArray dimensions) {
    NDValidation.validateNumerical("sum", "in", in);
    NDValidation.validateNumerical("sum", "dimensions", dimensions);
    return Nd4j.exec(new Sum(in, dimensions, false));
  }

  /**
   * Elementwise tangent operation: out = tan(x)
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray tan(INDArray x) {
    NDValidation.validateNumerical("tan", "x", x);
    return Nd4j.exec(new Tan(x));
  }

  /**
   * Elementwise tanh (hyperbolic tangent) operation: out = tanh(x)
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray tanh(INDArray x) {
    NDValidation.validateNumerical("tanh", "x", x);
    return Nd4j.exec(new Tanh(x));
  }

  /**
   * Matrix trace operation
   * For rank 2 matrices, the output is a scalar with the trace - i.e., sum of the main diagonal.
   * For higher rank inputs, output[a,b,c] = trace(in[a,b,c,:,:])
   *
   * @param in Input variable (NUMERIC type)
   * @return output Trace (NUMERIC type)
   */
  public INDArray trace(INDArray in) {
    NDValidation.validateNumerical("trace", "in", in);
    INDArray[] __tmp = Nd4j.exec(new Trace(in));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Boolean XOR (exclusive OR) operation: elementwise (x != 0) XOR (y != 0)
   * If x and y arrays have equal shape, the output shape is the same as these inputs.
   * Note: supports broadcasting if x and y have different shapes and are broadcastable.
   * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
   *
   * @param x Input 1 (BOOL type)
   * @param y Input 2 (BOOL type)
   * @return output INDArray with values 0 and 1 based on where the condition is satisfied (BOOL type)
   */
  public INDArray xor(INDArray x, INDArray y) {
    NDValidation.validateBool("xor", "x", x);
    NDValidation.validateBool("xor", "y", y);
    INDArray[] __tmp = Nd4j.exec(new LogicalXor(x, y));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Full array zero fraction array reduction operation, optionally along specified dimensions: out = (count(x == 0) / length(x))
   *
   * @param input Input variable (NUMERIC type)
   * @return output Reduced array of rank 0 (scalar) (NUMERIC type)
   */
  public INDArray zeroFraction(INDArray input) {
    NDValidation.validateNumerical("zeroFraction", "input", input);
    INDArray[] __tmp = Nd4j.exec(new ZeroFraction(input));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }
}

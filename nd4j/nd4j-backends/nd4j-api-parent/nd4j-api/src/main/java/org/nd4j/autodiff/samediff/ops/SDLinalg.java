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
import org.nd4j.linalg.api.buffer.DataType;

public class SDLinalg extends SDOps {
  public SDLinalg(SameDiff sameDiff) {
    super(sameDiff);
  }

  /**
   * Computes the Cholesky decomposition of one or more square matrices.<br>
   *
   * @param input Input tensor with inner-most 2 dimensions forming square matrices (NUMERIC type)
   * @return output Transformed tensor (NUMERIC type)
   */
  public SDVariable cholesky(SDVariable input) {
    SDValidation.validateNumerical("Cholesky", "input", input);
    return new org.nd4j.linalg.api.ops.impl.transforms.Cholesky(sd,input).outputVariable();
  }

  /**
   * Computes the Cholesky decomposition of one or more square matrices.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input tensor with inner-most 2 dimensions forming square matrices (NUMERIC type)
   * @return output Transformed tensor (NUMERIC type)
   */
  public SDVariable cholesky(String name, SDVariable input) {
    SDValidation.validateNumerical("Cholesky", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.Cholesky(sd,input).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Solver for linear squares problems.<br>
   *
   * @param matrix input tensor (NUMERIC type)
   * @param rhs input tensor (NUMERIC type)
   * @param l2_reguralizer regularizer
   * @param fast fast mode, defaults to True
   * @return output Transformed tensor (FLOATING_POINT type)
   */
  public SDVariable lstsq(SDVariable matrix, SDVariable rhs, double l2_reguralizer, boolean fast) {
    SDValidation.validateNumerical("Lstsq", "matrix", matrix);
    SDValidation.validateNumerical("Lstsq", "rhs", rhs);
    return new org.nd4j.linalg.api.ops.custom.Lstsq(sd,matrix, rhs, l2_reguralizer, fast).outputVariable();
  }

  /**
   * Solver for linear squares problems.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param matrix input tensor (NUMERIC type)
   * @param rhs input tensor (NUMERIC type)
   * @param l2_reguralizer regularizer
   * @param fast fast mode, defaults to True
   * @return output Transformed tensor (FLOATING_POINT type)
   */
  public SDVariable lstsq(String name, SDVariable matrix, SDVariable rhs, double l2_reguralizer,
      boolean fast) {
    SDValidation.validateNumerical("Lstsq", "matrix", matrix);
    SDValidation.validateNumerical("Lstsq", "rhs", rhs);
    SDVariable out =  new org.nd4j.linalg.api.ops.custom.Lstsq(sd,matrix, rhs, l2_reguralizer, fast).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Solver for linear squares problems.<br>
   *
   * @param matrix input tensor (NUMERIC type)
   * @param rhs input tensor (NUMERIC type)
   * @param l2_reguralizer regularizer
   * @return output Transformed tensor (FLOATING_POINT type)
   */
  public SDVariable lstsq(SDVariable matrix, SDVariable rhs, double l2_reguralizer) {
    SDValidation.validateNumerical("Lstsq", "matrix", matrix);
    SDValidation.validateNumerical("Lstsq", "rhs", rhs);
    return new org.nd4j.linalg.api.ops.custom.Lstsq(sd,matrix, rhs, l2_reguralizer, true).outputVariable();
  }

  /**
   * Solver for linear squares problems.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param matrix input tensor (NUMERIC type)
   * @param rhs input tensor (NUMERIC type)
   * @param l2_reguralizer regularizer
   * @return output Transformed tensor (FLOATING_POINT type)
   */
  public SDVariable lstsq(String name, SDVariable matrix, SDVariable rhs, double l2_reguralizer) {
    SDValidation.validateNumerical("Lstsq", "matrix", matrix);
    SDValidation.validateNumerical("Lstsq", "rhs", rhs);
    SDVariable out =  new org.nd4j.linalg.api.ops.custom.Lstsq(sd,matrix, rhs, l2_reguralizer, true).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Computes LU decomposition.<br>
   *
   * @param input input tensor (NUMERIC type)
   * @return output  (FLOATING_POINT type)
   */
  public SDVariable lu(SDVariable input) {
    SDValidation.validateNumerical("Lu", "input", input);
    return new org.nd4j.linalg.api.ops.custom.Lu(sd,input).outputVariable();
  }

  /**
   * Computes LU decomposition.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input input tensor (NUMERIC type)
   * @return output  (FLOATING_POINT type)
   */
  public SDVariable lu(String name, SDVariable input) {
    SDValidation.validateNumerical("Lu", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.custom.Lu(sd,input).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Performs matrix mutiplication on input tensors.<br>
   *
   * @param a input tensor (NUMERIC type)
   * @param b input tensor (NUMERIC type)
   * @return output  (FLOATING_POINT type)
   */
  public SDVariable matmul(SDVariable a, SDVariable b) {
    SDValidation.validateNumerical("Matmul", "a", a);
    SDValidation.validateNumerical("Matmul", "b", b);
    return new org.nd4j.linalg.api.ops.impl.reduce.Mmul(sd,a, b).outputVariable();
  }

  /**
   * Performs matrix mutiplication on input tensors.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param a input tensor (NUMERIC type)
   * @param b input tensor (NUMERIC type)
   * @return output  (FLOATING_POINT type)
   */
  public SDVariable matmul(String name, SDVariable a, SDVariable b) {
    SDValidation.validateNumerical("Matmul", "a", a);
    SDValidation.validateNumerical("Matmul", "b", b);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.Mmul(sd,a, b).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Copy a tensor setting outside a central band in each innermost matrix.<br>
   *
   * @param input input tensor (NUMERIC type)
   * @param minLower lower diagonal count
   * @param maxUpper upper diagonal count
   */
  public SDVariable[] matrixBandPart(SDVariable input, int minLower, int maxUpper) {
    SDValidation.validateNumerical("MatrixBandPart", "input", input);
    return new org.nd4j.linalg.api.ops.custom.MatrixBandPart(sd,input, minLower, maxUpper).outputVariables();
  }

  /**
   * Copy a tensor setting outside a central band in each innermost matrix.<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param input input tensor (NUMERIC type)
   * @param minLower lower diagonal count
   * @param maxUpper upper diagonal count
   */
  public SDVariable[] matrixBandPart(String[] names, SDVariable input, int minLower, int maxUpper) {
    SDValidation.validateNumerical("MatrixBandPart", "input", input);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.custom.MatrixBandPart(sd,input, minLower, maxUpper).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
  }

  /**
   * Computes the QR decompositions of input matrix.<br>
   *
   * @param input input tensor (NUMERIC type)
   * @param full full matrices mode
   */
  public SDVariable[] qr(SDVariable input, boolean full) {
    SDValidation.validateNumerical("Qr", "input", input);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.Qr(sd,input, full).outputVariables();
  }

  /**
   * Computes the QR decompositions of input matrix.<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param input input tensor (NUMERIC type)
   * @param full full matrices mode
   */
  public SDVariable[] qr(String[] names, SDVariable input, boolean full) {
    SDValidation.validateNumerical("Qr", "input", input);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.Qr(sd,input, full).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
  }

  /**
   * Computes the QR decompositions of input matrix.<br>
   *
   * @param input input tensor (NUMERIC type)
   */
  public SDVariable[] qr(SDVariable input) {
    SDValidation.validateNumerical("Qr", "input", input);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.Qr(sd,input, false).outputVariables();
  }

  /**
   * Computes the QR decompositions of input matrix.<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param input input tensor (NUMERIC type)
   */
  public SDVariable[] qr(String[] names, SDVariable input) {
    SDValidation.validateNumerical("Qr", "input", input);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.Qr(sd,input, false).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
  }

  /**
   * Solver for systems of linear equations.<br>
   *
   * @param matrix input tensor (NUMERIC type)
   * @param rhs input tensor (NUMERIC type)
   * @param adjoint adjoint mode, defaults to False
   * @return output Output tensor (FLOATING_POINT type)
   */
  public SDVariable solve(SDVariable matrix, SDVariable rhs, boolean adjoint) {
    SDValidation.validateNumerical("Solve", "matrix", matrix);
    SDValidation.validateNumerical("Solve", "rhs", rhs);
    return new org.nd4j.linalg.api.ops.custom.LinearSolve(sd,matrix, rhs, adjoint).outputVariable();
  }

  /**
   * Solver for systems of linear equations.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param matrix input tensor (NUMERIC type)
   * @param rhs input tensor (NUMERIC type)
   * @param adjoint adjoint mode, defaults to False
   * @return output Output tensor (FLOATING_POINT type)
   */
  public SDVariable solve(String name, SDVariable matrix, SDVariable rhs, boolean adjoint) {
    SDValidation.validateNumerical("Solve", "matrix", matrix);
    SDValidation.validateNumerical("Solve", "rhs", rhs);
    SDVariable out =  new org.nd4j.linalg.api.ops.custom.LinearSolve(sd,matrix, rhs, adjoint).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Solver for systems of linear equations.<br>
   *
   * @param matrix input tensor (NUMERIC type)
   * @param rhs input tensor (NUMERIC type)
   * @return output Output tensor (FLOATING_POINT type)
   */
  public SDVariable solve(SDVariable matrix, SDVariable rhs) {
    SDValidation.validateNumerical("Solve", "matrix", matrix);
    SDValidation.validateNumerical("Solve", "rhs", rhs);
    return new org.nd4j.linalg.api.ops.custom.LinearSolve(sd,matrix, rhs, false).outputVariable();
  }

  /**
   * Solver for systems of linear equations.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param matrix input tensor (NUMERIC type)
   * @param rhs input tensor (NUMERIC type)
   * @return output Output tensor (FLOATING_POINT type)
   */
  public SDVariable solve(String name, SDVariable matrix, SDVariable rhs) {
    SDValidation.validateNumerical("Solve", "matrix", matrix);
    SDValidation.validateNumerical("Solve", "rhs", rhs);
    SDVariable out =  new org.nd4j.linalg.api.ops.custom.LinearSolve(sd,matrix, rhs, false).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Solver for systems of linear questions.<br>
   *
   * @param matrix input tensor (NUMERIC type)
   * @param rhs input tensor (NUMERIC type)
   * @param lower defines whether innermost matrices in matrix are lower or upper triangular
   * @param adjoint adjoint mode
   * @return output  (FLOATING_POINT type)
   */
  public SDVariable triangularSolve(SDVariable matrix, SDVariable rhs, boolean lower,
      boolean adjoint) {
    SDValidation.validateNumerical("TriangularSolve", "matrix", matrix);
    SDValidation.validateNumerical("TriangularSolve", "rhs", rhs);
    return new org.nd4j.linalg.api.ops.custom.TriangularSolve(sd,matrix, rhs, lower, adjoint).outputVariable();
  }

  /**
   * Solver for systems of linear questions.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param matrix input tensor (NUMERIC type)
   * @param rhs input tensor (NUMERIC type)
   * @param lower defines whether innermost matrices in matrix are lower or upper triangular
   * @param adjoint adjoint mode
   * @return output  (FLOATING_POINT type)
   */
  public SDVariable triangularSolve(String name, SDVariable matrix, SDVariable rhs, boolean lower,
      boolean adjoint) {
    SDValidation.validateNumerical("TriangularSolve", "matrix", matrix);
    SDValidation.validateNumerical("TriangularSolve", "rhs", rhs);
    SDVariable out =  new org.nd4j.linalg.api.ops.custom.TriangularSolve(sd,matrix, rhs, lower, adjoint).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Computes pairwise cross product.<br>
   *
   * @param a  (NUMERIC type)
   * @param b  (NUMERIC type)
   * @return output  (FLOATING_POINT type)
   */
  public SDVariable cross(SDVariable a, SDVariable b) {
    SDValidation.validateNumerical("cross", "a", a);
    SDValidation.validateNumerical("cross", "b", b);
    return new org.nd4j.linalg.api.ops.impl.shape.Cross(sd,a, b).outputVariable();
  }

  /**
   * Computes pairwise cross product.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param a  (NUMERIC type)
   * @param b  (NUMERIC type)
   * @return output  (FLOATING_POINT type)
   */
  public SDVariable cross(String name, SDVariable a, SDVariable b) {
    SDValidation.validateNumerical("cross", "a", a);
    SDValidation.validateNumerical("cross", "b", b);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.Cross(sd,a, b).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Calculates diagonal tensor.<br>
   *
   * @param input  (NUMERIC type)
   * @return output  (FLOATING_POINT type)
   */
  public SDVariable diag(SDVariable input) {
    SDValidation.validateNumerical("diag", "input", input);
    return new org.nd4j.linalg.api.ops.impl.shape.Diag(sd,input).outputVariable();
  }

  /**
   * Calculates diagonal tensor.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input  (NUMERIC type)
   * @return output  (FLOATING_POINT type)
   */
  public SDVariable diag(String name, SDVariable input) {
    SDValidation.validateNumerical("diag", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.Diag(sd,input).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Calculates diagonal tensor.<br>
   *
   * @param input  (NUMERIC type)
   * @return output  (FLOATING_POINT type)
   */
  public SDVariable diag_part(SDVariable input) {
    SDValidation.validateNumerical("diag_part", "input", input);
    return new org.nd4j.linalg.api.ops.impl.shape.DiagPart(sd,input).outputVariable();
  }

  /**
   * Calculates diagonal tensor.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input  (NUMERIC type)
   * @return output  (FLOATING_POINT type)
   */
  public SDVariable diag_part(String name, SDVariable input) {
    SDValidation.validateNumerical("diag_part", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.DiagPart(sd,input).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Calculates log of determinant.<br>
   *
   * @param input  (NUMERIC type)
   * @return output  (FLOATING_POINT type)
   */
  public SDVariable logdet(SDVariable input) {
    SDValidation.validateNumerical("logdet", "input", input);
    return new org.nd4j.linalg.api.ops.custom.Logdet(sd,input).outputVariable();
  }

  /**
   * Calculates log of determinant.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input  (NUMERIC type)
   * @return output  (FLOATING_POINT type)
   */
  public SDVariable logdet(String name, SDVariable input) {
    SDValidation.validateNumerical("logdet", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.custom.Logdet(sd,input).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable mmul(SDVariable x, SDVariable y, boolean transposeX, boolean transposeY,
      boolean transposeZ) {
    SDValidation.validateNumerical("mmul", "x", x);
    SDValidation.validateNumerical("mmul", "y", y);
    return new org.nd4j.linalg.api.ops.impl.reduce.Mmul(sd,x, y, transposeX, transposeY, transposeZ).outputVariable();
  }

  /**
   * Matrix multiplication: out = mmul(x,y)<br>
   * Supports specifying transpose argument to perform operation such as mmul(a^T, b), etc.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x First input variable (NUMERIC type)
   * @param y Second input variable (NUMERIC type)
   * @param transposeX Transpose x (first argument)
   * @param transposeY Transpose y (second argument)
   * @param transposeZ Transpose result array
   * @return output  (NUMERIC type)
   */
  public SDVariable mmul(String name, SDVariable x, SDVariable y, boolean transposeX,
      boolean transposeY, boolean transposeZ) {
    SDValidation.validateNumerical("mmul", "x", x);
    SDValidation.validateNumerical("mmul", "y", y);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.Mmul(sd,x, y, transposeX, transposeY, transposeZ).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Matrix multiplication: out = mmul(x,y)<br>
   * Supports specifying transpose argument to perform operation such as mmul(a^T, b), etc.<br>
   *
   * @param x First input variable (NUMERIC type)
   * @param y Second input variable (NUMERIC type)
   * @return output  (NUMERIC type)
   */
  public SDVariable mmul(SDVariable x, SDVariable y) {
    SDValidation.validateNumerical("mmul", "x", x);
    SDValidation.validateNumerical("mmul", "y", y);
    return new org.nd4j.linalg.api.ops.impl.reduce.Mmul(sd,x, y, false, false, false).outputVariable();
  }

  /**
   * Matrix multiplication: out = mmul(x,y)<br>
   * Supports specifying transpose argument to perform operation such as mmul(a^T, b), etc.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x First input variable (NUMERIC type)
   * @param y Second input variable (NUMERIC type)
   * @return output  (NUMERIC type)
   */
  public SDVariable mmul(String name, SDVariable x, SDVariable y) {
    SDValidation.validateNumerical("mmul", "x", x);
    SDValidation.validateNumerical("mmul", "y", y);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.reduce.Mmul(sd,x, y, false, false, false).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Calculates singular value decomposition.<br>
   *
   * @param input  (NUMERIC type)
   * @param fullUV 
   * @param computeUV 
   * @param switchNum 
   * @return output  (FLOATING_POINT type)
   */
  public SDVariable svd(SDVariable input, boolean fullUV, boolean computeUV, int switchNum) {
    SDValidation.validateNumerical("svd", "input", input);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.Svd(sd,input, fullUV, computeUV, switchNum).outputVariable();
  }

  /**
   * Calculates singular value decomposition.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input  (NUMERIC type)
   * @param fullUV 
   * @param computeUV 
   * @param switchNum 
   * @return output  (FLOATING_POINT type)
   */
  public SDVariable svd(String name, SDVariable input, boolean fullUV, boolean computeUV,
      int switchNum) {
    SDValidation.validateNumerical("svd", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.Svd(sd,input, fullUV, computeUV, switchNum).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Calculates singular value decomposition.<br>
   *
   * @param input  (NUMERIC type)
   * @param fullUV 
   * @param computeUV 
   * @return output  (FLOATING_POINT type)
   */
  public SDVariable svd(SDVariable input, boolean fullUV, boolean computeUV) {
    SDValidation.validateNumerical("svd", "input", input);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.Svd(sd,input, fullUV, computeUV, 16).outputVariable();
  }

  /**
   * Calculates singular value decomposition.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input  (NUMERIC type)
   * @param fullUV 
   * @param computeUV 
   * @return output  (FLOATING_POINT type)
   */
  public SDVariable svd(String name, SDVariable input, boolean fullUV, boolean computeUV) {
    SDValidation.validateNumerical("svd", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.Svd(sd,input, fullUV, computeUV, 16).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * An array with ones at and below the given diagonal and zeros elsewhere.<br>
   *
   * @param dataType Data type
   * @param row 
   * @param column 
   * @param diagonal 
   * @return output  (FLOATING_POINT type)
   */
  public SDVariable tri(DataType dataType, int row, int column, int diagonal) {
    return new org.nd4j.linalg.api.ops.custom.Tri(sd,dataType, row, column, diagonal).outputVariable();
  }

  /**
   * An array with ones at and below the given diagonal and zeros elsewhere.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param dataType Data type
   * @param row 
   * @param column 
   * @param diagonal 
   * @return output  (FLOATING_POINT type)
   */
  public SDVariable tri(String name, DataType dataType, int row, int column, int diagonal) {
    SDVariable out =  new org.nd4j.linalg.api.ops.custom.Tri(sd,dataType, row, column, diagonal).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * An array with ones at and below the given diagonal and zeros elsewhere.<br>
   *
   * @param row 
   * @param column 
   * @return output  (FLOATING_POINT type)
   */
  public SDVariable tri(int row, int column) {
    return new org.nd4j.linalg.api.ops.custom.Tri(sd,DataType.FLOAT, row, column, 0).outputVariable();
  }

  /**
   * An array with ones at and below the given diagonal and zeros elsewhere.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param row 
   * @param column 
   * @return output  (FLOATING_POINT type)
   */
  public SDVariable tri(String name, int row, int column) {
    SDVariable out =  new org.nd4j.linalg.api.ops.custom.Tri(sd,DataType.FLOAT, row, column, 0).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Upper triangle of an array. Return a copy of a input tensor with the elements below the k-th diagonal zeroed.<br>
   *
   * @param input  (NUMERIC type)
   * @param diag 
   * @return output  (FLOATING_POINT type)
   */
  public SDVariable triu(SDVariable input, int diag) {
    SDValidation.validateNumerical("triu", "input", input);
    return new org.nd4j.linalg.api.ops.custom.Triu(sd,input, diag).outputVariable();
  }

  /**
   * Upper triangle of an array. Return a copy of a input tensor with the elements below the k-th diagonal zeroed.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input  (NUMERIC type)
   * @param diag 
   * @return output  (FLOATING_POINT type)
   */
  public SDVariable triu(String name, SDVariable input, int diag) {
    SDValidation.validateNumerical("triu", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.custom.Triu(sd,input, diag).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Upper triangle of an array. Return a copy of a input tensor with the elements below the k-th diagonal zeroed.<br>
   *
   * @param input  (NUMERIC type)
   * @return output  (FLOATING_POINT type)
   */
  public SDVariable triu(SDVariable input) {
    SDValidation.validateNumerical("triu", "input", input);
    return new org.nd4j.linalg.api.ops.custom.Triu(sd,input, 0).outputVariable();
  }

  /**
   * Upper triangle of an array. Return a copy of a input tensor with the elements below the k-th diagonal zeroed.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input  (NUMERIC type)
   * @return output  (FLOATING_POINT type)
   */
  public SDVariable triu(String name, SDVariable input) {
    SDValidation.validateNumerical("triu", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.custom.Triu(sd,input, 0).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }
}

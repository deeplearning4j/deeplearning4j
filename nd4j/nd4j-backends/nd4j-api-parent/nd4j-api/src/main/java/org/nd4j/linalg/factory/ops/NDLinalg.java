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

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDValidation;
import org.nd4j.linalg.factory.Nd4j;

public class NDLinalg {
  public NDLinalg() {
  }

  /**
   * Computes the Cholesky decomposition of one or more square matrices.<br>
   *
   * @param input Input tensor with inner-most 2 dimensions forming square matrices (NUMERIC type)
   * @return output Transformed tensor (NUMERIC type)
   */
  public INDArray cholesky(INDArray input) {
    NDValidation.validateNumerical("Cholesky", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.Cholesky(input))[0];
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
  public INDArray lstsq(INDArray matrix, INDArray rhs, double l2_reguralizer, boolean fast) {
    NDValidation.validateNumerical("Lstsq", "matrix", matrix);
    NDValidation.validateNumerical("Lstsq", "rhs", rhs);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.custom.Lstsq(matrix, rhs, l2_reguralizer, fast))[0];
  }

  /**
   * Solver for linear squares problems.<br>
   *
   * @param matrix input tensor (NUMERIC type)
   * @param rhs input tensor (NUMERIC type)
   * @param l2_reguralizer regularizer
   * @return output Transformed tensor (FLOATING_POINT type)
   */
  public INDArray lstsq(INDArray matrix, INDArray rhs, double l2_reguralizer) {
    NDValidation.validateNumerical("Lstsq", "matrix", matrix);
    NDValidation.validateNumerical("Lstsq", "rhs", rhs);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.custom.Lstsq(matrix, rhs, l2_reguralizer, true))[0];
  }

  /**
   * Computes LU decomposition.<br>
   *
   * @param input input tensor (NUMERIC type)
   * @return output  (FLOATING_POINT type)
   */
  public INDArray lu(INDArray input) {
    NDValidation.validateNumerical("Lu", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.custom.Lu(input))[0];
  }

  /**
   * Performs matrix mutiplication on input tensors.<br>
   *
   * @param a input tensor (NUMERIC type)
   * @param b input tensor (NUMERIC type)
   * @return output  (FLOATING_POINT type)
   */
  public INDArray matmul(INDArray a, INDArray b) {
    NDValidation.validateNumerical("Matmul", "a", a);
    NDValidation.validateNumerical("Matmul", "b", b);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.reduce.Mmul(a, b))[0];
  }

  /**
   * Copy a tensor setting outside a central band in each innermost matrix.<br>
   *
   * @param input input tensor (NUMERIC type)
   * @param minLower lower diagonal count
   * @param maxUpper upper diagonal count
   */
  public INDArray[] matrixBandPart(INDArray input, int minLower, int maxUpper) {
    NDValidation.validateNumerical("MatrixBandPart", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.custom.MatrixBandPart(input, minLower, maxUpper));
  }

  /**
   * Computes the QR decompositions of input matrix.<br>
   *
   * @param input input tensor (NUMERIC type)
   * @param full full matrices mode
   */
  public INDArray[] qr(INDArray input, boolean full) {
    NDValidation.validateNumerical("Qr", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.Qr(input, full));
  }

  /**
   * Computes the QR decompositions of input matrix.<br>
   *
   * @param input input tensor (NUMERIC type)
   */
  public INDArray[] qr(INDArray input) {
    NDValidation.validateNumerical("Qr", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.Qr(input, false));
  }

  /**
   * Solver for systems of linear equations.<br>
   *
   * @param matrix input tensor (NUMERIC type)
   * @param rhs input tensor (NUMERIC type)
   * @param adjoint adjoint mode, defaults to False
   * @return output Output tensor (FLOATING_POINT type)
   */
  public INDArray solve(INDArray matrix, INDArray rhs, boolean adjoint) {
    NDValidation.validateNumerical("Solve", "matrix", matrix);
    NDValidation.validateNumerical("Solve", "rhs", rhs);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.custom.LinearSolve(matrix, rhs, adjoint))[0];
  }

  /**
   * Solver for systems of linear equations.<br>
   *
   * @param matrix input tensor (NUMERIC type)
   * @param rhs input tensor (NUMERIC type)
   * @return output Output tensor (FLOATING_POINT type)
   */
  public INDArray solve(INDArray matrix, INDArray rhs) {
    NDValidation.validateNumerical("Solve", "matrix", matrix);
    NDValidation.validateNumerical("Solve", "rhs", rhs);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.custom.LinearSolve(matrix, rhs, false))[0];
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
  public INDArray triangularSolve(INDArray matrix, INDArray rhs, boolean lower, boolean adjoint) {
    NDValidation.validateNumerical("TriangularSolve", "matrix", matrix);
    NDValidation.validateNumerical("TriangularSolve", "rhs", rhs);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.custom.TriangularSolve(matrix, rhs, lower, adjoint))[0];
  }

  /**
   * Computes pairwise cross product.<br>
   *
   * @param a  (NUMERIC type)
   * @param b  (NUMERIC type)
   * @return output  (FLOATING_POINT type)
   */
  public INDArray cross(INDArray a, INDArray b) {
    NDValidation.validateNumerical("cross", "a", a);
    NDValidation.validateNumerical("cross", "b", b);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.Cross(a, b))[0];
  }

  /**
   * Calculates diagonal tensor.<br>
   *
   * @param input  (NUMERIC type)
   * @return output  (FLOATING_POINT type)
   */
  public INDArray diag(INDArray input) {
    NDValidation.validateNumerical("diag", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.Diag(input))[0];
  }

  /**
   * Calculates diagonal tensor.<br>
   *
   * @param input  (NUMERIC type)
   * @return output  (FLOATING_POINT type)
   */
  public INDArray diag_part(INDArray input) {
    NDValidation.validateNumerical("diag_part", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.DiagPart(input))[0];
  }

  /**
   * Calculates log of determinant.<br>
   *
   * @param input  (NUMERIC type)
   * @return output  (FLOATING_POINT type)
   */
  public INDArray logdet(INDArray input) {
    NDValidation.validateNumerical("logdet", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.custom.Logdet(input))[0];
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
   * Calculates singular value decomposition.<br>
   *
   * @param input  (NUMERIC type)
   * @param fullUV 
   * @param computeUV 
   * @param switchNum 
   * @return output  (FLOATING_POINT type)
   */
  public INDArray svd(INDArray input, boolean fullUV, boolean computeUV, int switchNum) {
    NDValidation.validateNumerical("svd", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.Svd(input, fullUV, computeUV, switchNum))[0];
  }

  /**
   * Calculates singular value decomposition.<br>
   *
   * @param input  (NUMERIC type)
   * @param fullUV 
   * @param computeUV 
   * @return output  (FLOATING_POINT type)
   */
  public INDArray svd(INDArray input, boolean fullUV, boolean computeUV) {
    NDValidation.validateNumerical("svd", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.Svd(input, fullUV, computeUV, 16))[0];
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
  public INDArray tri(DataType dataType, int row, int column, int diagonal) {
    return Nd4j.exec(new org.nd4j.linalg.api.ops.custom.Tri(dataType, row, column, diagonal))[0];
  }

  /**
   * An array with ones at and below the given diagonal and zeros elsewhere.<br>
   *
   * @param row 
   * @param column 
   * @return output  (FLOATING_POINT type)
   */
  public INDArray tri(int row, int column) {
    return Nd4j.exec(new org.nd4j.linalg.api.ops.custom.Tri(DataType.FLOAT, row, column, 0))[0];
  }

  /**
   * Upper triangle of an array. Return a copy of a input tensor with the elements below the k-th diagonal zeroed.<br>
   *
   * @param input  (NUMERIC type)
   * @param diag 
   * @return output  (FLOATING_POINT type)
   */
  public INDArray triu(INDArray input, int diag) {
    NDValidation.validateNumerical("triu", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.custom.Triu(input, diag))[0];
  }

  /**
   * Upper triangle of an array. Return a copy of a input tensor with the elements below the k-th diagonal zeroed.<br>
   *
   * @param input  (NUMERIC type)
   * @return output  (FLOATING_POINT type)
   */
  public INDArray triu(INDArray input) {
    NDValidation.validateNumerical("triu", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.custom.Triu(input, 0))[0];
  }
}

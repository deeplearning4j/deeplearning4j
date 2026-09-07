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

import java.lang.String;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.custom.Eig;
import org.nd4j.linalg.api.ops.custom.LinearSolve;
import org.nd4j.linalg.api.ops.custom.Logdet;
import org.nd4j.linalg.api.ops.custom.Lstsq;
import org.nd4j.linalg.api.ops.custom.Lu;
import org.nd4j.linalg.api.ops.custom.MatrixBandPart;
import org.nd4j.linalg.api.ops.custom.Tri;
import org.nd4j.linalg.api.ops.custom.TriangularSolve;
import org.nd4j.linalg.api.ops.custom.Triu;
import org.nd4j.linalg.api.ops.impl.reduce.Mmul;
import org.nd4j.linalg.api.ops.impl.shape.Cross;
import org.nd4j.linalg.api.ops.impl.shape.Diag;
import org.nd4j.linalg.api.ops.impl.shape.DiagPart;
import org.nd4j.linalg.api.ops.impl.transforms.Cholesky;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Einsum;
import org.nd4j.linalg.api.ops.impl.transforms.custom.MatrixDeterminant;
import org.nd4j.linalg.api.ops.impl.transforms.custom.MatrixInverse;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Qr;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Svd;
import org.nd4j.linalg.factory.NDValidation;
import org.nd4j.linalg.factory.Nd4j;

public class NDLinalg {
  public NDLinalg() {
  }

  /**
   * Computes the Cholesky decomposition of one or more square matrices.
   *
   * @param input Input tensor with inner-most 2 dimensions forming square matrices (NUMERIC type)
   * @return output Transformed tensor (NUMERIC type)
   */
  public INDArray cholesky(INDArray input) {
    NDValidation.validateNumerical("Cholesky", "input", input);
    INDArray[] __tmp = Nd4j.exec(new Cholesky(input));
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
   * Solver for linear squares problems.
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
    INDArray[] __tmp = Nd4j.exec(new Lstsq(matrix, rhs, l2_reguralizer, fast));
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
   * Solver for linear squares problems.
   *
   * @param matrix input tensor (NUMERIC type)
   * @param rhs input tensor (NUMERIC type)
   * @param l2_reguralizer regularizer
   * @return output Transformed tensor (FLOATING_POINT type)
   */
  public INDArray lstsq(INDArray matrix, INDArray rhs, double l2_reguralizer) {
    NDValidation.validateNumerical("Lstsq", "matrix", matrix);
    NDValidation.validateNumerical("Lstsq", "rhs", rhs);
    INDArray[] __tmp = Nd4j.exec(new Lstsq(matrix, rhs, l2_reguralizer, true));
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
   * Computes LU decomposition.
   *
   * @param input input tensor (NUMERIC type)
   * @return output  (FLOATING_POINT type)
   */
  public INDArray lu(INDArray input) {
    NDValidation.validateNumerical("Lu", "input", input);
    INDArray[] __tmp = Nd4j.exec(new Lu(input));
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
   * Performs matrix multiplication on input tensors.
   *
   * @param a input tensor (NUMERIC type)
   * @param b input tensor (NUMERIC type)
   * @param alpha Defaults to 1.0: the scalar multiplier for the product of a* b
   * @param beta Defaults to 0.0: the scalar multiplier for c
   * @param transA Whether to transpose a when running multiply
   * @param transB Whether to transpose b when running multiply
   * @return output  (FLOATING_POINT type)
   */
  public INDArray matmul(INDArray a, INDArray b, double alpha, double beta, boolean transA,
      boolean transB) {
    NDValidation.validateNumerical("Matmul", "a", a);
    NDValidation.validateNumerical("Matmul", "b", b);
    INDArray[] __tmp = Nd4j.exec(new Mmul(a, b, alpha, beta, transA, transB));
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
   * Performs matrix multiplication on input tensors.
   *
   * @param a input tensor (NUMERIC type)
   * @param b input tensor (NUMERIC type)
   * @return output  (FLOATING_POINT type)
   */
  public INDArray matmul(INDArray a, INDArray b) {
    NDValidation.validateNumerical("Matmul", "a", a);
    NDValidation.validateNumerical("Matmul", "b", b);
    INDArray[] __tmp = Nd4j.exec(new Mmul(a, b, 1.0, 0.0, false, false));
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
   * Copy a tensor setting outside a central band in each innermost matrix.
   *
   * @param input input tensor (NUMERIC type)
   * @param minLower lower diagonal count
   * @param maxUpper upper diagonal count
   * @return output1  (FLOATING_POINT type)
   * @return output2  (FLOATING_POINT type)
   */
  public INDArray[] matrixBandPart(INDArray input, int minLower, int maxUpper) {
    NDValidation.validateNumerical("MatrixBandPart", "input", input);
    return Nd4j.exec(new MatrixBandPart(input, minLower, maxUpper));
  }

  /**
   * Computes the QR decompositions of input matrix.
   *
   * @param input input tensor (NUMERIC type)
   * @param full full matrices mode
   * @return outputQ  (FLOATING_POINT type)
   * @return outputR  (FLOATING_POINT type)
   */
  public INDArray[] qr(INDArray input, boolean full) {
    NDValidation.validateNumerical("Qr", "input", input);
    return Nd4j.exec(new Qr(input, full));
  }

  /**
   * Computes the QR decompositions of input matrix.
   *
   * @param input input tensor (NUMERIC type)
   * @return outputQ  (FLOATING_POINT type)
   * @return outputR  (FLOATING_POINT type)
   */
  public INDArray[] qr(INDArray input) {
    NDValidation.validateNumerical("Qr", "input", input);
    return Nd4j.exec(new Qr(input, false));
  }

  /**
   * Solver for systems of linear equations.
   *
   * @param matrix input tensor (NUMERIC type)
   * @param rhs input tensor (NUMERIC type)
   * @param adjoint adjoint mode, defaults to False
   * @return output Output tensor (FLOATING_POINT type)
   */
  public INDArray solve(INDArray matrix, INDArray rhs, boolean adjoint) {
    NDValidation.validateNumerical("Solve", "matrix", matrix);
    NDValidation.validateNumerical("Solve", "rhs", rhs);
    INDArray[] __tmp = Nd4j.exec(new LinearSolve(matrix, rhs, adjoint));
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
   * Solver for systems of linear equations.
   *
   * @param matrix input tensor (NUMERIC type)
   * @param rhs input tensor (NUMERIC type)
   * @return output Output tensor (FLOATING_POINT type)
   */
  public INDArray solve(INDArray matrix, INDArray rhs) {
    NDValidation.validateNumerical("Solve", "matrix", matrix);
    NDValidation.validateNumerical("Solve", "rhs", rhs);
    INDArray[] __tmp = Nd4j.exec(new LinearSolve(matrix, rhs, false));
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
   * Solver for systems of linear questions.
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
    INDArray[] __tmp = Nd4j.exec(new TriangularSolve(matrix, rhs, lower, adjoint));
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
   * Computes pairwise cross product.
   *
   * @param a  (NUMERIC type)
   * @param b  (NUMERIC type)
   * @return output  (FLOATING_POINT type)
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
   * Calculates diagonal tensor.
   *
   * @param input  (NUMERIC type)
   * @return output  (FLOATING_POINT type)
   */
  public INDArray diag(INDArray input) {
    NDValidation.validateNumerical("diag", "input", input);
    INDArray[] __tmp = Nd4j.exec(new Diag(input));
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
   * Calculates diagonal tensor.
   *
   * @param input  (NUMERIC type)
   * @return output  (FLOATING_POINT type)
   */
  public INDArray diag_part(INDArray input) {
    NDValidation.validateNumerical("diag_part", "input", input);
    INDArray[] __tmp = Nd4j.exec(new DiagPart(input));
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
   * Calculates eigen values
   *
   * @param input  (NUMERIC type)
   * @return eigenValues  (FLOATING_POINT type)
   * @return eigenVectors  (FLOATING_POINT type)
   */
  public INDArray[] eig(INDArray input) {
    NDValidation.validateNumerical("eig", "input", input);
    return Nd4j.exec(new Eig(input));
  }

  /**
   * Einsum (Einstein summation) operation.
   *
   * Provides a powerful way to express tensor operations using Einstein summation notation.
   * The equation string specifies the subscripts for each input tensor and the output tensor.
   *
   * Examples:
   * - Matrix multiplication: "ij,jk-&gt;ik"
   * - Transpose: "ij-&gt;ji"
   * - Diagonal: "ii-&gt;i"
   * - Trace: "ii-&gt;"
   * - Batch matmul: "bij,bjk-&gt;bik"
   * - Dot product: "i,i-&gt;"
   * - Outer product: "i,j-&gt;ij"
   *
   * @param inputs Input tensors (NUMERIC type)
   * @param equation Einstein summation equation string
   * @return output Output tensor (NUMERIC type)
   */
  public INDArray einsum(INDArray[] inputs, String equation) {
    NDValidation.validateNumerical("einsum", "inputs", inputs);
    Preconditions.checkArgument(inputs.length >= 1, "inputs has incorrect size/length. Expected: inputs.length >= 1, got %s", inputs.length);
    INDArray[] __tmp = Nd4j.exec(new Einsum(inputs, equation));
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
   * Calculates log of determinant.
   *
   * @param input  (NUMERIC type)
   * @return output  (FLOATING_POINT type)
   */
  public INDArray logdet(INDArray input) {
    NDValidation.validateNumerical("logdet", "input", input);
    INDArray[] __tmp = Nd4j.exec(new Logdet(input));
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
   * Calculates matrix determinant.
   *
   * @param input  (NUMERIC type)
   * @return output  (FLOATING_POINT type)
   */
  public INDArray matrixDeterminant(INDArray input) {
    NDValidation.validateNumerical("matrixDeterminant", "input", input);
    INDArray[] __tmp = Nd4j.exec(new MatrixDeterminant(input));
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
   * Inverts a matrix
   *
   * @param input  (NUMERIC type)
   * @return output  (FLOATING_POINT type)
   */
  public INDArray matrixInverse(INDArray input) {
    NDValidation.validateNumerical("matrixInverse", "input", input);
    INDArray[] __tmp = Nd4j.exec(new MatrixInverse(input));
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
   * Matrix multiplication: out = mmul(x,y)
   * Supports specifying transpose argument to perform operation such as mmul(a^T, b), etc.
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
    INDArray[] __tmp = Nd4j.exec(new Mmul(x, y, transposeX, transposeY, transposeZ));
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
   * Matrix multiplication: out = mmul(x,y)
   * Supports specifying transpose argument to perform operation such as mmul(a^T, b), etc.
   *
   * @param x First input variable (NUMERIC type)
   * @param y Second input variable (NUMERIC type)
   * @return output  (NUMERIC type)
   */
  public INDArray mmul(INDArray x, INDArray y) {
    NDValidation.validateNumerical("mmul", "x", x);
    NDValidation.validateNumerical("mmul", "y", y);
    INDArray[] __tmp = Nd4j.exec(new Mmul(x, y, false, false, false));
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
   * Calculates singular value decomposition.
   *
   * @param input  (NUMERIC type)
   * @param fullUV
   * @param computeUV
   * @param switchNum
   * @return output  (FLOATING_POINT type)
   */
  public INDArray svd(INDArray input, boolean fullUV, boolean computeUV, int switchNum) {
    NDValidation.validateNumerical("svd", "input", input);
    INDArray[] __tmp = Nd4j.exec(new Svd(input, fullUV, computeUV, switchNum));
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
   * Calculates singular value decomposition.
   *
   * @param input  (NUMERIC type)
   * @param fullUV
   * @param computeUV
   * @return output  (FLOATING_POINT type)
   */
  public INDArray svd(INDArray input, boolean fullUV, boolean computeUV) {
    NDValidation.validateNumerical("svd", "input", input);
    INDArray[] __tmp = Nd4j.exec(new Svd(input, fullUV, computeUV, 16));
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
   * An array with ones at and below the given diagonal and zeros elsewhere.
   *
   * @param dataType Data type
   * @param row
   * @param column
   * @param diagonal
   * @return output  (FLOATING_POINT type)
   */
  public INDArray tri(DataType dataType, int row, int column, int diagonal) {
    INDArray[] __tmp = Nd4j.exec(new Tri(dataType, row, column, diagonal));
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
   * An array with ones at and below the given diagonal and zeros elsewhere.
   *
   * @param row
   * @param column
   * @return output  (FLOATING_POINT type)
   */
  public INDArray tri(int row, int column) {
    INDArray[] __tmp = Nd4j.exec(new Tri(DataType.FLOAT, row, column, 0));
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
   * Upper triangle of an array. Return a copy of a input tensor with the elements below the k-th diagonal zeroed.
   *
   * @param input  (NUMERIC type)
   * @param diag
   * @return output  (FLOATING_POINT type)
   */
  public INDArray triu(INDArray input, int diag) {
    NDValidation.validateNumerical("triu", "input", input);
    INDArray[] __tmp = Nd4j.exec(new Triu(input, diag));
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
   * Upper triangle of an array. Return a copy of a input tensor with the elements below the k-th diagonal zeroed.
   *
   * @param input  (NUMERIC type)
   * @return output  (FLOATING_POINT type)
   */
  public INDArray triu(INDArray input) {
    NDValidation.validateNumerical("triu", "input", input);
    INDArray[] __tmp = Nd4j.exec(new Triu(input, 0));
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

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

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDValidation;
import org.nd4j.linalg.factory.Nd4j;

public class NDSparse {
  public NDSparse() {
  }

  /**
   * BSR sparse matrix-dense matrix multiplication: C = A_bsr·B.<br>
   * A is in BSR format; B and C are dense.<br>
   * Equivalent to toDense(A_bsr).mmul(B) but skips zero blocks.<br>
   *
   * @param bsrValues 1D [nnzb * blockDim * blockDim] BSR non-zero block values of A (FLOATING_POINT type)
   * @param bsrColIdx 1D [nnzb] block-column indices of A (INT32) (INT type)
   * @param bsrRowPtr 1D [mb+1] block-row pointers of A (INT32), mb = rows / blockDim (INT type)
   * @param B 2D [cols, n] dense right-hand matrix (FLOATING_POINT type)
   * @param rows Number of rows in A (and rows of C)
   * @param cols Number of columns in A (and rows of B)
   * @param blockDim Square block size used in the BSR representation
   * @return C Dense result matrix [rows, n] (FLOATING_POINT type)
   */
  public INDArray bsrSpmm(INDArray bsrValues, INDArray bsrColIdx, INDArray bsrRowPtr, INDArray B,
      int rows, int cols, int blockDim) {
    NDValidation.validateFloatingPoint("bsrSpmm", "bsrValues", bsrValues);
    NDValidation.validateInteger("bsrSpmm", "bsrColIdx", bsrColIdx);
    NDValidation.validateInteger("bsrSpmm", "bsrRowPtr", bsrRowPtr);
    NDValidation.validateFloatingPoint("bsrSpmm", "B", B);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.sparse.BsrSpmm(bsrValues, bsrColIdx, bsrRowPtr, B, rows, cols, blockDim));
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
   * Convert a BSR (Block Sparse Row) sparse matrix to a dense matrix.<br>
   *
   * @param bsrValues 1D [nnzb * blockDim * blockDim] BSR non-zero block values (FLOATING_POINT type)
   * @param bsrColIdx 1D [nnzb] block-column indices (INT32) (INT type)
   * @param bsrRowPtr 1D [mb+1] block-row pointers (INT32), mb = rows / blockDim (INT type)
   * @param rows Number of rows in the logical dense matrix
   * @param cols Number of columns in the logical dense matrix
   * @param blockDim Square block size
   * @return dense Dense matrix [rows, cols] (FLOATING_POINT type)
   */
  public INDArray bsrToDense(INDArray bsrValues, INDArray bsrColIdx, INDArray bsrRowPtr, int rows,
      int cols, int blockDim) {
    NDValidation.validateFloatingPoint("bsrToDense", "bsrValues", bsrValues);
    NDValidation.validateInteger("bsrToDense", "bsrColIdx", bsrColIdx);
    NDValidation.validateInteger("bsrToDense", "bsrRowPtr", bsrRowPtr);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.sparse.BsrToDense(bsrValues, bsrColIdx, bsrRowPtr, rows, cols, blockDim));
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
   * Convert a COO (Coordinate) sparse matrix to CSR (Compressed Sparse Row) format.<br>
   * The COO entries are sorted into row-major order by the native op.<br>
   *
   * @param indices 2D [nnz, 2] INT64 row/col index pairs for each non-zero (INT type)
   * @param values 1D [nnz] non-zero values (FLOATING_POINT type)
   * @param rows Number of rows in the logical dense shape
   * @param cols Number of columns in the logical dense shape
   */
  public INDArray[] cooToCsr(INDArray indices, INDArray values, int rows, int cols) {
    NDValidation.validateInteger("cooToCsr", "indices", indices);
    NDValidation.validateFloatingPoint("cooToCsr", "values", values);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.sparse.CooToCsr(indices, values, rows, cols));
  }

  /**
   * Convert a CSC (Compressed Sparse Column) sparse matrix to a dense matrix.<br>
   *
   * @param cscValues 1D [nnz] CSC non-zero values in column-major order (FLOATING_POINT type)
   * @param cscRowIdx 1D [nnz] row index for each non-zero (INT32) (INT type)
   * @param cscColPtr 1D [cols+1] column pointers (INT32) (INT type)
   * @param rows Number of rows in the dense output
   * @param cols Number of columns in the dense output
   * @return dense Dense matrix [rows, cols] (FLOATING_POINT type)
   */
  public INDArray cscToDense(INDArray cscValues, INDArray cscRowIdx, INDArray cscColPtr, int rows,
      int cols) {
    NDValidation.validateFloatingPoint("cscToDense", "cscValues", cscValues);
    NDValidation.validateInteger("cscToDense", "cscRowIdx", cscRowIdx);
    NDValidation.validateInteger("cscToDense", "cscColPtr", cscColPtr);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.sparse.CscToDense(cscValues, cscRowIdx, cscColPtr, rows, cols));
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
   * Elementwise CSR sparse matrix addition: C = A + B.<br>
   * Both A and B must have the same logical shape [m, n].<br>
   * This op is forward-only; automatic differentiation is not supported.<br>
   *
   * @param aValues 1D [nnzA] non-zero values of A (FLOATING_POINT type)
   * @param aColIdx 1D [nnzA] column indices of A (INT32) (INT type)
   * @param aRowPtr 1D [m+1] row pointers of A (INT32) (INT type)
   * @param bValues 1D [nnzB] non-zero values of B (same dtype as aValues) (FLOATING_POINT type)
   * @param bColIdx 1D [nnzB] column indices of B (INT32) (INT type)
   * @param bRowPtr 1D [m+1] row pointers of B (INT32) (INT type)
   * @param m Number of rows (same for A and B)
   * @param n Number of columns (same for A and B)
   */
  public INDArray[] csrAdd(INDArray aValues, INDArray aColIdx, INDArray aRowPtr, INDArray bValues,
      INDArray bColIdx, INDArray bRowPtr, int m, int n) {
    NDValidation.validateFloatingPoint("csrAdd", "aValues", aValues);
    NDValidation.validateInteger("csrAdd", "aColIdx", aColIdx);
    NDValidation.validateInteger("csrAdd", "aRowPtr", aRowPtr);
    NDValidation.validateFloatingPoint("csrAdd", "bValues", bValues);
    NDValidation.validateInteger("csrAdd", "bColIdx", bColIdx);
    NDValidation.validateInteger("csrAdd", "bRowPtr", bRowPtr);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.sparse.CsrAdd(aValues, aColIdx, aRowPtr, bValues, bColIdx, bRowPtr, m, n));
  }

  /**
   * Diagonal-scaled sparse matrix product: out[e] = dl[i]*aValues[e]*dr[j] for each non-zero (i,j).<br>
   * Computes the non-zero values of Dl·A·Dr where Dl=diag(dl), Dr=diag(dr), keeping the sparsity pattern intact.<br>
   *
   * @param aValues 1D [nnz] CSR non-zero values of A (FLOATING_POINT type)
   * @param aColIdx 1D [nnz] column indices of A (INT32) (INT type)
   * @param aRowPtr 1D [rows+1] row pointers of A (INT32) (INT type)
   * @param dl 1D [rows] left diagonal scaling vector (FLOATING_POINT type)
   * @param dr 1D [cols] right diagonal scaling vector (FLOATING_POINT type)
   * @param rows Number of rows in the sparse matrix
   * @param cols Number of columns in the sparse matrix
   * @return outValues 1D [nnz] scaled non-zero values; sparsity structure unchanged (FLOATING_POINT type)
   */
  public INDArray csrDiagMm(INDArray aValues, INDArray aColIdx, INDArray aRowPtr, INDArray dl,
      INDArray dr, int rows, int cols) {
    NDValidation.validateFloatingPoint("csrDiagMm", "aValues", aValues);
    NDValidation.validateInteger("csrDiagMm", "aColIdx", aColIdx);
    NDValidation.validateInteger("csrDiagMm", "aRowPtr", aRowPtr);
    NDValidation.validateFloatingPoint("csrDiagMm", "dl", dl);
    NDValidation.validateFloatingPoint("csrDiagMm", "dr", dr);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.sparse.CsrDiagMm(aValues, aColIdx, aRowPtr, dl, dr, rows, cols));
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
   * Segment scatter-reduce: aggregate per-edge messages to per-node outputs (the N-step of MPNN).<br>
   * mode: 0=SUM, 1=MEAN, 2=MAX.<br>
   *
   * @param rowPtr 1D [rows+1] INT32 CSR row pointers (INT type)
   * @param edgeMsg 2D [nnz, F] per-edge message vectors (FLOATING_POINT type)
   * @param rows Number of target nodes (output rows)
   * @param mode Aggregation mode: 0=SUM, 1=MEAN, 2=MAX
   * @return out 2D [rows, F] aggregated per-node output (FLOATING_POINT type)
   */
  public INDArray csrEdgeAggregate(INDArray rowPtr, INDArray edgeMsg, int rows, int mode) {
    NDValidation.validateInteger("csrEdgeAggregate", "rowPtr", rowPtr);
    NDValidation.validateFloatingPoint("csrEdgeAggregate", "edgeMsg", edgeMsg);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.sparse.CsrEdgeAggregate(rowPtr, edgeMsg, rows, mode));
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
   * Edge-gather primitive: pull node-feature vectors onto edges.<br>
   * For each edge e and feature f: edgeFeat[e,f] = X[colIdx[e],f].<br>
   * n = X.shape[0] (number of nodes); required by the backward op to reconstruct dX shape [n, F].<br>
   *
   * @param colIdx 1D [nnz] INT32 source-node ids for each edge (INT type)
   * @param X 2D [n, F] dense node-feature matrix (FLOATING_POINT type)
   * @param n Number of nodes (= X.shape[0]); stored so the backward pass can reconstruct the dX shape [n, F]
   * @return edgeFeat 2D [nnz, F] edge feature vectors pulled from X (FLOATING_POINT type)
   */
  public INDArray csrEdgeGather(INDArray colIdx, INDArray X, int n) {
    NDValidation.validateInteger("csrEdgeGather", "colIdx", colIdx);
    NDValidation.validateFloatingPoint("csrEdgeGather", "X", X);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.sparse.CsrEdgeGather(colIdx, X, n));
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
   * Per-row softmax over CSR non-zero values: the GAT edge-softmax primitive.<br>
   * For each row i: alpha[k] = exp(values[k]) / sum_{k' in row i} exp(values[k']).<br>
   *
   * @param values 1D [nnz] non-zero attention logits (FLOATING_POINT type)
   * @param rowPtr 1D [rows+1] INT32 CSR row pointers (INT type)
   * @param rows Number of rows (source nodes)
   * @return alpha 1D [nnz] per-row softmax-normalised attention weights (FLOATING_POINT type)
   */
  public INDArray csrRowSoftmax(INDArray values, INDArray rowPtr, int rows) {
    NDValidation.validateFloatingPoint("csrRowSoftmax", "values", values);
    NDValidation.validateInteger("csrRowSoftmax", "rowPtr", rowPtr);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.sparse.CsrRowSoftmax(values, rowPtr, rows));
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
   * Sparse-sparse SDDMM: sample L·Mᵀ at positions given by the target CSR sparsity pattern.<br>
   * Used as the SpGEMM gradient kernel. Forward-only (no autodiff).<br>
   *
   * @param targetRowPtr 1D [P+1] INT32 row pointers of the target sparsity pattern (INT type)
   * @param targetColIdx 1D [tnnz] INT32 column indices of the target pattern (INT type)
   * @param Lvalues 1D [Lnnz] non-zero values of L [P, R] in CSR format (FLOATING_POINT type)
   * @param LcolIdx 1D [Lnnz] column indices of L (INT32) (INT type)
   * @param LrowPtr 1D [P+1] row pointers of L (INT32) (INT type)
   * @param Mvalues 1D [Mnnz] non-zero values of M [Q, R] in CSR format (FLOATING_POINT type)
   * @param McolIdx 1D [Mnnz] column indices of M (INT32) (INT type)
   * @param MrowPtr 1D [Q+1] row pointers of M (INT32) (INT type)
   * @param P Number of rows in L (= rows of target pattern)
   * @param Q Number of rows in M (= cols of target pattern)
   * @param R Shared inner dimension: cols of L = cols of M
   * @return outValues 1D [tnnz] sampled values of L·Mᵀ at the target pattern positions (FLOATING_POINT type)
   */
  public INDArray csrSddmmSparse(INDArray targetRowPtr, INDArray targetColIdx, INDArray Lvalues,
      INDArray LcolIdx, INDArray LrowPtr, INDArray Mvalues, INDArray McolIdx, INDArray MrowPtr,
      int P, int Q, int R) {
    NDValidation.validateInteger("csrSddmmSparse", "targetRowPtr", targetRowPtr);
    NDValidation.validateInteger("csrSddmmSparse", "targetColIdx", targetColIdx);
    NDValidation.validateFloatingPoint("csrSddmmSparse", "Lvalues", Lvalues);
    NDValidation.validateInteger("csrSddmmSparse", "LcolIdx", LcolIdx);
    NDValidation.validateInteger("csrSddmmSparse", "LrowPtr", LrowPtr);
    NDValidation.validateFloatingPoint("csrSddmmSparse", "Mvalues", Mvalues);
    NDValidation.validateInteger("csrSddmmSparse", "McolIdx", McolIdx);
    NDValidation.validateInteger("csrSddmmSparse", "MrowPtr", MrowPtr);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.sparse.CsrSddmmSparse(targetRowPtr, targetColIdx, Lvalues, LcolIdx, LrowPtr, Mvalues, McolIdx, MrowPtr, P, Q, R));
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
   * Neighbourhood max-aggregation over a CSR graph: the GraphSAGE-max primitive.<br>
   * For each row i and feature f: out[i,f] = max over source neighbours j of X[j,f].<br>
   *
   * @param colIdx 1D [nnz] INT32 column (source-node) indices (INT type)
   * @param rowPtr 1D [rows+1] INT32 row (segment) pointers (INT type)
   * @param X 2D [n, f] dense node feature matrix (FLOATING_POINT type)
   * @param rows Number of target-node rows (segments)
   * @return out 2D [rows, f] per-segment element-wise maximum of node features (FLOATING_POINT type)
   */
  public INDArray csrSegmentMax(INDArray colIdx, INDArray rowPtr, INDArray X, int rows) {
    NDValidation.validateInteger("csrSegmentMax", "colIdx", colIdx);
    NDValidation.validateInteger("csrSegmentMax", "rowPtr", rowPtr);
    NDValidation.validateFloatingPoint("csrSegmentMax", "X", X);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.sparse.CsrSegmentMax(colIdx, rowPtr, X, rows));
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
   * CSR sparse matrix-matrix multiplication (SpGEMM): C = A·B.<br>
   * Both A and B are in CSR format; output C is also in CSR format.<br>
   * The output nnz is data-dependent and determined by the native shape function.<br>
   *
   * @param aValues 1D [nnzA] non-zero values of A (FLOATING_POINT type)
   * @param aColIdx 1D [nnzA] column indices of A (INT32) (INT type)
   * @param aRowPtr 1D [m+1] row pointers of A (INT32) (INT type)
   * @param bValues 1D [nnzB] non-zero values of B (same dtype as aValues) (FLOATING_POINT type)
   * @param bColIdx 1D [nnzB] column indices of B (INT32) (INT type)
   * @param bRowPtr 1D [k+1] row pointers of B (INT32) (INT type)
   * @param m Number of rows of A (= rows of C)
   * @param k Number of columns of A (= rows of B)
   * @param n Number of columns of B (= columns of C)
   */
  public INDArray[] csrSpgemm(INDArray aValues, INDArray aColIdx, INDArray aRowPtr,
      INDArray bValues, INDArray bColIdx, INDArray bRowPtr, int m, int k, int n) {
    NDValidation.validateFloatingPoint("csrSpgemm", "aValues", aValues);
    NDValidation.validateInteger("csrSpgemm", "aColIdx", aColIdx);
    NDValidation.validateInteger("csrSpgemm", "aRowPtr", aRowPtr);
    NDValidation.validateFloatingPoint("csrSpgemm", "bValues", bValues);
    NDValidation.validateInteger("csrSpgemm", "bColIdx", bColIdx);
    NDValidation.validateInteger("csrSpgemm", "bRowPtr", bRowPtr);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.sparse.CsrSpgemm(aValues, aColIdx, aRowPtr, bValues, bColIdx, bRowPtr, m, k, n));
  }

  /**
   * CSR sparse matrix-matrix product: C = A·B (or Aᵀ·B when transposeA=true).<br>
   *
   * @param values 1D [nnz] CSR non-zero values (FLOATING_POINT type)
   * @param colIdx 1D [nnz] CSR column indices (INT32) (INT type)
   * @param rowPtr 1D [rows+1] CSR row pointers (INT32) (INT type)
   * @param B 2D dense matrix [cols, n] (or [rows, n] when transposeA=true) (FLOATING_POINT type)
   * @param rows Number of rows in the sparse matrix A
   * @param cols Number of columns in the sparse matrix A
   * @param transposeA If true compute Aᵀ·B instead of A·B
   * @return C Dense result matrix [rows, n] (or [cols, n] when transposeA=true) (FLOATING_POINT type)
   */
  public INDArray csrSpmm(INDArray values, INDArray colIdx, INDArray rowPtr, INDArray B, int rows,
      int cols, boolean transposeA) {
    NDValidation.validateFloatingPoint("csrSpmm", "values", values);
    NDValidation.validateInteger("csrSpmm", "colIdx", colIdx);
    NDValidation.validateInteger("csrSpmm", "rowPtr", rowPtr);
    NDValidation.validateFloatingPoint("csrSpmm", "B", B);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.sparse.CsrSpmm(values, colIdx, rowPtr, B, rows, cols, transposeA));
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
   * CSR sparse matrix-vector product: y = A·x (or Aᵀ·x when transposeA=true).<br>
   *
   * @param values 1D [nnz] CSR non-zero values (FLOATING_POINT type)
   * @param colIdx 1D [nnz] CSR column indices (INT32) (INT type)
   * @param rowPtr 1D [rows+1] CSR row pointers (INT32) (INT type)
   * @param x 1D dense vector: length cols (or rows when transposeA=true) (FLOATING_POINT type)
   * @param rows Number of rows in the sparse matrix
   * @param cols Number of columns in the sparse matrix
   * @param transposeA If true compute Aᵀ·x instead of A·x
   * @return y Result vector of length rows (or cols when transposeA=true) (FLOATING_POINT type)
   */
  public INDArray csrSpmv(INDArray values, INDArray colIdx, INDArray rowPtr, INDArray x, int rows,
      int cols, boolean transposeA) {
    NDValidation.validateFloatingPoint("csrSpmv", "values", values);
    NDValidation.validateInteger("csrSpmv", "colIdx", colIdx);
    NDValidation.validateInteger("csrSpmv", "rowPtr", rowPtr);
    NDValidation.validateFloatingPoint("csrSpmv", "x", x);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.sparse.CsrSpmv(values, colIdx, rowPtr, x, rows, cols, transposeA));
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
   * Convert a CSR sparse matrix to BSR (Block Sparse Row) format.<br>
   * Both rows and cols must be exact multiples of blockDim.<br>
   *
   * @param csrValues 1D [nnz] CSR non-zero values (FLOATING_POINT type)
   * @param csrColIdx 1D [nnz] CSR column indices (INT32) (INT type)
   * @param csrRowPtr 1D [rows+1] CSR row pointers (INT32) (INT type)
   * @param rows Number of rows (must be a multiple of blockDim)
   * @param cols Number of columns (must be a multiple of blockDim)
   * @param blockDim Square block size; rows and cols must be exact multiples
   */
  public INDArray[] csrToBsr(INDArray csrValues, INDArray csrColIdx, INDArray csrRowPtr, int rows,
      int cols, int blockDim) {
    NDValidation.validateFloatingPoint("csrToBsr", "csrValues", csrValues);
    NDValidation.validateInteger("csrToBsr", "csrColIdx", csrColIdx);
    NDValidation.validateInteger("csrToBsr", "csrRowPtr", csrRowPtr);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.sparse.CsrToBsr(csrValues, csrColIdx, csrRowPtr, rows, cols, blockDim));
  }

  /**
   * Convert a CSR sparse matrix to CSC (Compressed Sparse Column) format.<br>
   * The CSC of A is algebraically identical to the CSR of Aᵀ, so the output also provides a free sparse transpose.<br>
   *
   * @param values 1D [nnz] CSR non-zero values (FLOATING_POINT type)
   * @param colIdx 1D [nnz] CSR column indices (INT32) (INT type)
   * @param rowPtr 1D [rows+1] CSR row pointers (INT32) (INT type)
   * @param rows Number of rows in the logical matrix
   * @param cols Number of columns in the logical matrix
   */
  public INDArray[] csrToCsc(INDArray values, INDArray colIdx, INDArray rowPtr, int rows,
      int cols) {
    NDValidation.validateFloatingPoint("csrToCsc", "values", values);
    NDValidation.validateInteger("csrToCsc", "colIdx", colIdx);
    NDValidation.validateInteger("csrToCsc", "rowPtr", rowPtr);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.sparse.CsrToCsc(values, colIdx, rowPtr, rows, cols));
  }

  /**
   * Convert a CSR (Compressed Sparse Row) sparse matrix to a dense matrix.<br>
   * Inputs are the three CSR component arrays (values, colIdx, rowPtr) plus integer shape arguments rows and cols.<br>
   *
   * @param values 1D [nnz] non-zero values of the CSR matrix (FLOATING_POINT type)
   * @param colIdx 1D [nnz] column indices (INT32/INT64) (INT type)
   * @param rowPtr 1D [rows+1] row pointers (INT32/INT64) (INT type)
   * @param rows Number of rows in the dense output
   * @param cols Number of columns in the dense output
   * @return dense Dense matrix [rows, cols] (FLOATING_POINT type)
   */
  public INDArray csrToDense(INDArray values, INDArray colIdx, INDArray rowPtr, int rows,
      int cols) {
    NDValidation.validateFloatingPoint("csrToDense", "values", values);
    NDValidation.validateInteger("csrToDense", "colIdx", colIdx);
    NDValidation.validateInteger("csrToDense", "rowPtr", rowPtr);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.sparse.CsrToDense(values, colIdx, rowPtr, rows, cols));
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
   * Convert a dense matrix to COO (Coordinate) sparse representation.<br>
   * Returns indices [nnz, 2] (INT64) and values [nnz] in corresponding order.<br>
   *
   * @param dense 2D dense input matrix [rows, cols] (FLOATING_POINT type)
   * @param threshold Keep entries where |x| > threshold (0.0 keeps all non-zeros)
   */
  public INDArray[] denseToCoo(INDArray dense, double threshold) {
    NDValidation.validateFloatingPoint("denseToCoo", "dense", dense);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.sparse.DenseToCoo(dense, threshold));
  }

  /**
   * Convert a dense matrix to CSC (Compressed Sparse Column) sparse representation.<br>
   * Only entries with |x| > threshold are kept.<br>
   *
   * @param dense 2D dense input matrix [rows, cols] (FLOATING_POINT type)
   * @param threshold Keep entries where |x| > threshold (0.0 keeps all non-zeros)
   */
  public INDArray[] denseToCsc(INDArray dense, double threshold) {
    NDValidation.validateFloatingPoint("denseToCsc", "dense", dense);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.sparse.DenseToCsc(dense, threshold));
  }

  /**
   * Convert a dense matrix to CSR (Compressed Sparse Row) sparse representation.<br>
   * Only entries with |x| > threshold are kept; pass threshold=0.0 to retain all structurally non-zero entries.<br>
   *
   * @param dense 2D dense input matrix [rows, cols] (FLOATING_POINT type)
   * @param threshold Keep entries where |x| > threshold (0.0 keeps all non-zeros)
   */
  public INDArray[] denseToCsr(INDArray dense, double threshold) {
    NDValidation.validateFloatingPoint("denseToCsr", "dense", dense);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.sparse.DenseToCsr(dense, threshold));
  }

  /**
   * Sampled Dense-Dense Matrix Multiplication (SDDMM).<br>
   * For each non-zero position (i,j) in the sparsity pattern computes sum_l D1[i,l]*D2[j,l].<br>
   *
   * @param rowPtr 1D [rows+1] INT32 row pointers of the sparsity pattern (INT type)
   * @param colIdx 1D [nnz] INT32 column indices of the sparsity pattern (INT type)
   * @param D1 2D [rows, p] left dense factor (FLOATING_POINT type)
   * @param D2 2D [cols, p] right dense factor (FLOATING_POINT type)
   * @param rows Number of rows in the sparsity pattern
   * @param cols Number of columns in the sparsity pattern
   * @return values 1D [nnz] sampled values of D1·D2ᵀ at the pattern positions (FLOATING_POINT type)
   */
  public INDArray sddmm(INDArray rowPtr, INDArray colIdx, INDArray D1, INDArray D2, int rows,
      int cols) {
    NDValidation.validateInteger("sddmm", "rowPtr", rowPtr);
    NDValidation.validateInteger("sddmm", "colIdx", colIdx);
    NDValidation.validateFloatingPoint("sddmm", "D1", D1);
    NDValidation.validateFloatingPoint("sddmm", "D2", D2);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.sparse.Sddmm(rowPtr, colIdx, D1, D2, rows, cols));
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
   * Build an n×n diagonal CSR sparse matrix from a 1D diagonal vector.<br>
   * The result has exactly n non-zeros, one per diagonal entry.<br>
   *
   * @param diag 1D [n] diagonal values (FLOATING_POINT type)
   * @param n Size of the resulting n×n square matrix
   */
  public INDArray[] spdiags(INDArray diag, int n) {
    NDValidation.validateFloatingPoint("spdiags", "diag", diag);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.sparse.Spdiags(diag, n));
  }
}

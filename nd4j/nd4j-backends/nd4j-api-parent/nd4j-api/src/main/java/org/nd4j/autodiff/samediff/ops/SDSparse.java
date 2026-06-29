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

package org.nd4j.autodiff.samediff.ops;

import static org.nd4j.autodiff.samediff.ops.SDValidation.isSameType;

import java.lang.String;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

public class SDSparse extends SDOps {
  public SDSparse(SameDiff sameDiff) {
    super(sameDiff);
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
  public SDVariable bsrSpmm(SDVariable bsrValues, SDVariable bsrColIdx, SDVariable bsrRowPtr,
      SDVariable B, int rows, int cols, int blockDim) {
    SDValidation.validateFloatingPoint("bsrSpmm", "bsrValues", bsrValues);
    SDValidation.validateInteger("bsrSpmm", "bsrColIdx", bsrColIdx);
    SDValidation.validateInteger("bsrSpmm", "bsrRowPtr", bsrRowPtr);
    SDValidation.validateFloatingPoint("bsrSpmm", "B", B);
    return new org.nd4j.linalg.api.ops.impl.sparse.BsrSpmm(sd,bsrValues, bsrColIdx, bsrRowPtr, B, rows, cols, blockDim).outputVariable();
  }

  /**
   * BSR sparse matrix-dense matrix multiplication: C = A_bsr·B.<br>
   * A is in BSR format; B and C are dense.<br>
   * Equivalent to toDense(A_bsr).mmul(B) but skips zero blocks.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param bsrValues 1D [nnzb * blockDim * blockDim] BSR non-zero block values of A (FLOATING_POINT type)
   * @param bsrColIdx 1D [nnzb] block-column indices of A (INT32) (INT type)
   * @param bsrRowPtr 1D [mb+1] block-row pointers of A (INT32), mb = rows / blockDim (INT type)
   * @param B 2D [cols, n] dense right-hand matrix (FLOATING_POINT type)
   * @param rows Number of rows in A (and rows of C)
   * @param cols Number of columns in A (and rows of B)
   * @param blockDim Square block size used in the BSR representation
   * @return C Dense result matrix [rows, n] (FLOATING_POINT type)
   */
  public SDVariable bsrSpmm(String name, SDVariable bsrValues, SDVariable bsrColIdx,
      SDVariable bsrRowPtr, SDVariable B, int rows, int cols, int blockDim) {
    SDValidation.validateFloatingPoint("bsrSpmm", "bsrValues", bsrValues);
    SDValidation.validateInteger("bsrSpmm", "bsrColIdx", bsrColIdx);
    SDValidation.validateInteger("bsrSpmm", "bsrRowPtr", bsrRowPtr);
    SDValidation.validateFloatingPoint("bsrSpmm", "B", B);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.sparse.BsrSpmm(sd,bsrValues, bsrColIdx, bsrRowPtr, B, rows, cols, blockDim).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable bsrToDense(SDVariable bsrValues, SDVariable bsrColIdx, SDVariable bsrRowPtr,
      int rows, int cols, int blockDim) {
    SDValidation.validateFloatingPoint("bsrToDense", "bsrValues", bsrValues);
    SDValidation.validateInteger("bsrToDense", "bsrColIdx", bsrColIdx);
    SDValidation.validateInteger("bsrToDense", "bsrRowPtr", bsrRowPtr);
    return new org.nd4j.linalg.api.ops.impl.sparse.BsrToDense(sd,bsrValues, bsrColIdx, bsrRowPtr, rows, cols, blockDim).outputVariable();
  }

  /**
   * Convert a BSR (Block Sparse Row) sparse matrix to a dense matrix.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param bsrValues 1D [nnzb * blockDim * blockDim] BSR non-zero block values (FLOATING_POINT type)
   * @param bsrColIdx 1D [nnzb] block-column indices (INT32) (INT type)
   * @param bsrRowPtr 1D [mb+1] block-row pointers (INT32), mb = rows / blockDim (INT type)
   * @param rows Number of rows in the logical dense matrix
   * @param cols Number of columns in the logical dense matrix
   * @param blockDim Square block size
   * @return dense Dense matrix [rows, cols] (FLOATING_POINT type)
   */
  public SDVariable bsrToDense(String name, SDVariable bsrValues, SDVariable bsrColIdx,
      SDVariable bsrRowPtr, int rows, int cols, int blockDim) {
    SDValidation.validateFloatingPoint("bsrToDense", "bsrValues", bsrValues);
    SDValidation.validateInteger("bsrToDense", "bsrColIdx", bsrColIdx);
    SDValidation.validateInteger("bsrToDense", "bsrRowPtr", bsrRowPtr);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.sparse.BsrToDense(sd,bsrValues, bsrColIdx, bsrRowPtr, rows, cols, blockDim).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable[] cooToCsr(SDVariable indices, SDVariable values, int rows, int cols) {
    SDValidation.validateInteger("cooToCsr", "indices", indices);
    SDValidation.validateFloatingPoint("cooToCsr", "values", values);
    return new org.nd4j.linalg.api.ops.impl.sparse.CooToCsr(sd,indices, values, rows, cols).outputVariables();
  }

  /**
   * Convert a COO (Coordinate) sparse matrix to CSR (Compressed Sparse Row) format.<br>
   * The COO entries are sorted into row-major order by the native op.<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param indices 2D [nnz, 2] INT64 row/col index pairs for each non-zero (INT type)
   * @param values 1D [nnz] non-zero values (FLOATING_POINT type)
   * @param rows Number of rows in the logical dense shape
   * @param cols Number of columns in the logical dense shape
   */
  public SDVariable[] cooToCsr(String[] names, SDVariable indices, SDVariable values, int rows,
      int cols) {
    SDValidation.validateInteger("cooToCsr", "indices", indices);
    SDValidation.validateFloatingPoint("cooToCsr", "values", values);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.sparse.CooToCsr(sd,indices, values, rows, cols).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
  public SDVariable cscToDense(SDVariable cscValues, SDVariable cscRowIdx, SDVariable cscColPtr,
      int rows, int cols) {
    SDValidation.validateFloatingPoint("cscToDense", "cscValues", cscValues);
    SDValidation.validateInteger("cscToDense", "cscRowIdx", cscRowIdx);
    SDValidation.validateInteger("cscToDense", "cscColPtr", cscColPtr);
    return new org.nd4j.linalg.api.ops.impl.sparse.CscToDense(sd,cscValues, cscRowIdx, cscColPtr, rows, cols).outputVariable();
  }

  /**
   * Convert a CSC (Compressed Sparse Column) sparse matrix to a dense matrix.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param cscValues 1D [nnz] CSC non-zero values in column-major order (FLOATING_POINT type)
   * @param cscRowIdx 1D [nnz] row index for each non-zero (INT32) (INT type)
   * @param cscColPtr 1D [cols+1] column pointers (INT32) (INT type)
   * @param rows Number of rows in the dense output
   * @param cols Number of columns in the dense output
   * @return dense Dense matrix [rows, cols] (FLOATING_POINT type)
   */
  public SDVariable cscToDense(String name, SDVariable cscValues, SDVariable cscRowIdx,
      SDVariable cscColPtr, int rows, int cols) {
    SDValidation.validateFloatingPoint("cscToDense", "cscValues", cscValues);
    SDValidation.validateInteger("cscToDense", "cscRowIdx", cscRowIdx);
    SDValidation.validateInteger("cscToDense", "cscColPtr", cscColPtr);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.sparse.CscToDense(sd,cscValues, cscRowIdx, cscColPtr, rows, cols).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable[] csrAdd(SDVariable aValues, SDVariable aColIdx, SDVariable aRowPtr,
      SDVariable bValues, SDVariable bColIdx, SDVariable bRowPtr, int m, int n) {
    SDValidation.validateFloatingPoint("csrAdd", "aValues", aValues);
    SDValidation.validateInteger("csrAdd", "aColIdx", aColIdx);
    SDValidation.validateInteger("csrAdd", "aRowPtr", aRowPtr);
    SDValidation.validateFloatingPoint("csrAdd", "bValues", bValues);
    SDValidation.validateInteger("csrAdd", "bColIdx", bColIdx);
    SDValidation.validateInteger("csrAdd", "bRowPtr", bRowPtr);
    return new org.nd4j.linalg.api.ops.impl.sparse.CsrAdd(sd,aValues, aColIdx, aRowPtr, bValues, bColIdx, bRowPtr, m, n).outputVariables();
  }

  /**
   * Elementwise CSR sparse matrix addition: C = A + B.<br>
   * Both A and B must have the same logical shape [m, n].<br>
   * This op is forward-only; automatic differentiation is not supported.<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param aValues 1D [nnzA] non-zero values of A (FLOATING_POINT type)
   * @param aColIdx 1D [nnzA] column indices of A (INT32) (INT type)
   * @param aRowPtr 1D [m+1] row pointers of A (INT32) (INT type)
   * @param bValues 1D [nnzB] non-zero values of B (same dtype as aValues) (FLOATING_POINT type)
   * @param bColIdx 1D [nnzB] column indices of B (INT32) (INT type)
   * @param bRowPtr 1D [m+1] row pointers of B (INT32) (INT type)
   * @param m Number of rows (same for A and B)
   * @param n Number of columns (same for A and B)
   */
  public SDVariable[] csrAdd(String[] names, SDVariable aValues, SDVariable aColIdx,
      SDVariable aRowPtr, SDVariable bValues, SDVariable bColIdx, SDVariable bRowPtr, int m,
      int n) {
    SDValidation.validateFloatingPoint("csrAdd", "aValues", aValues);
    SDValidation.validateInteger("csrAdd", "aColIdx", aColIdx);
    SDValidation.validateInteger("csrAdd", "aRowPtr", aRowPtr);
    SDValidation.validateFloatingPoint("csrAdd", "bValues", bValues);
    SDValidation.validateInteger("csrAdd", "bColIdx", bColIdx);
    SDValidation.validateInteger("csrAdd", "bRowPtr", bRowPtr);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.sparse.CsrAdd(sd,aValues, aColIdx, aRowPtr, bValues, bColIdx, bRowPtr, m, n).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
  public SDVariable csrDiagMm(SDVariable aValues, SDVariable aColIdx, SDVariable aRowPtr,
      SDVariable dl, SDVariable dr, int rows, int cols) {
    SDValidation.validateFloatingPoint("csrDiagMm", "aValues", aValues);
    SDValidation.validateInteger("csrDiagMm", "aColIdx", aColIdx);
    SDValidation.validateInteger("csrDiagMm", "aRowPtr", aRowPtr);
    SDValidation.validateFloatingPoint("csrDiagMm", "dl", dl);
    SDValidation.validateFloatingPoint("csrDiagMm", "dr", dr);
    return new org.nd4j.linalg.api.ops.impl.sparse.CsrDiagMm(sd,aValues, aColIdx, aRowPtr, dl, dr, rows, cols).outputVariable();
  }

  /**
   * Diagonal-scaled sparse matrix product: out[e] = dl[i]*aValues[e]*dr[j] for each non-zero (i,j).<br>
   * Computes the non-zero values of Dl·A·Dr where Dl=diag(dl), Dr=diag(dr), keeping the sparsity pattern intact.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param aValues 1D [nnz] CSR non-zero values of A (FLOATING_POINT type)
   * @param aColIdx 1D [nnz] column indices of A (INT32) (INT type)
   * @param aRowPtr 1D [rows+1] row pointers of A (INT32) (INT type)
   * @param dl 1D [rows] left diagonal scaling vector (FLOATING_POINT type)
   * @param dr 1D [cols] right diagonal scaling vector (FLOATING_POINT type)
   * @param rows Number of rows in the sparse matrix
   * @param cols Number of columns in the sparse matrix
   * @return outValues 1D [nnz] scaled non-zero values; sparsity structure unchanged (FLOATING_POINT type)
   */
  public SDVariable csrDiagMm(String name, SDVariable aValues, SDVariable aColIdx,
      SDVariable aRowPtr, SDVariable dl, SDVariable dr, int rows, int cols) {
    SDValidation.validateFloatingPoint("csrDiagMm", "aValues", aValues);
    SDValidation.validateInteger("csrDiagMm", "aColIdx", aColIdx);
    SDValidation.validateInteger("csrDiagMm", "aRowPtr", aRowPtr);
    SDValidation.validateFloatingPoint("csrDiagMm", "dl", dl);
    SDValidation.validateFloatingPoint("csrDiagMm", "dr", dr);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.sparse.CsrDiagMm(sd,aValues, aColIdx, aRowPtr, dl, dr, rows, cols).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable csrEdgeAggregate(SDVariable rowPtr, SDVariable edgeMsg, int rows, int mode) {
    SDValidation.validateInteger("csrEdgeAggregate", "rowPtr", rowPtr);
    SDValidation.validateFloatingPoint("csrEdgeAggregate", "edgeMsg", edgeMsg);
    return new org.nd4j.linalg.api.ops.impl.sparse.CsrEdgeAggregate(sd,rowPtr, edgeMsg, rows, mode).outputVariable();
  }

  /**
   * Segment scatter-reduce: aggregate per-edge messages to per-node outputs (the N-step of MPNN).<br>
   * mode: 0=SUM, 1=MEAN, 2=MAX.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param rowPtr 1D [rows+1] INT32 CSR row pointers (INT type)
   * @param edgeMsg 2D [nnz, F] per-edge message vectors (FLOATING_POINT type)
   * @param rows Number of target nodes (output rows)
   * @param mode Aggregation mode: 0=SUM, 1=MEAN, 2=MAX
   * @return out 2D [rows, F] aggregated per-node output (FLOATING_POINT type)
   */
  public SDVariable csrEdgeAggregate(String name, SDVariable rowPtr, SDVariable edgeMsg, int rows,
      int mode) {
    SDValidation.validateInteger("csrEdgeAggregate", "rowPtr", rowPtr);
    SDValidation.validateFloatingPoint("csrEdgeAggregate", "edgeMsg", edgeMsg);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.sparse.CsrEdgeAggregate(sd,rowPtr, edgeMsg, rows, mode).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable csrEdgeGather(SDVariable colIdx, SDVariable X, int n) {
    SDValidation.validateInteger("csrEdgeGather", "colIdx", colIdx);
    SDValidation.validateFloatingPoint("csrEdgeGather", "X", X);
    return new org.nd4j.linalg.api.ops.impl.sparse.CsrEdgeGather(sd,colIdx, X, n).outputVariable();
  }

  /**
   * Edge-gather primitive: pull node-feature vectors onto edges.<br>
   * For each edge e and feature f: edgeFeat[e,f] = X[colIdx[e],f].<br>
   * n = X.shape[0] (number of nodes); required by the backward op to reconstruct dX shape [n, F].<br>
   *
   * @param name name May be null. Name for the output variable
   * @param colIdx 1D [nnz] INT32 source-node ids for each edge (INT type)
   * @param X 2D [n, F] dense node-feature matrix (FLOATING_POINT type)
   * @param n Number of nodes (= X.shape[0]); stored so the backward pass can reconstruct the dX shape [n, F]
   * @return edgeFeat 2D [nnz, F] edge feature vectors pulled from X (FLOATING_POINT type)
   */
  public SDVariable csrEdgeGather(String name, SDVariable colIdx, SDVariable X, int n) {
    SDValidation.validateInteger("csrEdgeGather", "colIdx", colIdx);
    SDValidation.validateFloatingPoint("csrEdgeGather", "X", X);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.sparse.CsrEdgeGather(sd,colIdx, X, n).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable csrRowSoftmax(SDVariable values, SDVariable rowPtr, int rows) {
    SDValidation.validateFloatingPoint("csrRowSoftmax", "values", values);
    SDValidation.validateInteger("csrRowSoftmax", "rowPtr", rowPtr);
    return new org.nd4j.linalg.api.ops.impl.sparse.CsrRowSoftmax(sd,values, rowPtr, rows).outputVariable();
  }

  /**
   * Per-row softmax over CSR non-zero values: the GAT edge-softmax primitive.<br>
   * For each row i: alpha[k] = exp(values[k]) / sum_{k' in row i} exp(values[k']).<br>
   *
   * @param name name May be null. Name for the output variable
   * @param values 1D [nnz] non-zero attention logits (FLOATING_POINT type)
   * @param rowPtr 1D [rows+1] INT32 CSR row pointers (INT type)
   * @param rows Number of rows (source nodes)
   * @return alpha 1D [nnz] per-row softmax-normalised attention weights (FLOATING_POINT type)
   */
  public SDVariable csrRowSoftmax(String name, SDVariable values, SDVariable rowPtr, int rows) {
    SDValidation.validateFloatingPoint("csrRowSoftmax", "values", values);
    SDValidation.validateInteger("csrRowSoftmax", "rowPtr", rowPtr);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.sparse.CsrRowSoftmax(sd,values, rowPtr, rows).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable csrSddmmSparse(SDVariable targetRowPtr, SDVariable targetColIdx,
      SDVariable Lvalues, SDVariable LcolIdx, SDVariable LrowPtr, SDVariable Mvalues,
      SDVariable McolIdx, SDVariable MrowPtr, int P, int Q, int R) {
    SDValidation.validateInteger("csrSddmmSparse", "targetRowPtr", targetRowPtr);
    SDValidation.validateInteger("csrSddmmSparse", "targetColIdx", targetColIdx);
    SDValidation.validateFloatingPoint("csrSddmmSparse", "Lvalues", Lvalues);
    SDValidation.validateInteger("csrSddmmSparse", "LcolIdx", LcolIdx);
    SDValidation.validateInteger("csrSddmmSparse", "LrowPtr", LrowPtr);
    SDValidation.validateFloatingPoint("csrSddmmSparse", "Mvalues", Mvalues);
    SDValidation.validateInteger("csrSddmmSparse", "McolIdx", McolIdx);
    SDValidation.validateInteger("csrSddmmSparse", "MrowPtr", MrowPtr);
    return new org.nd4j.linalg.api.ops.impl.sparse.CsrSddmmSparse(sd,targetRowPtr, targetColIdx, Lvalues, LcolIdx, LrowPtr, Mvalues, McolIdx, MrowPtr, P, Q, R).outputVariable();
  }

  /**
   * Sparse-sparse SDDMM: sample L·Mᵀ at positions given by the target CSR sparsity pattern.<br>
   * Used as the SpGEMM gradient kernel. Forward-only (no autodiff).<br>
   *
   * @param name name May be null. Name for the output variable
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
  public SDVariable csrSddmmSparse(String name, SDVariable targetRowPtr, SDVariable targetColIdx,
      SDVariable Lvalues, SDVariable LcolIdx, SDVariable LrowPtr, SDVariable Mvalues,
      SDVariable McolIdx, SDVariable MrowPtr, int P, int Q, int R) {
    SDValidation.validateInteger("csrSddmmSparse", "targetRowPtr", targetRowPtr);
    SDValidation.validateInteger("csrSddmmSparse", "targetColIdx", targetColIdx);
    SDValidation.validateFloatingPoint("csrSddmmSparse", "Lvalues", Lvalues);
    SDValidation.validateInteger("csrSddmmSparse", "LcolIdx", LcolIdx);
    SDValidation.validateInteger("csrSddmmSparse", "LrowPtr", LrowPtr);
    SDValidation.validateFloatingPoint("csrSddmmSparse", "Mvalues", Mvalues);
    SDValidation.validateInteger("csrSddmmSparse", "McolIdx", McolIdx);
    SDValidation.validateInteger("csrSddmmSparse", "MrowPtr", MrowPtr);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.sparse.CsrSddmmSparse(sd,targetRowPtr, targetColIdx, Lvalues, LcolIdx, LrowPtr, Mvalues, McolIdx, MrowPtr, P, Q, R).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable csrSegmentMax(SDVariable colIdx, SDVariable rowPtr, SDVariable X, int rows) {
    SDValidation.validateInteger("csrSegmentMax", "colIdx", colIdx);
    SDValidation.validateInteger("csrSegmentMax", "rowPtr", rowPtr);
    SDValidation.validateFloatingPoint("csrSegmentMax", "X", X);
    return new org.nd4j.linalg.api.ops.impl.sparse.CsrSegmentMax(sd,colIdx, rowPtr, X, rows).outputVariable();
  }

  /**
   * Neighbourhood max-aggregation over a CSR graph: the GraphSAGE-max primitive.<br>
   * For each row i and feature f: out[i,f] = max over source neighbours j of X[j,f].<br>
   *
   * @param name name May be null. Name for the output variable
   * @param colIdx 1D [nnz] INT32 column (source-node) indices (INT type)
   * @param rowPtr 1D [rows+1] INT32 row (segment) pointers (INT type)
   * @param X 2D [n, f] dense node feature matrix (FLOATING_POINT type)
   * @param rows Number of target-node rows (segments)
   * @return out 2D [rows, f] per-segment element-wise maximum of node features (FLOATING_POINT type)
   */
  public SDVariable csrSegmentMax(String name, SDVariable colIdx, SDVariable rowPtr, SDVariable X,
      int rows) {
    SDValidation.validateInteger("csrSegmentMax", "colIdx", colIdx);
    SDValidation.validateInteger("csrSegmentMax", "rowPtr", rowPtr);
    SDValidation.validateFloatingPoint("csrSegmentMax", "X", X);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.sparse.CsrSegmentMax(sd,colIdx, rowPtr, X, rows).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable[] csrSpgemm(SDVariable aValues, SDVariable aColIdx, SDVariable aRowPtr,
      SDVariable bValues, SDVariable bColIdx, SDVariable bRowPtr, int m, int k, int n) {
    SDValidation.validateFloatingPoint("csrSpgemm", "aValues", aValues);
    SDValidation.validateInteger("csrSpgemm", "aColIdx", aColIdx);
    SDValidation.validateInteger("csrSpgemm", "aRowPtr", aRowPtr);
    SDValidation.validateFloatingPoint("csrSpgemm", "bValues", bValues);
    SDValidation.validateInteger("csrSpgemm", "bColIdx", bColIdx);
    SDValidation.validateInteger("csrSpgemm", "bRowPtr", bRowPtr);
    return new org.nd4j.linalg.api.ops.impl.sparse.CsrSpgemm(sd,aValues, aColIdx, aRowPtr, bValues, bColIdx, bRowPtr, m, k, n).outputVariables();
  }

  /**
   * CSR sparse matrix-matrix multiplication (SpGEMM): C = A·B.<br>
   * Both A and B are in CSR format; output C is also in CSR format.<br>
   * The output nnz is data-dependent and determined by the native shape function.<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
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
  public SDVariable[] csrSpgemm(String[] names, SDVariable aValues, SDVariable aColIdx,
      SDVariable aRowPtr, SDVariable bValues, SDVariable bColIdx, SDVariable bRowPtr, int m, int k,
      int n) {
    SDValidation.validateFloatingPoint("csrSpgemm", "aValues", aValues);
    SDValidation.validateInteger("csrSpgemm", "aColIdx", aColIdx);
    SDValidation.validateInteger("csrSpgemm", "aRowPtr", aRowPtr);
    SDValidation.validateFloatingPoint("csrSpgemm", "bValues", bValues);
    SDValidation.validateInteger("csrSpgemm", "bColIdx", bColIdx);
    SDValidation.validateInteger("csrSpgemm", "bRowPtr", bRowPtr);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.sparse.CsrSpgemm(sd,aValues, aColIdx, aRowPtr, bValues, bColIdx, bRowPtr, m, k, n).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
  public SDVariable csrSpmm(SDVariable values, SDVariable colIdx, SDVariable rowPtr, SDVariable B,
      int rows, int cols, boolean transposeA) {
    SDValidation.validateFloatingPoint("csrSpmm", "values", values);
    SDValidation.validateInteger("csrSpmm", "colIdx", colIdx);
    SDValidation.validateInteger("csrSpmm", "rowPtr", rowPtr);
    SDValidation.validateFloatingPoint("csrSpmm", "B", B);
    return new org.nd4j.linalg.api.ops.impl.sparse.CsrSpmm(sd,values, colIdx, rowPtr, B, rows, cols, transposeA).outputVariable();
  }

  /**
   * CSR sparse matrix-matrix product: C = A·B (or Aᵀ·B when transposeA=true).<br>
   *
   * @param name name May be null. Name for the output variable
   * @param values 1D [nnz] CSR non-zero values (FLOATING_POINT type)
   * @param colIdx 1D [nnz] CSR column indices (INT32) (INT type)
   * @param rowPtr 1D [rows+1] CSR row pointers (INT32) (INT type)
   * @param B 2D dense matrix [cols, n] (or [rows, n] when transposeA=true) (FLOATING_POINT type)
   * @param rows Number of rows in the sparse matrix A
   * @param cols Number of columns in the sparse matrix A
   * @param transposeA If true compute Aᵀ·B instead of A·B
   * @return C Dense result matrix [rows, n] (or [cols, n] when transposeA=true) (FLOATING_POINT type)
   */
  public SDVariable csrSpmm(String name, SDVariable values, SDVariable colIdx, SDVariable rowPtr,
      SDVariable B, int rows, int cols, boolean transposeA) {
    SDValidation.validateFloatingPoint("csrSpmm", "values", values);
    SDValidation.validateInteger("csrSpmm", "colIdx", colIdx);
    SDValidation.validateInteger("csrSpmm", "rowPtr", rowPtr);
    SDValidation.validateFloatingPoint("csrSpmm", "B", B);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.sparse.CsrSpmm(sd,values, colIdx, rowPtr, B, rows, cols, transposeA).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable csrSpmv(SDVariable values, SDVariable colIdx, SDVariable rowPtr, SDVariable x,
      int rows, int cols, boolean transposeA) {
    SDValidation.validateFloatingPoint("csrSpmv", "values", values);
    SDValidation.validateInteger("csrSpmv", "colIdx", colIdx);
    SDValidation.validateInteger("csrSpmv", "rowPtr", rowPtr);
    SDValidation.validateFloatingPoint("csrSpmv", "x", x);
    return new org.nd4j.linalg.api.ops.impl.sparse.CsrSpmv(sd,values, colIdx, rowPtr, x, rows, cols, transposeA).outputVariable();
  }

  /**
   * CSR sparse matrix-vector product: y = A·x (or Aᵀ·x when transposeA=true).<br>
   *
   * @param name name May be null. Name for the output variable
   * @param values 1D [nnz] CSR non-zero values (FLOATING_POINT type)
   * @param colIdx 1D [nnz] CSR column indices (INT32) (INT type)
   * @param rowPtr 1D [rows+1] CSR row pointers (INT32) (INT type)
   * @param x 1D dense vector: length cols (or rows when transposeA=true) (FLOATING_POINT type)
   * @param rows Number of rows in the sparse matrix
   * @param cols Number of columns in the sparse matrix
   * @param transposeA If true compute Aᵀ·x instead of A·x
   * @return y Result vector of length rows (or cols when transposeA=true) (FLOATING_POINT type)
   */
  public SDVariable csrSpmv(String name, SDVariable values, SDVariable colIdx, SDVariable rowPtr,
      SDVariable x, int rows, int cols, boolean transposeA) {
    SDValidation.validateFloatingPoint("csrSpmv", "values", values);
    SDValidation.validateInteger("csrSpmv", "colIdx", colIdx);
    SDValidation.validateInteger("csrSpmv", "rowPtr", rowPtr);
    SDValidation.validateFloatingPoint("csrSpmv", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.sparse.CsrSpmv(sd,values, colIdx, rowPtr, x, rows, cols, transposeA).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable[] csrToBsr(SDVariable csrValues, SDVariable csrColIdx, SDVariable csrRowPtr,
      int rows, int cols, int blockDim) {
    SDValidation.validateFloatingPoint("csrToBsr", "csrValues", csrValues);
    SDValidation.validateInteger("csrToBsr", "csrColIdx", csrColIdx);
    SDValidation.validateInteger("csrToBsr", "csrRowPtr", csrRowPtr);
    return new org.nd4j.linalg.api.ops.impl.sparse.CsrToBsr(sd,csrValues, csrColIdx, csrRowPtr, rows, cols, blockDim).outputVariables();
  }

  /**
   * Convert a CSR sparse matrix to BSR (Block Sparse Row) format.<br>
   * Both rows and cols must be exact multiples of blockDim.<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param csrValues 1D [nnz] CSR non-zero values (FLOATING_POINT type)
   * @param csrColIdx 1D [nnz] CSR column indices (INT32) (INT type)
   * @param csrRowPtr 1D [rows+1] CSR row pointers (INT32) (INT type)
   * @param rows Number of rows (must be a multiple of blockDim)
   * @param cols Number of columns (must be a multiple of blockDim)
   * @param blockDim Square block size; rows and cols must be exact multiples
   */
  public SDVariable[] csrToBsr(String[] names, SDVariable csrValues, SDVariable csrColIdx,
      SDVariable csrRowPtr, int rows, int cols, int blockDim) {
    SDValidation.validateFloatingPoint("csrToBsr", "csrValues", csrValues);
    SDValidation.validateInteger("csrToBsr", "csrColIdx", csrColIdx);
    SDValidation.validateInteger("csrToBsr", "csrRowPtr", csrRowPtr);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.sparse.CsrToBsr(sd,csrValues, csrColIdx, csrRowPtr, rows, cols, blockDim).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
  public SDVariable[] csrToCsc(SDVariable values, SDVariable colIdx, SDVariable rowPtr, int rows,
      int cols) {
    SDValidation.validateFloatingPoint("csrToCsc", "values", values);
    SDValidation.validateInteger("csrToCsc", "colIdx", colIdx);
    SDValidation.validateInteger("csrToCsc", "rowPtr", rowPtr);
    return new org.nd4j.linalg.api.ops.impl.sparse.CsrToCsc(sd,values, colIdx, rowPtr, rows, cols).outputVariables();
  }

  /**
   * Convert a CSR sparse matrix to CSC (Compressed Sparse Column) format.<br>
   * The CSC of A is algebraically identical to the CSR of Aᵀ, so the output also provides a free sparse transpose.<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param values 1D [nnz] CSR non-zero values (FLOATING_POINT type)
   * @param colIdx 1D [nnz] CSR column indices (INT32) (INT type)
   * @param rowPtr 1D [rows+1] CSR row pointers (INT32) (INT type)
   * @param rows Number of rows in the logical matrix
   * @param cols Number of columns in the logical matrix
   */
  public SDVariable[] csrToCsc(String[] names, SDVariable values, SDVariable colIdx,
      SDVariable rowPtr, int rows, int cols) {
    SDValidation.validateFloatingPoint("csrToCsc", "values", values);
    SDValidation.validateInteger("csrToCsc", "colIdx", colIdx);
    SDValidation.validateInteger("csrToCsc", "rowPtr", rowPtr);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.sparse.CsrToCsc(sd,values, colIdx, rowPtr, rows, cols).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
  public SDVariable csrToDense(SDVariable values, SDVariable colIdx, SDVariable rowPtr, int rows,
      int cols) {
    SDValidation.validateFloatingPoint("csrToDense", "values", values);
    SDValidation.validateInteger("csrToDense", "colIdx", colIdx);
    SDValidation.validateInteger("csrToDense", "rowPtr", rowPtr);
    return new org.nd4j.linalg.api.ops.impl.sparse.CsrToDense(sd,values, colIdx, rowPtr, rows, cols).outputVariable();
  }

  /**
   * Convert a CSR (Compressed Sparse Row) sparse matrix to a dense matrix.<br>
   * Inputs are the three CSR component arrays (values, colIdx, rowPtr) plus integer shape arguments rows and cols.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param values 1D [nnz] non-zero values of the CSR matrix (FLOATING_POINT type)
   * @param colIdx 1D [nnz] column indices (INT32/INT64) (INT type)
   * @param rowPtr 1D [rows+1] row pointers (INT32/INT64) (INT type)
   * @param rows Number of rows in the dense output
   * @param cols Number of columns in the dense output
   * @return dense Dense matrix [rows, cols] (FLOATING_POINT type)
   */
  public SDVariable csrToDense(String name, SDVariable values, SDVariable colIdx, SDVariable rowPtr,
      int rows, int cols) {
    SDValidation.validateFloatingPoint("csrToDense", "values", values);
    SDValidation.validateInteger("csrToDense", "colIdx", colIdx);
    SDValidation.validateInteger("csrToDense", "rowPtr", rowPtr);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.sparse.CsrToDense(sd,values, colIdx, rowPtr, rows, cols).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Convert a dense matrix to COO (Coordinate) sparse representation.<br>
   * Returns indices [nnz, 2] (INT64) and values [nnz] in corresponding order.<br>
   *
   * @param dense 2D dense input matrix [rows, cols] (FLOATING_POINT type)
   * @param threshold Keep entries where |x| > threshold (0.0 keeps all non-zeros)
   */
  public SDVariable[] denseToCoo(SDVariable dense, double threshold) {
    SDValidation.validateFloatingPoint("denseToCoo", "dense", dense);
    return new org.nd4j.linalg.api.ops.impl.sparse.DenseToCoo(sd,dense, threshold).outputVariables();
  }

  /**
   * Convert a dense matrix to COO (Coordinate) sparse representation.<br>
   * Returns indices [nnz, 2] (INT64) and values [nnz] in corresponding order.<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param dense 2D dense input matrix [rows, cols] (FLOATING_POINT type)
   * @param threshold Keep entries where |x| > threshold (0.0 keeps all non-zeros)
   */
  public SDVariable[] denseToCoo(String[] names, SDVariable dense, double threshold) {
    SDValidation.validateFloatingPoint("denseToCoo", "dense", dense);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.sparse.DenseToCoo(sd,dense, threshold).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
  }

  /**
   * Convert a dense matrix to CSC (Compressed Sparse Column) sparse representation.<br>
   * Only entries with |x| > threshold are kept.<br>
   *
   * @param dense 2D dense input matrix [rows, cols] (FLOATING_POINT type)
   * @param threshold Keep entries where |x| > threshold (0.0 keeps all non-zeros)
   */
  public SDVariable[] denseToCsc(SDVariable dense, double threshold) {
    SDValidation.validateFloatingPoint("denseToCsc", "dense", dense);
    return new org.nd4j.linalg.api.ops.impl.sparse.DenseToCsc(sd,dense, threshold).outputVariables();
  }

  /**
   * Convert a dense matrix to CSC (Compressed Sparse Column) sparse representation.<br>
   * Only entries with |x| > threshold are kept.<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param dense 2D dense input matrix [rows, cols] (FLOATING_POINT type)
   * @param threshold Keep entries where |x| > threshold (0.0 keeps all non-zeros)
   */
  public SDVariable[] denseToCsc(String[] names, SDVariable dense, double threshold) {
    SDValidation.validateFloatingPoint("denseToCsc", "dense", dense);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.sparse.DenseToCsc(sd,dense, threshold).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
  }

  /**
   * Convert a dense matrix to CSR (Compressed Sparse Row) sparse representation.<br>
   * Only entries with |x| > threshold are kept; pass threshold=0.0 to retain all structurally non-zero entries.<br>
   *
   * @param dense 2D dense input matrix [rows, cols] (FLOATING_POINT type)
   * @param threshold Keep entries where |x| > threshold (0.0 keeps all non-zeros)
   */
  public SDVariable[] denseToCsr(SDVariable dense, double threshold) {
    SDValidation.validateFloatingPoint("denseToCsr", "dense", dense);
    return new org.nd4j.linalg.api.ops.impl.sparse.DenseToCsr(sd,dense, threshold).outputVariables();
  }

  /**
   * Convert a dense matrix to CSR (Compressed Sparse Row) sparse representation.<br>
   * Only entries with |x| > threshold are kept; pass threshold=0.0 to retain all structurally non-zero entries.<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param dense 2D dense input matrix [rows, cols] (FLOATING_POINT type)
   * @param threshold Keep entries where |x| > threshold (0.0 keeps all non-zeros)
   */
  public SDVariable[] denseToCsr(String[] names, SDVariable dense, double threshold) {
    SDValidation.validateFloatingPoint("denseToCsr", "dense", dense);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.sparse.DenseToCsr(sd,dense, threshold).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
  public SDVariable sddmm(SDVariable rowPtr, SDVariable colIdx, SDVariable D1, SDVariable D2,
      int rows, int cols) {
    SDValidation.validateInteger("sddmm", "rowPtr", rowPtr);
    SDValidation.validateInteger("sddmm", "colIdx", colIdx);
    SDValidation.validateFloatingPoint("sddmm", "D1", D1);
    SDValidation.validateFloatingPoint("sddmm", "D2", D2);
    return new org.nd4j.linalg.api.ops.impl.sparse.Sddmm(sd,rowPtr, colIdx, D1, D2, rows, cols).outputVariable();
  }

  /**
   * Sampled Dense-Dense Matrix Multiplication (SDDMM).<br>
   * For each non-zero position (i,j) in the sparsity pattern computes sum_l D1[i,l]*D2[j,l].<br>
   *
   * @param name name May be null. Name for the output variable
   * @param rowPtr 1D [rows+1] INT32 row pointers of the sparsity pattern (INT type)
   * @param colIdx 1D [nnz] INT32 column indices of the sparsity pattern (INT type)
   * @param D1 2D [rows, p] left dense factor (FLOATING_POINT type)
   * @param D2 2D [cols, p] right dense factor (FLOATING_POINT type)
   * @param rows Number of rows in the sparsity pattern
   * @param cols Number of columns in the sparsity pattern
   * @return values 1D [nnz] sampled values of D1·D2ᵀ at the pattern positions (FLOATING_POINT type)
   */
  public SDVariable sddmm(String name, SDVariable rowPtr, SDVariable colIdx, SDVariable D1,
      SDVariable D2, int rows, int cols) {
    SDValidation.validateInteger("sddmm", "rowPtr", rowPtr);
    SDValidation.validateInteger("sddmm", "colIdx", colIdx);
    SDValidation.validateFloatingPoint("sddmm", "D1", D1);
    SDValidation.validateFloatingPoint("sddmm", "D2", D2);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.sparse.Sddmm(sd,rowPtr, colIdx, D1, D2, rows, cols).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Build an n×n diagonal CSR sparse matrix from a 1D diagonal vector.<br>
   * The result has exactly n non-zeros, one per diagonal entry.<br>
   *
   * @param diag 1D [n] diagonal values (FLOATING_POINT type)
   * @param n Size of the resulting n×n square matrix
   */
  public SDVariable[] spdiags(SDVariable diag, int n) {
    SDValidation.validateFloatingPoint("spdiags", "diag", diag);
    return new org.nd4j.linalg.api.ops.impl.sparse.Spdiags(sd,diag, n).outputVariables();
  }

  /**
   * Build an n×n diagonal CSR sparse matrix from a 1D diagonal vector.<br>
   * The result has exactly n non-zeros, one per diagonal entry.<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param diag 1D [n] diagonal values (FLOATING_POINT type)
   * @param n Size of the resulting n×n square matrix
   */
  public SDVariable[] spdiags(String[] names, SDVariable diag, int n) {
    SDValidation.validateFloatingPoint("spdiags", "diag", diag);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.sparse.Spdiags(sd,diag, n).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
  }
}

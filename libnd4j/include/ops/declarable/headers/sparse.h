/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef SAMEDIFF_SPARSE_H
#define SAMEDIFF_SPARSE_H
#include <ops/declarable/headers/common.h>

namespace sd {
namespace ops {

/**
 * Converts a CSR sparse tensor to a dense NDArray.
 *
 * Inputs:
 *   [0] values  – 1D [nnz], floating dtype
 *   [1] colIdx  – 1D [nnz], INT32 or INT64
 *   [2] rowPtr  – 1D [rows+1], same INT dtype as colIdx
 * IArgs:
 *   [0] rows
 *   [1] cols
 * Output:
 *   [0] dense [rows, cols], dtype = values dtype
 */
#if NOT_EXCLUDED(OP_csr_to_dense)
DECLARE_CUSTOM_OP(csr_to_dense, 3, 1, false, 0, 2);
#endif

/**
 * Converts a dense NDArray to CSR sparse representation.
 *
 * Input:
 *   [0] dense [rows, cols]
 * TArgs:
 *   [0] threshold (default 0.0 — keep entries where |x| > threshold)
 * Outputs:
 *   [0] values  – 1D [nnz], same dtype as input
 *   [1] colIdx  – 1D [nnz], INT32
 *   [2] rowPtr  – 1D [rows+1], INT32
 */
#if NOT_EXCLUDED(OP_dense_to_csr)
DECLARE_CUSTOM_OP(dense_to_csr, 1, 3, false, 1, 0);
#endif

/**
 * CSR sparse matrix-vector product: y = op(A) * x
 *
 * Inputs:
 *   [0] values  – 1D [nnz], floating dtype  (non-zero values of A)
 *   [1] colIdx  – 1D [nnz], INT32 or INT64  (column indices)
 *   [2] rowPtr  – 1D [rows+1], same INT     (row pointer array)
 *   [3] x       – dense 1D vector, length = (transposeA==0 ? cols : rows)
 * IArgs:
 *   [0] rows
 *   [1] cols
 *   [2] transposeA  (0 = y=A*x, 1 = y=A^T*x)
 * Output:
 *   [0] y dense 1D, length = (transposeA==0 ? rows : cols)
 */
#if NOT_EXCLUDED(OP_csr_spmv)
DECLARE_CUSTOM_OP(csr_spmv, 4, 1, false, 0, 3);
#endif

/**
 * CSR sparse matrix-matrix product: C = op(A) * B
 *
 * Inputs:
 *   [0] values  – 1D [nnz], floating dtype  (non-zero values of A)
 *   [1] colIdx  – 1D [nnz], INT32 or INT64  (column indices)
 *   [2] rowPtr  – 1D [rows+1], same INT     (row pointer array)
 *   [3] B       – dense 2D matrix, shape = (transposeA==0 ? [cols,n] : [rows,n])
 * IArgs:
 *   [0] rows
 *   [1] cols
 *   [2] transposeA  (0 = C=A*B, 1 = C=A^T*B)
 * Output:
 *   [0] C dense 2D, shape = (transposeA==0 ? [rows,n] : [cols,n])
 */
#if NOT_EXCLUDED(OP_csr_spmm)
DECLARE_CUSTOM_OP(csr_spmm, 4, 1, false, 0, 3);
#endif

/**
 * SDDMM — sampled dense-dense matrix multiplication (gradient kernel).
 * outValues[k] = dot(D1[i,:], D2[j,:]) for each CSR nonzero at (i, j=colIdx[k]).
 *
 * Inputs:
 *   [0] rowPtr    – 1D [rows+1], INT32 or INT64
 *   [1] colIdx    – 1D [nnz],   same INT
 *   [2] D1        – dense 2D [rows, p]
 *   [3] D2        – dense 2D [cols, p]
 * IArgs:
 *   [0] rows
 *   [1] cols
 * Output:
 *   [0] outValues – 1D [nnz], same float dtype as D1
 */
#if NOT_EXCLUDED(OP_sddmm)
DECLARE_CUSTOM_OP(sddmm, 4, 1, false, 0, 2);
#endif

/**
 * Converts a dense 2-D matrix to COO sparse representation.
 *
 * Input:
 *   [0] dense [rows, cols], floating dtype
 * TArgs:
 *   [0] threshold (default 0.0 — keep entries where |x| > threshold)
 * Outputs:
 *   [0] indices [nnz, 2], INT64
 *       indices[k, 0] = row index of the k-th non-zero (row-major scan order)
 *       indices[k, 1] = col index of the k-th non-zero
 *   [1] values  [nnz], same float dtype as input
 */
#if NOT_EXCLUDED(OP_dense_to_coo)
DECLARE_CUSTOM_OP(dense_to_coo, 1, 2, false, 1, 0);
#endif

/**
 * Converts COO sparse representation to CSR sparse representation.
 *
 * Entries are sorted by (row, then col) in the output CSR arrays.
 *
 * Inputs:
 *   [0] indices  [nnz, 2], INT (col 0 = row index, col 1 = col index)
 *   [1] values   [nnz], float
 * IArgs:
 *   [0] rows
 *   [1] cols
 * Outputs:
 *   [0] csrValues [nnz],    same float dtype as values
 *   [1] colIdx    [nnz],    INT32
 *   [2] rowPtr    [rows+1], INT32
 */
#if NOT_EXCLUDED(OP_coo_to_csr)
DECLARE_CUSTOM_OP(coo_to_csr, 2, 3, false, 0, 2);
#endif

/**
 * CSR sparse-sparse matrix multiplication (SpGEMM): C = A * B
 *
 * A is [m, k] in CSR format, B is [k, n] in CSR format.
 * C is produced in CSR format [m, n].
 * The output sparsity is determined symbolically (Gustavson's algorithm)
 * in DECLARE_SHAPE_FN, so output buffers are exactly sized.
 *
 * Inputs:
 *   [0] aValues  [annz]  float   — non-zero values of A
 *   [1] aColIdx  [annz]  int     — column indices of A
 *   [2] aRowPtr  [m+1]   int     — row pointers of A
 *   [3] bValues  [bnnz]  float   — non-zero values of B
 *   [4] bColIdx  [bnnz]  int     — column indices of B
 *   [5] bRowPtr  [k+1]   int     — row pointers of B
 * IArgs:
 *   [0] m  — rows in A / rows in C
 *   [1] k  — cols in A / rows in B
 *   [2] n  — cols in B / cols in C
 * Outputs:
 *   [0] cValues  [cnnz]  float   — non-zero values of C (same dtype as aValues)
 *   [1] cColIdx  [cnnz]  INT32   — column indices of C
 *   [2] cRowPtr  [m+1]   INT32   — row pointers of C
 */
#if NOT_EXCLUDED(OP_csr_spgemm)
DECLARE_CUSTOM_OP(csr_spgemm, 6, 3, false, 0, 3);
#endif

/**
 * CSR sparse-SDDMM: sampled sparse×sparseᵀ dot product.
 *
 * For each target nonzero t at (row p, col q = targetColIdx[t]) computes:
 *   outValues[t] = dot(L row p, M row q)
 *                = sum_{c in BOTH L-row-p AND M-row-q} L[p,c] * M[q,c]
 *
 * Equivalently: out = (L · Mᵀ) sampled at the target CSR sparsity pattern.
 * This is the gradient kernel for differentiable sparse SpGEMM.
 *
 * Inputs:
 *   [0] targetRowPtr  [P+1]   int    — target sparsity row pointers
 *   [1] targetColIdx  [tnnz]  int    — target sparsity column indices
 *   [2] Lvalues       [Lnnz]  float  — non-zero values of L [P, R]
 *   [3] LcolIdx       [Lnnz]  int    — column indices of L
 *   [4] LrowPtr       [P+1]   int    — row pointers of L
 *   [5] Mvalues       [Mnnz]  float  — non-zero values of M [Q, R]
 *   [6] McolIdx       [Mnnz]  int    — column indices of M
 *   [7] MrowPtr       [Q+1]   int    — row pointers of M
 * IArgs:
 *   [0] P   — rows in L and target
 *   [1] Q   — rows in M and cols in target
 *   [2] R   — inner dimension (cols in both L and M)
 * Output:
 *   [0] outValues [tnnz]  float  — same dtype as Lvalues
 */
#if NOT_EXCLUDED(OP_csr_sddmm_sparse)
DECLARE_CUSTOM_OP(csr_sddmm_sparse, 8, 1, false, 0, 3);
#endif

/**
 * CSR sparse-sparse elementwise addition: C = A + B
 *
 * A and B must be [m, n] in CSR format (same logical shape).
 * C is produced in CSR format [m, n] whose sparsity pattern is the
 * column-set UNION of A and B per row; overlapping column entries are summed.
 * The output nnz is determined symbolically (sorted-merge union per row) in
 * DECLARE_SHAPE_FN, so output buffers are exactly sized.
 *
 * Inputs:
 *   [0] aValues  [annz]  float   — non-zero values of A
 *   [1] aColIdx  [annz]  int     — column indices of A  (sorted per row)
 *   [2] aRowPtr  [m+1]   int     — row pointers of A
 *   [3] bValues  [bnnz]  float   — non-zero values of B
 *   [4] bColIdx  [bnnz]  int     — column indices of B  (sorted per row)
 *   [5] bRowPtr  [m+1]   int     — row pointers of B
 * IArgs:
 *   [0] m  — number of rows
 *   [1] n  — number of columns
 * Outputs:
 *   [0] cValues  [cnnz]  float   — non-zero values of C (same dtype as aValues)
 *   [1] cColIdx  [cnnz]  INT32   — column indices of C
 *   [2] cRowPtr  [m+1]   INT32   — row pointers of C
 */
#if NOT_EXCLUDED(OP_csr_add)
// Expanded from DECLARE_CUSTOM_OP(csr_add, 6, 3, false, 0, 2) to override emptyHandling() = EMPTY_EXECUTE.
// csr_add produces a NON-EMPTY result when ONE operand is an empty (all-zero) CSR (A + 0 = A), but the
// default EMPTY_SKIP (DeclarableOp.cpp ~1028) makes the op early-return Status::OK WITHOUT executing
// whenever ANY input array is empty — leaving the shape-fn-sized output zero-initialized (cRowPtr
// unfilled → rowPtr[rows]=0). EMPTY_EXECUTE runs the op so the helper (CPU sorted-merge / CUDA
// copy-the-non-empty-operand) fills the correct union result. The op-signature counts remain encoded
// in CUSTOM_OP_IMPL(csr_add, 6, 3, false, 0, 2).
SD_BACKEND_OPS_INLINE_NAMESPACE_BEGIN
class SD_LIB_EXPORT csr_add : public sd::ops::DeclarableCustomOp {
 protected:
  void registerTypes();
  SD_DECLARABLE_OP_EXECUTION_METHODS

 public:
  csr_add();
  sd::ShapeList* calculateOutputShape(sd::ShapeList* inputShape, sd::graph::Context& block);
  samediff::EmptyHandling emptyHandling() override { return samediff::EmptyHandling::EMPTY_EXECUTE; }
};
SD_BACKEND_OPS_INLINE_NAMESPACE_END
REGISTER_H(csr_add)
#endif

/**
 * Converts a CSR sparse matrix to CSC sparse format.
 * (CSC of A == CSR of Aᵀ relabelled — this op therefore gives sparse transpose.)
 *
 * Inputs:
 *   [0] csrValues  [nnz],     float dtype
 *   [1] csrColIdx  [nnz],     INT32 or INT64
 *   [2] csrRowPtr  [rows+1],  same INT dtype
 * IArgs:
 *   [0] rows
 *   [1] cols
 * Outputs:
 *   [0] cscValues  [nnz],     same float dtype as csrValues
 *   [1] cscRowIdx  [nnz],     INT32
 *   [2] cscColPtr  [cols+1],  INT32
 */
#if NOT_EXCLUDED(OP_csr_to_csc)
DECLARE_CUSTOM_OP(csr_to_csc, 3, 3, false, 0, 2);
#endif

/**
 * Converts a CSC sparse matrix to a dense NDArray.
 *
 * Inputs:
 *   [0] cscValues  [nnz],     float dtype
 *   [1] cscRowIdx  [nnz],     INT32 or INT64
 *   [2] cscColPtr  [cols+1],  same INT dtype as cscRowIdx
 * IArgs:
 *   [0] rows
 *   [1] cols
 * Output:
 *   [0] dense [rows, cols], dtype = cscValues dtype
 */
#if NOT_EXCLUDED(OP_csc_to_dense)
DECLARE_CUSTOM_OP(csc_to_dense, 3, 1, false, 0, 2);
#endif

/**
 * Converts a dense matrix to CSC sparse representation (column-major scan).
 *
 * Input:
 *   [0] dense [rows, cols], floating dtype
 * TArgs:
 *   [0] threshold (default 0.0 — keep entries where |x| > threshold)
 * Outputs:
 *   [0] cscValues  [nnz],     same dtype as input
 *   [1] cscRowIdx  [nnz],     INT32
 *   [2] cscColPtr  [cols+1],  INT32
 */
#if NOT_EXCLUDED(OP_dense_to_csc)
DECLARE_CUSTOM_OP(dense_to_csc, 1, 3, false, 1, 0);
#endif

/**
 * Converts a CSR sparse matrix to BSR (block-sparse-row) format.
 *
 * Inputs:
 *   [0] csrValues  [nnz],     float dtype
 *   [1] csrColIdx  [nnz],     INT32 or INT64
 *   [2] csrRowPtr  [rows+1],  same INT dtype
 * IArgs:
 *   [0] rows
 *   [1] cols
 *   [2] blockDim   — block size (rows and cols must be divisible by blockDim)
 * Outputs:
 *   [0] bsrValues  [nnzb*blockDim*blockDim], same float dtype as csrValues
 *   [1] bsrColIdx  [nnzb],    INT32
 *   [2] bsrRowPtr  [mb+1],    INT32  (mb = rows/blockDim)
 */
#if NOT_EXCLUDED(OP_csr_to_bsr)
DECLARE_CUSTOM_OP(csr_to_bsr, 3, 3, false, 0, 3);
#endif

/**
 * Converts BSR (block-sparse-row) sparse format to a dense matrix.
 *
 * Inputs:
 *   [0] bsrValues  [nnzb*blockDim*blockDim], float dtype
 *   [1] bsrColIdx  [nnzb],    INT32 or INT64
 *   [2] bsrRowPtr  [mb+1],    same INT dtype  (mb = rows/blockDim)
 * IArgs:
 *   [0] rows
 *   [1] cols
 *   [2] blockDim
 * Output:
 *   [0] dense [rows, cols], dtype = bsrValues dtype
 */
#if NOT_EXCLUDED(OP_bsr_to_dense)
DECLARE_CUSTOM_OP(bsr_to_dense, 3, 1, false, 0, 3);
#endif

/**
 * BSR sparse matrix-matrix multiply: C = A_bsr * B
 *
 * A_bsr is [rows, cols] in BSR format. B is dense [cols, n]. C is dense [rows, n].
 *
 * Inputs:
 *   [0] bsrValues  [nnzb*blockDim*blockDim], float dtype
 *   [1] bsrColIdx  [nnzb],    INT32 or INT64
 *   [2] bsrRowPtr  [mb+1],    same INT dtype
 *   [3] B          dense [cols, n], same float dtype
 * IArgs:
 *   [0] rows
 *   [1] cols
 *   [2] blockDim
 * Output:
 *   [0] C dense [rows, n], same float dtype
 */
#if NOT_EXCLUDED(OP_bsr_spmm)
DECLARE_CUSTOM_OP(bsr_spmm, 4, 1, false, 0, 3);
#endif


/**
 * Build the CSR representation of an n×n diagonal matrix from a float vector.
 *
 * Given diag[n], produces the diagonal matrix with diag[i] at (i,i).
 * nnz = n (not data-dependent); useful for constructing sparse identity
 * and degree matrix D for the graph Laplacian.
 *
 * Input:
 *   [0] diag  [n], floating dtype
 * IArgs:
 *   [0] n  — diagonal size
 * Outputs:
 *   [0] values  [n]   — same dtype as diag
 *   [1] colIdx  [n]   — INT32, colIdx[i] = i
 *   [2] rowPtr  [n+1] — INT32, rowPtr[i] = i
 */
#if NOT_EXCLUDED(OP_spdiags)
DECLARE_CUSTOM_OP(spdiags, 1, 3, false, 0, 1);
#endif

/**
 * Two-sided diagonal scaling of a CSR matrix: out[i,j] = dl[i] * A[i,j] * dr[j].
 *
 * This implements the GCN normalization D^{-1/2} A D^{-1/2} (and related).
 * Only the value array changes; the sparsity structure is unchanged (NOT output).
 * The op is a clean elementwise scale, differentiable on the Java SameDiff side.
 *
 * Inputs:
 *   [0] aValues  [nnz]     float — stored nonzero values of A
 *   [1] aColIdx  [nnz]     int   — column indices of A (sorted per row)
 *   [2] aRowPtr  [rows+1]  int   — row pointers of A
 *   [3] dl       [rows]    float — left diagonal
 *   [4] dr       [cols]    float — right diagonal
 * IArgs:
 *   [0] rows
 *   [1] cols
 * Output:
 *   [0] outValues [nnz]  float — same dtype as aValues
 */
#if NOT_EXCLUDED(OP_csr_diag_mm)
DECLARE_CUSTOM_OP(csr_diag_mm, 5, 1, false, 0, 2);
#endif

/**
 * Backprop (_bp) ops for the differentiable sparse ops. Each forward op's doDiff
 * constructs the matching _bp op (canonical nd4j forward/_bp pattern). Inputs are
 * the forward inputs + the upstream gradient(s); outputs are the input gradients.
 */
#if NOT_EXCLUDED(OP_csr_to_dense_bp)
DECLARE_CUSTOM_OP(csr_to_dense_bp, 3, 1, false, 0, 2);
#endif
#if NOT_EXCLUDED(OP_dense_to_csr_bp)
DECLARE_CUSTOM_OP(dense_to_csr_bp, 3, 1, false, 0, 2);
#endif
#if NOT_EXCLUDED(OP_csr_to_csc_bp)
DECLARE_CUSTOM_OP(csr_to_csc_bp, 5, 1, false, 0, 2);
#endif
#if NOT_EXCLUDED(OP_dense_to_coo_bp)
DECLARE_CUSTOM_OP(dense_to_coo_bp, 2, 1, false, 0, 2);
#endif
#if NOT_EXCLUDED(OP_coo_to_csr_bp)
DECLARE_CUSTOM_OP(coo_to_csr_bp, 4, 1, false, 0, 2);
#endif
#if NOT_EXCLUDED(OP_csr_spmv_bp)
DECLARE_CUSTOM_OP(csr_spmv_bp, 5, 2, false, 0, 3);
#endif
#if NOT_EXCLUDED(OP_csr_spmm_bp)
DECLARE_CUSTOM_OP(csr_spmm_bp, 5, 2, false, 0, 3);
#endif
#if NOT_EXCLUDED(OP_sddmm_bp)
DECLARE_CUSTOM_OP(sddmm_bp, 5, 2, false, 0, 2);
#endif
#if NOT_EXCLUDED(OP_csr_spgemm_bp)
DECLARE_CUSTOM_OP(csr_spgemm_bp, 9, 2, false, 0, 3);
#endif
#if NOT_EXCLUDED(OP_csr_diag_mm_bp)
DECLARE_CUSTOM_OP(csr_diag_mm_bp, 6, 3, false, 0, 2);
#endif

/**
 * GNN message-passing ops: edge-softmax (GAT), neighbor max-aggregation (GraphSAGE-max),
 * edge gather + segment scatter-reduce (general MPNN). Differentiable via native _bp.
 * (csr_spmv_semiring / csr_spmm_semiring are declared in headers/sparse_semiring.h.)
 */
#if NOT_EXCLUDED(OP_csr_row_softmax)
DECLARE_CUSTOM_OP(csr_row_softmax, 2, 1, false, 0, 1);
#endif
#if NOT_EXCLUDED(OP_csr_row_softmax_bp)
DECLARE_CUSTOM_OP(csr_row_softmax_bp, 3, 1, false, 0, 1);
#endif
#if NOT_EXCLUDED(OP_csr_segment_max)
DECLARE_CUSTOM_OP(csr_segment_max, 3, 1, false, 0, 1);
#endif
#if NOT_EXCLUDED(OP_csr_segment_max_bp)
DECLARE_CUSTOM_OP(csr_segment_max_bp, 4, 1, false, 0, 1);
#endif
#if NOT_EXCLUDED(OP_csr_edge_gather)
DECLARE_CUSTOM_OP(csr_edge_gather, 2, 1, false, 0, 0);
#endif
#if NOT_EXCLUDED(OP_csr_edge_gather_bp)
DECLARE_CUSTOM_OP(csr_edge_gather_bp, 3, 1, false, 0, 0);
#endif
#if NOT_EXCLUDED(OP_csr_edge_aggregate)
DECLARE_CUSTOM_OP(csr_edge_aggregate, 2, 1, false, 0, 2);
#endif
#if NOT_EXCLUDED(OP_csr_edge_aggregate_bp)
DECLARE_CUSTOM_OP(csr_edge_aggregate_bp, 3, 1, false, 0, 2);
#endif

/**
 * Extracts the induced subgraph for a selected set of K nodes from a CSR graph.
 *
 * Keeps edge (i -> j) iff BOTH i and j appear in nodeIdx (sorted ascending).
 * Remaps both endpoints to their 0-based position in nodeIdx.
 * nnz' is data-dependent and is counted exactly in DECLARE_SHAPE_FN.
 *
 * Inputs:
 *   [0] values   [nnz]   float  — edge weights of the original graph
 *   [1] colIdx   [nnz]   int    — column indices (destination node ids)
 *   [2] rowPtr   [N+1]   int    — row pointer array of the original graph
 *   [3] nodeIdx  [K]     int    — SORTED ascending selected node ids
 * IArgs:
 *   [0] N  — original node count
 *   [1] K  — number of selected nodes
 * Outputs:
 *   [0] newValues  [nnz']  float  — edge weights of the extracted subgraph
 *   [1] newColIdx  [nnz']  INT32  — remapped column indices (0..K-1)
 *   [2] newRowPtr  [K+1]   INT32  — row pointers of the extracted subgraph
 */
#if NOT_EXCLUDED(OP_csr_subgraph_extract)
DECLARE_CUSTOM_OP(csr_subgraph_extract, 4, 3, false, 0, 2);
#endif

/**
 * Backward pass for csr_subgraph_extract.
 *
 * Gradient flows only through values: for each kept edge e -> e',
 *   dValues[e] = dNewValues[e']
 * Dropped edges get zero gradient. Structural inputs receive zero gradients.
 *
 * Inputs:
 *   [0] values      [nnz]   float  — forward input edge weights
 *   [1] colIdx      [nnz]   int    — forward input column indices
 *   [2] rowPtr      [N+1]   int    — forward input row pointers
 *   [3] nodeIdx     [K]     int    — forward input selected node ids (sorted)
 *   [4] dNewValues  [nnz']  float  — upstream gradient w.r.t. newValues
 * IArgs:
 *   [0] N
 *   [1] K
 * Output:
 *   [0] dValues  [nnz]  float  — gradient w.r.t. values
 */
#if NOT_EXCLUDED(OP_csr_subgraph_extract_bp)
DECLARE_CUSTOM_OP(csr_subgraph_extract_bp, 5, 1, false, 0, 2);
#endif

/**
 * Backward pass for csr_add: gradient flows to A.values and B.values.
 * Inputs: aColIdx, aRowPtr, bColIdx, bRowPtr, cColIdx, cRowPtr, gradCValues
 * IArgs: m, n
 * Outputs: dAValues, dBValues
 */
#if NOT_EXCLUDED(OP_csr_add_bp)
DECLARE_CUSTOM_OP(csr_add_bp, 7, 2, false, 0, 2);
#endif

/**
 * Backward pass for bsr_to_dense.
 * Inputs: bsrColIdx, bsrRowPtr, gradDense   IArgs: rows, cols, blockDim
 * Output: dBsrValues
 */
#if NOT_EXCLUDED(OP_bsr_to_dense_bp)
DECLARE_CUSTOM_OP(bsr_to_dense_bp, 3, 1, false, 0, 3);
#endif

/**
 * Backward pass for bsr_spmm.
 * Inputs: bsrValues, bsrColIdx, bsrRowPtr, B, gradC   IArgs: rows, cols, blockDim
 * Outputs: dBsrValues, dB
 */
#if NOT_EXCLUDED(OP_bsr_spmm_bp)
DECLARE_CUSTOM_OP(bsr_spmm_bp, 5, 2, false, 0, 3);
#endif

/**
 * Backward pass for csc_to_dense.
 * Inputs: cscRowIdx, cscColPtr, gradDense   IArgs: rows, cols
 * Output: dCscValues
 */
#if NOT_EXCLUDED(OP_csc_to_dense_bp)
DECLARE_CUSTOM_OP(csc_to_dense_bp, 3, 1, false, 0, 2);
#endif

/**
 * Backward pass for csr_to_bsr.
 * Inputs: csrColIdx, csrRowPtr, bsrColIdx, bsrRowPtr, gradBsrValues   IArgs: rows, cols, blockDim
 * Output: dCsrValues
 */
#if NOT_EXCLUDED(OP_csr_to_bsr_bp)
DECLARE_CUSTOM_OP(csr_to_bsr_bp, 5, 1, false, 0, 3);
#endif

/**
 * Backward pass for dense_to_csc.
 * Inputs: cscRowIdx, cscColPtr, gradCscValues   IArgs: rows, cols
 * Output: dDense [rows, cols]
 */
#if NOT_EXCLUDED(OP_dense_to_csc_bp)
DECLARE_CUSTOM_OP(dense_to_csc_bp, 3, 1, false, 0, 2);
#endif

/**
 * Backward pass for csr_sddmm_sparse.
 * Inputs: targetRowPtr, targetColIdx, LcolIdx, LrowPtr, McolIdx, MrowPtr, Lvalues, Mvalues, gradOut
 * IArgs: P, Q, R
 * Outputs: dLvalues, dMvalues
 */
#if NOT_EXCLUDED(OP_csr_sddmm_sparse_bp)
DECLARE_CUSTOM_OP(csr_sddmm_sparse_bp, 9, 2, false, 0, 3);
#endif

/**
 * Build a block-diagonal batched graph from K variable-size graphs.
 *
 * Input layout: 4*K inputs (K Xs [N_k,F], K vals [nnz_k], K colIdxs [nnz_k], K rowPtrs [N_k+1])
 * IArgs: [0] K
 * Outputs: X_combined [sumN,F], vals_combined [sumNnz], colIdx_combined [sumNnz],
 *          rowPtr_combined [sumN+1], batchVec [sumN] (INT32)
 */
#if NOT_EXCLUDED(OP_graph_disjoint_union)
DECLARE_CUSTOM_OP(graph_disjoint_union, -2, 5, false, 0, 1);
DECLARE_CUSTOM_OP(graph_disjoint_union_bp, -2, -2, false, 0, 1);
#endif

}  // namespace ops
}  // namespace sd

#endif  // SAMEDIFF_SPARSE_H

/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.codegen.ops

import org.nd4j.codegen.api.DataType.*
import org.nd4j.codegen.api.Language
import org.nd4j.codegen.api.doc.DocScope
import org.nd4j.codegen.dsl.*

fun Sparse() = Namespace("Sparse") {
    val namespaceJavaPackage = "org.nd4j.linalg.api.ops.impl.sparse"

    // -------------------------------------------------------------------------
    // Conversions
    // -------------------------------------------------------------------------

    Op("csrToDense") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "CsrToDense"

        Input(FLOATING_POINT, "values") { description = "1D [nnz] non-zero values of the CSR matrix" }
        Input(INT, "colIdx")            { description = "1D [nnz] column indices (INT32/INT64)" }
        Input(INT, "rowPtr")            { description = "1D [rows+1] row pointers (INT32/INT64)" }
        Arg(INT, "rows")                { description = "Number of rows in the dense output" }
        Arg(INT, "cols")                { description = "Number of columns in the dense output" }

        Output(FLOATING_POINT, "dense") { description = "Dense matrix [rows, cols]" }

        Doc(Language.ANY, DocScope.ALL) {
            """
            Convert a CSR (Compressed Sparse Row) sparse matrix to a dense matrix.
            Inputs are the three CSR component arrays (values, colIdx, rowPtr) plus integer shape arguments rows and cols.
            """.trimIndent()
        }
    }

    Op("denseToCsr") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "DenseToCsr"

        Input(FLOATING_POINT, "dense")   { description = "2D dense input matrix [rows, cols]" }
        Arg(FLOATING_POINT, "threshold") { description = "Keep entries where |x| > threshold (0.0 keeps all non-zeros)" }

        Output(FLOATING_POINT, "values") { description = "1D [nnz] non-zero values" }
        Output(INT, "colIdx")            { description = "1D [nnz] column indices (INT32)" }
        Output(INT, "rowPtr")            { description = "1D [rows+1] row pointers (INT32)" }

        Doc(Language.ANY, DocScope.ALL) {
            """
            Convert a dense matrix to CSR (Compressed Sparse Row) sparse representation.
            Only entries with |x| > threshold are kept; pass threshold=0.0 to retain all structurally non-zero entries.
            """.trimIndent()
        }
    }

    Op("csrToCsc") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "CsrToCsc"

        Input(FLOATING_POINT, "values") { description = "1D [nnz] CSR non-zero values" }
        Input(INT, "colIdx")            { description = "1D [nnz] CSR column indices (INT32)" }
        Input(INT, "rowPtr")            { description = "1D [rows+1] CSR row pointers (INT32)" }
        Arg(INT, "rows")                { description = "Number of rows in the logical matrix" }
        Arg(INT, "cols")                { description = "Number of columns in the logical matrix" }

        Output(FLOATING_POINT, "cscValues") { description = "1D [nnz] non-zero values in column-major order" }
        Output(INT, "cscRowIdx")            { description = "1D [nnz] row indices for each non-zero (INT32)" }
        Output(INT, "cscColPtr")            { description = "1D [cols+1] column pointers (INT32)" }

        Doc(Language.ANY, DocScope.ALL) {
            """
            Convert a CSR sparse matrix to CSC (Compressed Sparse Column) format.
            The CSC of A is algebraically identical to the CSR of Aᵀ, so the output also provides a free sparse transpose.
            """.trimIndent()
        }
    }

    Op("cscToDense") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "CscToDense"

        Input(FLOATING_POINT, "cscValues") { description = "1D [nnz] CSC non-zero values in column-major order" }
        Input(INT, "cscRowIdx")            { description = "1D [nnz] row index for each non-zero (INT32)" }
        Input(INT, "cscColPtr")            { description = "1D [cols+1] column pointers (INT32)" }
        Arg(INT, "rows")                   { description = "Number of rows in the dense output" }
        Arg(INT, "cols")                   { description = "Number of columns in the dense output" }

        Output(FLOATING_POINT, "dense") { description = "Dense matrix [rows, cols]" }

        Doc(Language.ANY, DocScope.ALL) {
            """
            Convert a CSC (Compressed Sparse Column) sparse matrix to a dense matrix.
            """.trimIndent()
        }
    }

    Op("denseToCsc") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "DenseToCsc"

        Input(FLOATING_POINT, "dense")   { description = "2D dense input matrix [rows, cols]" }
        Arg(FLOATING_POINT, "threshold") { description = "Keep entries where |x| > threshold (0.0 keeps all non-zeros)" }

        Output(FLOATING_POINT, "cscValues") { description = "1D [nnz] non-zero values in column-major order" }
        Output(INT, "cscRowIdx")            { description = "1D [nnz] row index for each non-zero (INT32)" }
        Output(INT, "cscColPtr")            { description = "1D [cols+1] column pointers (INT32)" }

        Doc(Language.ANY, DocScope.ALL) {
            """
            Convert a dense matrix to CSC (Compressed Sparse Column) sparse representation.
            Only entries with |x| > threshold are kept.
            """.trimIndent()
        }
    }

    Op("denseToCoo") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "DenseToCoo"

        Input(FLOATING_POINT, "dense")   { description = "2D dense input matrix [rows, cols]" }
        Arg(FLOATING_POINT, "threshold") { description = "Keep entries where |x| > threshold (0.0 keeps all non-zeros)" }

        Output(INT, "indices")           { description = "2D [nnz, 2] INT64 row/col index pairs for each non-zero" }
        Output(FLOATING_POINT, "values") { description = "1D [nnz] non-zero values" }

        Doc(Language.ANY, DocScope.ALL) {
            """
            Convert a dense matrix to COO (Coordinate) sparse representation.
            Returns indices [nnz, 2] (INT64) and values [nnz] in corresponding order.
            """.trimIndent()
        }
    }

    Op("cooToCsr") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "CooToCsr"

        Input(INT, "indices")           { description = "2D [nnz, 2] INT64 row/col index pairs for each non-zero" }
        Input(FLOATING_POINT, "values") { description = "1D [nnz] non-zero values" }
        Arg(INT, "rows")                { description = "Number of rows in the logical dense shape" }
        Arg(INT, "cols")                { description = "Number of columns in the logical dense shape" }

        Output(FLOATING_POINT, "csrValues") { description = "1D [nnz] non-zero values in CSR row-major order" }
        Output(INT, "colIdx")               { description = "1D [nnz] column indices (INT32)" }
        Output(INT, "rowPtr")               { description = "1D [rows+1] row pointers (INT32)" }

        Doc(Language.ANY, DocScope.ALL) {
            """
            Convert a COO (Coordinate) sparse matrix to CSR (Compressed Sparse Row) format.
            The COO entries are sorted into row-major order by the native op.
            """.trimIndent()
        }
    }

    Op("csrToBsr") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "CsrToBsr"

        Input(FLOATING_POINT, "csrValues") { description = "1D [nnz] CSR non-zero values" }
        Input(INT, "csrColIdx")            { description = "1D [nnz] CSR column indices (INT32)" }
        Input(INT, "csrRowPtr")            { description = "1D [rows+1] CSR row pointers (INT32)" }
        Arg(INT, "rows")                   { description = "Number of rows (must be a multiple of blockDim)" }
        Arg(INT, "cols")                   { description = "Number of columns (must be a multiple of blockDim)" }
        Arg(INT, "blockDim")               { description = "Square block size; rows and cols must be exact multiples" }

        Output(FLOATING_POINT, "bsrValues") { description = "1D [nnzb * blockDim * blockDim] BSR non-zero block values" }
        Output(INT, "bsrColIdx")            { description = "1D [nnzb] block-column indices (INT32)" }
        Output(INT, "bsrRowPtr")            { description = "1D [mb+1] block-row pointers (INT32), mb = rows / blockDim" }

        Doc(Language.ANY, DocScope.ALL) {
            """
            Convert a CSR sparse matrix to BSR (Block Sparse Row) format.
            Both rows and cols must be exact multiples of blockDim.
            """.trimIndent()
        }
    }

    Op("bsrToDense") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "BsrToDense"

        Input(FLOATING_POINT, "bsrValues") { description = "1D [nnzb * blockDim * blockDim] BSR non-zero block values" }
        Input(INT, "bsrColIdx")            { description = "1D [nnzb] block-column indices (INT32)" }
        Input(INT, "bsrRowPtr")            { description = "1D [mb+1] block-row pointers (INT32), mb = rows / blockDim" }
        Arg(INT, "rows")                   { description = "Number of rows in the logical dense matrix" }
        Arg(INT, "cols")                   { description = "Number of columns in the logical dense matrix" }
        Arg(INT, "blockDim")               { description = "Square block size" }

        Output(FLOATING_POINT, "dense") { description = "Dense matrix [rows, cols]" }

        Doc(Language.ANY, DocScope.ALL) {
            """
            Convert a BSR (Block Sparse Row) sparse matrix to a dense matrix.
            """.trimIndent()
        }
    }

    // -------------------------------------------------------------------------
    // Compute / BLAS
    // -------------------------------------------------------------------------

    Op("csrSpmv") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "CsrSpmv"

        Input(FLOATING_POINT, "values") { description = "1D [nnz] CSR non-zero values" }
        Input(INT, "colIdx")            { description = "1D [nnz] CSR column indices (INT32)" }
        Input(INT, "rowPtr")            { description = "1D [rows+1] CSR row pointers (INT32)" }
        Input(FLOATING_POINT, "x")      { description = "1D dense vector: length cols (or rows when transposeA=true)" }
        Arg(INT, "rows")                { description = "Number of rows in the sparse matrix" }
        Arg(INT, "cols")                { description = "Number of columns in the sparse matrix" }
        Arg(BOOL, "transposeA")         { description = "If true compute Aᵀ·x instead of A·x" }

        Output(FLOATING_POINT, "y") { description = "Result vector of length rows (or cols when transposeA=true)" }

        Doc(Language.ANY, DocScope.ALL) {
            """
            CSR sparse matrix-vector product: y = A·x (or Aᵀ·x when transposeA=true).
            """.trimIndent()
        }
    }

    Op("csrSpmm") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "CsrSpmm"

        Input(FLOATING_POINT, "values") { description = "1D [nnz] CSR non-zero values" }
        Input(INT, "colIdx")            { description = "1D [nnz] CSR column indices (INT32)" }
        Input(INT, "rowPtr")            { description = "1D [rows+1] CSR row pointers (INT32)" }
        Input(FLOATING_POINT, "B")      { description = "2D dense matrix [cols, n] (or [rows, n] when transposeA=true)" }
        Arg(INT, "rows")                { description = "Number of rows in the sparse matrix A" }
        Arg(INT, "cols")                { description = "Number of columns in the sparse matrix A" }
        Arg(BOOL, "transposeA")         { description = "If true compute Aᵀ·B instead of A·B" }

        Output(FLOATING_POINT, "C") { description = "Dense result matrix [rows, n] (or [cols, n] when transposeA=true)" }

        Doc(Language.ANY, DocScope.ALL) {
            """
            CSR sparse matrix-matrix product: C = A·B (or Aᵀ·B when transposeA=true).
            """.trimIndent()
        }
    }

    Op("sddmm") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "Sddmm"

        Input(INT, "rowPtr")            { description = "1D [rows+1] INT32 row pointers of the sparsity pattern" }
        Input(INT, "colIdx")            { description = "1D [nnz] INT32 column indices of the sparsity pattern" }
        Input(FLOATING_POINT, "D1")     { description = "2D [rows, p] left dense factor" }
        Input(FLOATING_POINT, "D2")     { description = "2D [cols, p] right dense factor" }
        Arg(INT, "rows")                { description = "Number of rows in the sparsity pattern" }
        Arg(INT, "cols")                { description = "Number of columns in the sparsity pattern" }

        Output(FLOATING_POINT, "values") { description = "1D [nnz] sampled values of D1·D2ᵀ at the pattern positions" }

        Doc(Language.ANY, DocScope.ALL) {
            """
            Sampled Dense-Dense Matrix Multiplication (SDDMM).
            For each non-zero position (i,j) in the sparsity pattern computes sum_l D1[i,l]*D2[j,l].
            """.trimIndent()
        }
    }

    Op("csrSpgemm") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "CsrSpgemm"

        Input(FLOATING_POINT, "aValues") { description = "1D [nnzA] non-zero values of A" }
        Input(INT, "aColIdx")            { description = "1D [nnzA] column indices of A (INT32)" }
        Input(INT, "aRowPtr")            { description = "1D [m+1] row pointers of A (INT32)" }
        Input(FLOATING_POINT, "bValues") { description = "1D [nnzB] non-zero values of B (same dtype as aValues)" }
        Input(INT, "bColIdx")            { description = "1D [nnzB] column indices of B (INT32)" }
        Input(INT, "bRowPtr")            { description = "1D [k+1] row pointers of B (INT32)" }
        Arg(INT, "m")                    { description = "Number of rows of A (= rows of C)" }
        Arg(INT, "k")                    { description = "Number of columns of A (= rows of B)" }
        Arg(INT, "n")                    { description = "Number of columns of B (= columns of C)" }

        Output(FLOATING_POINT, "cValues") { description = "1D [cnnz] non-zero values of C; cnnz is data-dependent" }
        Output(INT, "cColIdx")            { description = "1D [cnnz] column indices of C (INT32)" }
        Output(INT, "cRowPtr")            { description = "1D [m+1] row pointers of C (INT32)" }

        Doc(Language.ANY, DocScope.ALL) {
            """
            CSR sparse matrix-matrix multiplication (SpGEMM): C = A·B.
            Both A and B are in CSR format; output C is also in CSR format.
            The output nnz is data-dependent and determined by the native shape function.
            """.trimIndent()
        }
    }

    Op("bsrSpmm") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "BsrSpmm"

        Input(FLOATING_POINT, "bsrValues") { description = "1D [nnzb * blockDim * blockDim] BSR non-zero block values of A" }
        Input(INT, "bsrColIdx")            { description = "1D [nnzb] block-column indices of A (INT32)" }
        Input(INT, "bsrRowPtr")            { description = "1D [mb+1] block-row pointers of A (INT32), mb = rows / blockDim" }
        Input(FLOATING_POINT, "B")         { description = "2D [cols, n] dense right-hand matrix" }
        Arg(INT, "rows")                   { description = "Number of rows in A (and rows of C)" }
        Arg(INT, "cols")                   { description = "Number of columns in A (and rows of B)" }
        Arg(INT, "blockDim")               { description = "Square block size used in the BSR representation" }

        Output(FLOATING_POINT, "C") { description = "Dense result matrix [rows, n]" }

        Doc(Language.ANY, DocScope.ALL) {
            """
            BSR sparse matrix-dense matrix multiplication: C = A_bsr·B.
            A is in BSR format; B and C are dense.
            Equivalent to toDense(A_bsr).mmul(B) but skips zero blocks.
            """.trimIndent()
        }
    }

    Op("csrAdd") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "CsrAdd"

        Input(FLOATING_POINT, "aValues") { description = "1D [nnzA] non-zero values of A" }
        Input(INT, "aColIdx")            { description = "1D [nnzA] column indices of A (INT32)" }
        Input(INT, "aRowPtr")            { description = "1D [m+1] row pointers of A (INT32)" }
        Input(FLOATING_POINT, "bValues") { description = "1D [nnzB] non-zero values of B (same dtype as aValues)" }
        Input(INT, "bColIdx")            { description = "1D [nnzB] column indices of B (INT32)" }
        Input(INT, "bRowPtr")            { description = "1D [m+1] row pointers of B (INT32)" }
        Arg(INT, "m")                    { description = "Number of rows (same for A and B)" }
        Arg(INT, "n")                    { description = "Number of columns (same for A and B)" }

        Output(FLOATING_POINT, "cValues") { description = "1D [cnnz] non-zero values of C = A + B" }
        Output(INT, "cColIdx")            { description = "1D [cnnz] column indices of C (INT32)" }
        Output(INT, "cRowPtr")            { description = "1D [m+1] row pointers of C (INT32)" }

        Doc(Language.ANY, DocScope.ALL) {
            """
            Elementwise CSR sparse matrix addition: C = A + B.
            Both A and B must have the same logical shape [m, n].
            This op is forward-only; automatic differentiation is not supported.
            """.trimIndent()
        }
    }

    Op("csrDiagMm") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "CsrDiagMm"

        Input(FLOATING_POINT, "aValues") { description = "1D [nnz] CSR non-zero values of A" }
        Input(INT, "aColIdx")            { description = "1D [nnz] column indices of A (INT32)" }
        Input(INT, "aRowPtr")            { description = "1D [rows+1] row pointers of A (INT32)" }
        Input(FLOATING_POINT, "dl")      { description = "1D [rows] left diagonal scaling vector" }
        Input(FLOATING_POINT, "dr")      { description = "1D [cols] right diagonal scaling vector" }
        Arg(INT, "rows")                 { description = "Number of rows in the sparse matrix" }
        Arg(INT, "cols")                 { description = "Number of columns in the sparse matrix" }

        Output(FLOATING_POINT, "outValues") { description = "1D [nnz] scaled non-zero values; sparsity structure unchanged" }

        Doc(Language.ANY, DocScope.ALL) {
            """
            Diagonal-scaled sparse matrix product: out[e] = dl[i]*aValues[e]*dr[j] for each non-zero (i,j).
            Computes the non-zero values of Dl·A·Dr where Dl=diag(dl), Dr=diag(dr), keeping the sparsity pattern intact.
            """.trimIndent()
        }
    }

    Op("spdiags") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "Spdiags"

        Input(FLOATING_POINT, "diag") { description = "1D [n] diagonal values" }
        Arg(INT, "n")                 { description = "Size of the resulting n×n square matrix" }

        Output(FLOATING_POINT, "values") { description = "1D [n] non-zero values (identical to diag)" }
        Output(INT, "colIdx")            { description = "1D [n] column indices [0,1,...,n-1] (INT32)" }
        Output(INT, "rowPtr")            { description = "1D [n+1] row pointers [0,1,...,n] (INT32)" }

        Doc(Language.ANY, DocScope.ALL) {
            """
            Build an n×n diagonal CSR sparse matrix from a 1D diagonal vector.
            The result has exactly n non-zeros, one per diagonal entry.
            """.trimIndent()
        }
    }

    Op("csrSddmmSparse") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "CsrSddmmSparse"

        Input(INT, "targetRowPtr")          { description = "1D [P+1] INT32 row pointers of the target sparsity pattern" }
        Input(INT, "targetColIdx")          { description = "1D [tnnz] INT32 column indices of the target pattern" }
        Input(FLOATING_POINT, "Lvalues")    { description = "1D [Lnnz] non-zero values of L [P, R] in CSR format" }
        Input(INT, "LcolIdx")               { description = "1D [Lnnz] column indices of L (INT32)" }
        Input(INT, "LrowPtr")               { description = "1D [P+1] row pointers of L (INT32)" }
        Input(FLOATING_POINT, "Mvalues")    { description = "1D [Mnnz] non-zero values of M [Q, R] in CSR format" }
        Input(INT, "McolIdx")               { description = "1D [Mnnz] column indices of M (INT32)" }
        Input(INT, "MrowPtr")               { description = "1D [Q+1] row pointers of M (INT32)" }
        Arg(INT, "P")                       { description = "Number of rows in L (= rows of target pattern)" }
        Arg(INT, "Q")                       { description = "Number of rows in M (= cols of target pattern)" }
        Arg(INT, "R")                       { description = "Shared inner dimension: cols of L = cols of M" }

        Output(FLOATING_POINT, "outValues") { description = "1D [tnnz] sampled values of L·Mᵀ at the target pattern positions" }

        Doc(Language.ANY, DocScope.ALL) {
            """
            Sparse-sparse SDDMM: sample L·Mᵀ at positions given by the target CSR sparsity pattern.
            Used as the SpGEMM gradient kernel. Forward-only (no autodiff).
            """.trimIndent()
        }
    }

    // -------------------------------------------------------------------------
    // GNN
    // -------------------------------------------------------------------------

    Op("csrRowSoftmax") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "CsrRowSoftmax"

        Input(FLOATING_POINT, "values") { description = "1D [nnz] non-zero attention logits" }
        Input(INT, "rowPtr")            { description = "1D [rows+1] INT32 CSR row pointers" }
        Arg(INT, "rows")                { description = "Number of rows (source nodes)" }

        Output(FLOATING_POINT, "alpha") { description = "1D [nnz] per-row softmax-normalised attention weights" }

        Doc(Language.ANY, DocScope.ALL) {
            """
            Per-row softmax over CSR non-zero values: the GAT edge-softmax primitive.
            For each row i: alpha[k] = exp(values[k]) / sum_{k' in row i} exp(values[k']).
            """.trimIndent()
        }
    }

    Op("csrSegmentMax") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "CsrSegmentMax"

        Input(INT, "colIdx")       { description = "1D [nnz] INT32 column (source-node) indices" }
        Input(INT, "rowPtr")       { description = "1D [rows+1] INT32 row (segment) pointers" }
        Input(FLOATING_POINT, "X") { description = "2D [n, f] dense node feature matrix" }
        Arg(INT, "rows")           { description = "Number of target-node rows (segments)" }

        Output(FLOATING_POINT, "out") { description = "2D [rows, f] per-segment element-wise maximum of node features" }

        Doc(Language.ANY, DocScope.ALL) {
            """
            Neighbourhood max-aggregation over a CSR graph: the GraphSAGE-max primitive.
            For each row i and feature f: out[i,f] = max over source neighbours j of X[j,f].
            """.trimIndent()
        }
    }

    Op("csrEdgeGather") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "CsrEdgeGather"

        Input(INT, "colIdx")       { description = "1D [nnz] INT32 source-node ids for each edge" }
        Input(FLOATING_POINT, "X") { description = "2D [n, F] dense node-feature matrix" }
        Arg(INT, "n")              { description = "Number of nodes (= X.shape[0]); stored so the backward pass can reconstruct the dX shape [n, F]" }

        Output(FLOATING_POINT, "edgeFeat") { description = "2D [nnz, F] edge feature vectors pulled from X" }

        Doc(Language.ANY, DocScope.ALL) {
            """
            Edge-gather primitive: pull node-feature vectors onto edges.
            For each edge e and feature f: edgeFeat[e,f] = X[colIdx[e],f].
            n = X.shape[0] (number of nodes); required by the backward op to reconstruct dX shape [n, F].
            """.trimIndent()
        }
    }

    Op("csrEdgeAggregate") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "CsrEdgeAggregate"

        Input(INT, "rowPtr")             { description = "1D [rows+1] INT32 CSR row pointers" }
        Input(FLOATING_POINT, "edgeMsg") { description = "2D [nnz, F] per-edge message vectors" }
        Arg(INT, "rows")                 { description = "Number of target nodes (output rows)" }
        Arg(INT, "mode")                 { description = "Aggregation mode: 0=SUM, 1=MEAN, 2=MAX" }

        Output(FLOATING_POINT, "out") { description = "2D [rows, F] aggregated per-node output" }

        Doc(Language.ANY, DocScope.ALL) {
            """
            Segment scatter-reduce: aggregate per-edge messages to per-node outputs (the N-step of MPNN).
            mode: 0=SUM, 1=MEAN, 2=MAX.
            """.trimIndent()
        }
    }

}

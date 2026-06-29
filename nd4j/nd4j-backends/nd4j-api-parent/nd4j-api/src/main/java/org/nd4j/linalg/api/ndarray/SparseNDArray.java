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

package org.nd4j.linalg.api.ndarray;

import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.impl.sparse.BsrSpmm;
import org.nd4j.linalg.api.ops.impl.sparse.BsrToDense;
import org.nd4j.linalg.api.ops.impl.sparse.CooToCsr;
import org.nd4j.linalg.api.ops.impl.sparse.CscToDense;
import org.nd4j.linalg.api.ops.impl.sparse.CsrAdd;
import org.nd4j.linalg.api.ops.impl.sparse.CsrSpgemm;
import org.nd4j.linalg.api.ops.impl.sparse.CsrSpmm;
import org.nd4j.linalg.api.ops.impl.sparse.CsrSpmv;
import org.nd4j.linalg.api.ops.impl.sparse.CsrSpmvSemiring;
import org.nd4j.linalg.api.ops.impl.sparse.CsrSpmmSemiring;
import org.nd4j.linalg.api.ops.impl.sparse.Semiring;
import org.nd4j.linalg.api.ops.impl.sparse.CsrToBsr;
import org.nd4j.linalg.api.ops.impl.sparse.CsrToCsc;
import org.nd4j.linalg.api.ops.impl.sparse.CsrDiagMm;
import org.nd4j.linalg.api.ops.impl.sparse.CsrRowSoftmax;
import org.nd4j.linalg.api.ops.impl.sparse.CsrEdgeAggregate;
import org.nd4j.linalg.api.ops.impl.sparse.CsrEdgeGather;
import org.nd4j.linalg.api.ops.impl.sparse.CsrSegmentMax;
import org.nd4j.linalg.api.ops.impl.sparse.CsrToDense;
import org.nd4j.linalg.api.ops.impl.sparse.Spdiags;
import org.nd4j.linalg.api.ops.compat.CompatSparseToDense;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * A plain container for a sparse tensor stored in CSR (Compressed Sparse Row), COO
 * (Coordinate), or CSC (Compressed Sparse Column) format.
 *
 * <h3>CSR layout</h3>
 * <ul>
 *   <li>{@code values}  – 1D [nnz], floating dtype — the non-zero values</li>
 *   <li>{@code colIdx}  – 1D [nnz], INT32 — column index for each non-zero</li>
 *   <li>{@code rowPtr}  – 1D [rows+1], INT32 — row pointer (rowPtr[r]..rowPtr[r+1]-1 are the
 *       non-zeros in row r)</li>
 * </ul>
 *
 * <h3>COO layout</h3>
 * <ul>
 *   <li>{@code indices} – 2D [nnz, 2], INT64 — (row, col) pairs for each non-zero</li>
 *   <li>{@code values}  – 1D [nnz], floating dtype — the non-zero values</li>
 * </ul>
 *
 * <h3>CSC layout</h3>
 * <ul>
 *   <li>{@code values}  – 1D [nnz], floating dtype — the non-zero values in column-major order</li>
 *   <li>row-index array – 1D [nnz], INT32 — row index for each non-zero
 *       (access via {@link #getRowIdx()})</li>
 *   <li>column-pointer array – 1D [cols+1], INT32 — column pointers
 *       (access via {@link #getColPtr()})</li>
 * </ul>
 *
 * <p>This class does NOT implement {@link INDArray}. It is a thin wrapper that carries
 * the component arrays together with the logical dense shape and format tag.
 * Use {@link #toDense()} to materialise a dense matrix, {@link #toCsr()} to convert
 * a COO instance to CSR, {@link #toCsc()} to convert a CSR instance to CSC, or
 * {@link #transpose()} to obtain the sparse transpose of a CSR matrix.
 */
public class SparseNDArray {

    // CSR/CSC/BSR fields: colIdx and rowPtr are reused across formats.
    //   CSR:  colIdx = column indices [nnz],   rowPtr = row pointers [rows+1]
    //   CSC:  colIdx = row indices    [nnz],   rowPtr = col pointers [cols+1]
    //   BSR:  colIdx = block-col idx  [nnzb],  rowPtr = block-row ptr [mb+1]
    private final INDArray colIdx;
    private final INDArray rowPtr;

    // COO field (non-null only when format == COO)
    private final INDArray indices;

    // Common fields
    private final INDArray values;
    private final long[]   shape;
    private final SparseFormat format;
    private final DataType dataType;

    /**
     * Square block dimension for BSR format; {@code -1} for non-BSR formats.
     * Stored as {@code int} because block sizes are always small (typically 2, 4, 8).
     */
    private final int blockDim;

    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /**
     * Construct a CSR or CSC SparseNDArray from pre-built component arrays.
     *
     * <h4>CSR ({@link SparseFormat#CSR})</h4>
     * <ul>
     *   <li>{@code secondArray} = colIdx — 1D [nnz] INT32 column indices</li>
     *   <li>{@code thirdArray}  = rowPtr — 1D [rows+1] INT32 row pointers</li>
     * </ul>
     *
     * <h4>CSC ({@link SparseFormat#CSC})</h4>
     * <ul>
     *   <li>{@code secondArray} = rowIdx — 1D [nnz] INT32 row indices (access via
     *       {@link #getRowIdx()})</li>
     *   <li>{@code thirdArray}  = colPtr — 1D [cols+1] INT32 column pointers (access via
     *       {@link #getColPtr()})</li>
     * </ul>
     *
     * <p>For CSR, {@code thirdArray} is validated to have length {@code rows+1}.
     * For CSC, {@code thirdArray} is validated to have length {@code cols+1}.
     * The CSR-named accessors {@link #getColIdx()} / {@link #getRowPtr()} are only valid on CSR
     * instances; use {@link #getRowIdx()} / {@link #getColPtr()} for CSC instances.
     *
     * @param values       1D [nnz] non-zero values
     * @param secondArray  colIdx (CSR) or rowIdx (CSC) — 1D [nnz], INT32
     * @param thirdArray   rowPtr (CSR) or colPtr (CSC) — 1D [rows+1] or [cols+1], INT32
     * @param shape        logical dense shape {rows, cols}
     * @param format       {@link SparseFormat#CSR} or {@link SparseFormat#CSC}
     */
    public SparseNDArray(INDArray values, INDArray secondArray, INDArray thirdArray,
                         long[] shape, SparseFormat format) {
        Preconditions.checkNotNull(values,      "values must not be null");
        Preconditions.checkNotNull(secondArray, "secondArray must not be null");
        Preconditions.checkNotNull(thirdArray,  "thirdArray must not be null");
        Preconditions.checkNotNull(shape,       "shape must not be null");
        Preconditions.checkNotNull(format,      "format must not be null");
        Preconditions.checkArgument(format == SparseFormat.CSR || format == SparseFormat.CSC,
                "This constructor supports SparseFormat.CSR and SparseFormat.CSC; got: " + format
                + ". For COO use SparseNDArray(INDArray indices, INDArray values, long[], SparseFormat)."
                + " For BSR use SparseNDArray(INDArray values, INDArray bsrColIdx, INDArray bsrRowPtr, long[], int blockDim, SparseFormat).");
        Preconditions.checkArgument(shape.length == 2,
                "%s sparse format requires a 2D logical shape, got rank %d", format, shape.length);
        if (format == SparseFormat.CSR) {
            Preconditions.checkArgument(thirdArray.length() == shape[0] + 1,
                    "rowPtr length must be rows+1=%d, got %d", shape[0] + 1, thirdArray.length());
        } else {
            // CSC: thirdArray is colPtr, must have length cols+1
            Preconditions.checkArgument(thirdArray.length() == shape[1] + 1,
                    "colPtr length must be cols+1=%d, got %d", shape[1] + 1, thirdArray.length());
        }
        Preconditions.checkArgument(values.length() == secondArray.length(),
                "values and secondArray must have the same length, got %d vs %d",
                values.length(), secondArray.length());

        this.values   = values;
        this.colIdx   = secondArray;   // CSR: colIdx; CSC: rowIdx
        this.rowPtr   = thirdArray;    // CSR: rowPtr; CSC: colPtr
        this.indices  = null;
        this.shape    = shape.clone();
        this.format   = format;
        this.dataType = values.dataType();
        this.blockDim = -1;            // not a BSR instance
    }

    /**
     * Construct a COO SparseNDArray from pre-built component arrays.
     *
     * @param indices  2D [nnz, 2] INT64 array of (row, col) index pairs
     * @param values   1D [nnz] float array of non-zero values
     * @param shape    logical dense shape {rows, cols}
     * @param format   sparse storage format (must be {@link SparseFormat#COO})
     */
    public SparseNDArray(INDArray indices, INDArray values, long[] shape, SparseFormat format) {
        Preconditions.checkNotNull(indices, "indices must not be null");
        Preconditions.checkNotNull(values,  "values must not be null");
        Preconditions.checkNotNull(shape,   "shape must not be null");
        Preconditions.checkNotNull(format,  "format must not be null");
        Preconditions.checkArgument(format == SparseFormat.COO,
                "This constructor is for SparseFormat.COO; got: " + format
                + ". For CSR use SparseNDArray(INDArray values, INDArray colIdx, INDArray rowPtr, long[], SparseFormat).");
        Preconditions.checkArgument(shape.length == 2,
                "COO sparse format requires a 2D logical shape, got rank %d", shape.length);
        Preconditions.checkArgument(indices.length() == values.length() * 2 || indices.rank() == 2,
                "indices must be a 2D array of shape [nnz, 2]");

        this.indices  = indices;
        this.values   = values;
        this.colIdx   = null;
        this.rowPtr   = null;
        this.shape    = shape.clone();
        this.format   = format;
        this.dataType = values.dataType();
        this.blockDim = -1;            // not a BSR instance
    }

    /**
     * Construct a BSR (Block Sparse Row) SparseNDArray from pre-built BSR component arrays.
     *
     * <h4>BSR layout</h4>
     * <ul>
     *   <li>{@code bsrValues}  – 1D [nnzb * blockDim * blockDim], floating dtype —
     *       the non-zero block values in row-major block order</li>
     *   <li>{@code bsrColIdx}  – 1D [nnzb], INT32 — block-column index for each stored block</li>
     *   <li>{@code bsrRowPtr}  – 1D [mb+1], INT32 — block-row pointers,
     *       mb = rows / blockDim</li>
     * </ul>
     *
     * <p>Preconditions:
     * <ul>
     *   <li>{@code rows % blockDim == 0} and {@code cols % blockDim == 0}</li>
     *   <li>{@code bsrRowPtr.length() == rows / blockDim + 1}</li>
     *   <li>{@code bsrValues.length() % (blockDim * blockDim) == 0}</li>
     *   <li>{@code bsrColIdx.length() == bsrValues.length() / (blockDim * blockDim)}</li>
     * </ul>
     *
     * <p>Access BSR component arrays via {@link #getBsrValues()}, {@link #getBsrColIdx()},
     * {@link #getBsrRowPtr()}.  Use {@link #getBlockDim()} to retrieve the block size.
     *
     * @param bsrValues  1D [nnzb * blockDim * blockDim] non-zero block values
     * @param bsrColIdx  1D [nnzb] block-column indices (INT32)
     * @param bsrRowPtr  1D [mb+1] block-row pointers (INT32)
     * @param shape      logical dense shape {rows, cols}; both must be multiples of blockDim
     * @param blockDim   the square block size (must be &gt; 0)
     * @param format     must be {@link SparseFormat#BSR}
     */
    public SparseNDArray(INDArray bsrValues, INDArray bsrColIdx, INDArray bsrRowPtr,
                         long[] shape, int blockDim, SparseFormat format) {
        Preconditions.checkNotNull(bsrValues,  "bsrValues must not be null");
        Preconditions.checkNotNull(bsrColIdx,  "bsrColIdx must not be null");
        Preconditions.checkNotNull(bsrRowPtr,  "bsrRowPtr must not be null");
        Preconditions.checkNotNull(shape,      "shape must not be null");
        Preconditions.checkNotNull(format,     "format must not be null");
        Preconditions.checkArgument(format == SparseFormat.BSR,
                "This constructor is for SparseFormat.BSR; got: " + format);
        Preconditions.checkArgument(blockDim > 0,
                "blockDim must be positive, got %d", blockDim);
        Preconditions.checkArgument(shape.length == 2,
                "BSR format requires a 2D logical shape, got rank %d", shape.length);
        Preconditions.checkArgument(shape[0] % blockDim == 0,
                "rows (%d) must be a multiple of blockDim (%d)", shape[0], blockDim);
        Preconditions.checkArgument(shape[1] % blockDim == 0,
                "cols (%d) must be a multiple of blockDim (%d)", shape[1], blockDim);
        long mb = shape[0] / blockDim;
        Preconditions.checkArgument(bsrRowPtr.length() == mb + 1,
                "bsrRowPtr length must be mb+1=%d (mb=rows/blockDim=%d), got %d",
                mb + 1, mb, bsrRowPtr.length());
        long bd2 = (long) blockDim * blockDim;
        Preconditions.checkArgument(bsrValues.length() % bd2 == 0,
                "bsrValues length (%d) must be a multiple of blockDim*blockDim (%d)",
                bsrValues.length(), bd2);
        long nnzb = bsrValues.length() / bd2;
        Preconditions.checkArgument(bsrColIdx.length() == nnzb,
                "bsrColIdx length (%d) must equal nnzb=%d (bsrValues.length / blockDim^2)",
                bsrColIdx.length(), nnzb);

        this.values   = bsrValues;
        this.colIdx   = bsrColIdx;   // BSR: block-column indices
        this.rowPtr   = bsrRowPtr;   // BSR: block-row pointers
        this.indices  = null;
        this.shape    = shape.clone();
        this.format   = format;
        this.dataType = bsrValues.dataType();
        this.blockDim = blockDim;
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /** Returns the 1D non-zero values array [nnz]. */
    public INDArray getValues() { return values; }

    /**
     * Returns the 1D column-index array [nnz] (INT32). Only valid for CSR format.
     *
     * @throws IllegalStateException if format is not CSR
     */
    public INDArray getColIdx() {
        Preconditions.checkState(format == SparseFormat.CSR,
                "getColIdx() is only valid for CSR format; this is " + format);
        return colIdx;
    }

    /**
     * Returns the 1D row-pointer array [rows+1] (INT32). Only valid for CSR format.
     *
     * @throws IllegalStateException if format is not CSR
     */
    public INDArray getRowPtr() {
        Preconditions.checkState(format == SparseFormat.CSR,
                "getRowPtr() is only valid for CSR format; this is " + format);
        return rowPtr;
    }

    /**
     * Returns the 2D index array [nnz, 2] (INT64). Only valid for COO format.
     *
     * @throws IllegalStateException if format is not COO
     */
    public INDArray getIndices() {
        Preconditions.checkState(format == SparseFormat.COO,
                "getIndices() is only valid for COO format; this is " + format);
        return indices;
    }

    /**
     * Returns the 1D row-index array [nnz] (INT32). Only valid for CSC format.
     *
     * <p>This is the {@code cscRowIdx} array: {@code rowIdx[k]} gives the row of the k-th
     * stored non-zero in column-major order.
     *
     * @throws IllegalStateException if format is not CSC
     */
    public INDArray getRowIdx() {
        Preconditions.checkState(format == SparseFormat.CSC,
                "getRowIdx() is only valid for CSC format; this is " + format);
        return colIdx;   // stored in the colIdx field (see CSC constructor)
    }

    /**
     * Returns the 1D column-pointer array [cols+1] (INT32). Only valid for CSC format.
     *
     * <p>This is the {@code cscColPtr} array: the non-zeros in column {@code c} are at
     * positions {@code colPtr[c]} through {@code colPtr[c+1]-1} in the values/rowIdx arrays.
     *
     * @throws IllegalStateException if format is not CSC
     */
    public INDArray getColPtr() {
        Preconditions.checkState(format == SparseFormat.CSC,
                "getColPtr() is only valid for CSC format; this is " + format);
        return rowPtr;   // stored in the rowPtr field (see CSC constructor)
    }

    /** Returns the logical dense shape {rows, cols}. */
    public long[] getShape() { return shape.clone(); }

    /** Returns the sparse storage format. */
    public SparseFormat getFormat() { return format; }

    /** Returns the data type of the non-zero values. */
    public DataType dataType() { return dataType; }

    /** Returns the number of non-zero elements. */
    public long nnz() { return values.length(); }

    /** Returns the number of rows in the logical dense shape. */
    public long rows() { return shape[0]; }

    /** Returns the number of columns in the logical dense shape. */
    public long cols() { return shape[1]; }

    /**
     * Returns the square block dimension for BSR format.
     *
     * @return the block dimension set at construction time
     * @throws IllegalStateException if this array is not in BSR format
     */
    public int getBlockDim() {
        Preconditions.checkState(format == SparseFormat.BSR,
                "getBlockDim() is only valid for BSR format; this is " + format);
        return blockDim;
    }

    /**
     * Returns the 1D non-zero block values array [nnzb * blockDim * blockDim]. Only valid for BSR format.
     *
     * @throws IllegalStateException if format is not BSR
     */
    public INDArray getBsrValues() {
        Preconditions.checkState(format == SparseFormat.BSR,
                "getBsrValues() is only valid for BSR format; this is " + format);
        return values;
    }

    /**
     * Returns the 1D block-column index array [nnzb] (INT32). Only valid for BSR format.
     *
     * @throws IllegalStateException if format is not BSR
     */
    public INDArray getBsrColIdx() {
        Preconditions.checkState(format == SparseFormat.BSR,
                "getBsrColIdx() is only valid for BSR format; this is " + format);
        return colIdx;
    }

    /**
     * Returns the 1D block-row pointer array [mb+1] (INT32), where mb = rows / blockDim.
     * Only valid for BSR format.
     *
     * @throws IllegalStateException if format is not BSR
     */
    public INDArray getBsrRowPtr() {
        Preconditions.checkState(format == SparseFormat.BSR,
                "getBsrRowPtr() is only valid for BSR format; this is " + format);
        return rowPtr;
    }

    // -----------------------------------------------------------------------
    // Conversion
    // -----------------------------------------------------------------------

    /**
     * Materialises the dense matrix by executing the appropriate sparse-to-dense op.
     *
     * <ul>
     *   <li>CSR: executes the {@code csr_to_dense} op</li>
     *   <li>COO: executes {@code compat_sparse_to_dense} with a shape vector</li>
     * </ul>
     *
     * @return a new dense INDArray of shape [rows, cols]
     */
    public INDArray toDense() {
        if (format == SparseFormat.CSR) {
            CsrToDense op = new CsrToDense(values, colIdx, rowPtr, rows(), cols());
            INDArray[] results = Nd4j.exec(op);
            return results[0];
        } else if (format == SparseFormat.COO) {
            // COO -> dense via the verified COO->CSR->dense path (coo_to_csr + csr_to_dense).
            return toCsr().toDense();
        } else if (format == SparseFormat.CSC) {
            // colIdx holds cscRowIdx; rowPtr holds cscColPtr (see CSC constructor)
            CscToDense op = new CscToDense(values, colIdx, rowPtr, rows(), cols());
            INDArray[] results = Nd4j.exec(op);
            return results[0];
        } else if (format == SparseFormat.BSR) {
            // colIdx holds bsrColIdx; rowPtr holds bsrRowPtr (see BSR constructor)
            BsrToDense op = new BsrToDense(values, colIdx, rowPtr, rows(), cols(), blockDim);
            INDArray[] results = Nd4j.exec(op);
            return results[0];
        } else {
            throw new UnsupportedOperationException("toDense() not supported for format: " + format);
        }
    }

    /**
     * Converts this sparse array to CSR format.
     *
     * <ul>
     *   <li>If already CSR, returns {@code this}.</li>
     *   <li>If COO, runs the {@code coo_to_csr} op and wraps the result.</li>
     * </ul>
     *
     * @return a CSR SparseNDArray equivalent to this array
     */
    public SparseNDArray toCsr() {
        if (format == SparseFormat.CSR) {
            return this;
        } else if (format == SparseFormat.COO) {
            CooToCsr op = new CooToCsr(indices, values, rows(), cols());
            INDArray[] results = Nd4j.exec(op);
            // results[0] = values, results[1] = colIdx, results[2] = rowPtr
            return new SparseNDArray(results[0], results[1], results[2], shape.clone(), SparseFormat.CSR);
        } else {
            throw new UnsupportedOperationException("toCsr() not supported from format: " + format);
        }
    }

    /**
     * Converts this sparse array to CSC (Compressed Sparse Column) format.
     *
     * <ul>
     *   <li>If already CSC, returns {@code this}.</li>
     *   <li>If CSR, runs the {@code csr_to_csc} op and wraps the three result arrays as a
     *       new CSC {@link SparseNDArray} of the same logical shape.</li>
     *   <li>If COO, converts to CSR first ({@link #toCsr()}), then to CSC.</li>
     * </ul>
     *
     * @return a CSC SparseNDArray equivalent to this array
     */
    public SparseNDArray toCsc() {
        if (format == SparseFormat.CSC) {
            return this;
        } else if (format == SparseFormat.CSR) {
            CsrToCsc op = new CsrToCsc(values, colIdx, rowPtr, rows(), cols());
            INDArray[] results = Nd4j.exec(op);
            // results[0] = cscValues, results[1] = cscRowIdx, results[2] = cscColPtr
            return new SparseNDArray(results[0], results[1], results[2], shape.clone(), SparseFormat.CSC);
        } else if (format == SparseFormat.COO) {
            return toCsr().toCsc();
        } else {
            throw new UnsupportedOperationException("toCsc() not supported from format: " + format);
        }
    }

    /**
     * Converts this CSR sparse array to BSR (Block Sparse Row) format using the given block size.
     *
     * <p>Executes the {@code csr_to_bsr} native op, which groups the CSR non-zeros into
     * fixed-size {@code blockDim × blockDim} blocks.  Only blocks that contain at least one
     * non-zero element in the original CSR matrix are stored; blocks that are entirely zero
     * are absent from the BSR representation.
     *
     * <p><b>Requirements:</b>
     * <ul>
     *   <li>This array must be in {@link SparseFormat#CSR} format.</li>
     *   <li>{@code rows() % blockDim == 0} and {@code cols() % blockDim == 0}.</li>
     * </ul>
     *
     * @param bDim square block size (must be &gt; 0; both rows and cols must be exact multiples)
     * @return a new BSR {@link SparseNDArray} equivalent to this CSR array
     * @throws IllegalStateException    if this array is not in CSR format
     * @throws IllegalArgumentException if dims are not multiples of blockDim (also enforced
     *                                  by the native REQUIRE_TRUE, which throws on violation)
     */
    public SparseNDArray toBsr(int bDim) {
        Preconditions.checkState(format == SparseFormat.CSR,
                "toBsr() requires CSR format; call toCsr() first. Current format: " + format);
        Preconditions.checkArgument(bDim > 0, "blockDim must be positive, got %d", bDim);
        Preconditions.checkArgument(shape[0] % bDim == 0,
                "rows (%d) must be a multiple of blockDim (%d)", shape[0], bDim);
        Preconditions.checkArgument(shape[1] % bDim == 0,
                "cols (%d) must be a multiple of blockDim (%d)", shape[1], bDim);

        CsrToBsr op = new CsrToBsr(values, colIdx, rowPtr, rows(), cols(), bDim);
        INDArray[] results = Nd4j.exec(op);
        // results[0] = bsrValues, results[1] = bsrColIdx, results[2] = bsrRowPtr
        return new SparseNDArray(results[0], results[1], results[2], shape.clone(), bDim, SparseFormat.BSR);
    }

    /**
     * Returns the sparse transpose Aᵀ as a CSR {@link SparseNDArray} of shape
     * {@code [cols, rows]}.
     *
     * <p>The implementation exploits the identity <em>CSC(A) ≡ CSR(Aᵀ)</em>: it runs
     * {@code csr_to_csc} once to obtain the CSC of A, then reinterprets the three output
     * arrays as CSR arrays of Aᵀ with the shape dimensions swapped — no additional data
     * movement is needed.
     *
     * <pre>
     *   CSR of Aᵀ:
     *     values  = cscValues  (same nnz non-zero values)
     *     colIdx  = cscRowIdx  (becomes column indices of Aᵀ)
     *     rowPtr  = cscColPtr  (becomes row pointers of Aᵀ, length = cols(A)+1)
     *     shape   = [cols(A), rows(A)]
     * </pre>
     *
     * @return a new CSR SparseNDArray representing Aᵀ
     * @throws IllegalStateException if this array is not in CSR format (call toCsr() first)
     */
    public SparseNDArray transpose() {
        Preconditions.checkState(format == SparseFormat.CSR,
                "transpose() requires CSR format; call toCsr() first. Current format: " + format);
        // csr_to_csc(A) → cscValues, cscRowIdx, cscColPtr
        CsrToCsc op = new CsrToCsc(values, colIdx, rowPtr, rows(), cols());
        INDArray[] csc = Nd4j.exec(op);
        // Reinterpret as CSR of Aᵀ: shape [cols, rows]
        //   csc[0] = cscValues  → values of Aᵀ
        //   csc[1] = cscRowIdx  → colIdx of Aᵀ (column indices)
        //   csc[2] = cscColPtr  → rowPtr of Aᵀ (row pointers, length cols(A)+1 = rows(Aᵀ)+1) ✓
        long[] transposedShape = new long[]{cols(), rows()};
        return new SparseNDArray(csc[0], csc[1], csc[2], transposedShape, SparseFormat.CSR);
    }

    // -----------------------------------------------------------------------
    // Sparse BLAS ergonomics (CSR only)
    // -----------------------------------------------------------------------

    /**
     * Sparse matrix-matrix product: {@code C = A * B}.
     *
     * <p>Executes the {@code csr_spmm} op (A · B, non-transposed) and returns the
     * result as a new dense {@link INDArray} of shape [rows, n].
     *
     * @param B dense matrix [cols, n]
     * @return dense result C [rows, n]
     * @throws IllegalStateException if this array is not in CSR format
     */
    public INDArray mmul(INDArray B) {
        if (format == SparseFormat.BSR) {
            // BSR SpMM: A_bsr · B  →  C [rows, n]
            BsrSpmm op = new BsrSpmm(values, colIdx, rowPtr, B, rows(), cols(), blockDim);
            return Nd4j.exec(op)[0];
        }
        Preconditions.checkState(format == SparseFormat.CSR,
                "mmul() requires CSR or BSR format; call toCsr() or toBsr() first. Current format: " + format);
        CsrSpmm op = new CsrSpmm(values, colIdx, rowPtr, B, rows(), cols(), false);
        return Nd4j.exec(op)[0];
    }

    /**
     * Sparse matrix-matrix product (SpGEMM): {@code C = A · B}, where both A and B are
     * sparse CSR matrices.
     *
     * <p>Executes the {@code csr_spgemm} native op and wraps the three result arrays
     * (cValues, cColIdx, cRowPtr) as a new CSR {@link SparseNDArray} of shape
     * {@code [this.rows(), other.cols()]}.
     *
     * <p><b>Requirements:</b>
     * <ul>
     *   <li>Both {@code this} and {@code other} must be in {@link SparseFormat#CSR} format.</li>
     *   <li>{@code this.cols() == other.rows()} (inner dimension must match).</li>
     * </ul>
     *
     * @param other the right-hand sparse matrix B in CSR format
     * @return a new CSR {@link SparseNDArray} C of logical shape [this.rows(), other.cols()]
     * @throws IllegalStateException     if either matrix is not in CSR format
     * @throws IllegalArgumentException  if the inner dimensions do not match
     */
    public SparseNDArray mmul(SparseNDArray other) {
        Preconditions.checkState(format == SparseFormat.CSR,
                "mmul(SparseNDArray) requires this to be CSR format; call toCsr() first. Current format: " + format);
        Preconditions.checkState(other.getFormat() == SparseFormat.CSR,
                "mmul(SparseNDArray) requires other to be CSR format; call other.toCsr() first. Other format: " + other.getFormat());
        Preconditions.checkArgument(this.cols() == other.rows(),
                "Inner dimensions must match: this.cols()=%d but other.rows()=%d", this.cols(), other.rows());

        long m = this.rows();
        long k = this.cols();   // == other.rows()
        long n = other.cols();

        CsrSpgemm op = new CsrSpgemm(
                this.values,  this.colIdx,  this.rowPtr,
                other.values, other.colIdx, other.rowPtr,
                m, k, n);
        INDArray[] results = Nd4j.exec(op);
        // results[0] = cValues, results[1] = cColIdx, results[2] = cRowPtr
        return new SparseNDArray(results[0], results[1], results[2],
                new long[]{m, n}, SparseFormat.CSR);
    }

    /**
     * Sparse matrix-vector product: {@code y = A * x}.
     *
     * <p>Executes the {@code csr_spmv} op (A · x, non-transposed) and returns the
     * result as a new dense 1D {@link INDArray} of length rows.
     *
     * @param x dense vector [cols]
     * @return dense result y [rows]
     * @throws IllegalStateException if this array is not in CSR format
     */
    public INDArray mv(INDArray x) {
        Preconditions.checkState(format == SparseFormat.CSR,
                "mv() requires CSR format; call toCsr() first. Current format: " + format);
        CsrSpmv op = new CsrSpmv(values, colIdx, rowPtr, x, rows(), cols(), false);
        return Nd4j.exec(op)[0];
    }

    /**
     * Semiring sparse matrix-vector product:
     * {@code out[i] = REDUCE_{k in row i}( SR.add(acc, SR.mul(values[k], x[colIdx[k]])) )}
     * starting from the semiring's additive identity.
     *
     * <table>
     *   <tr><th>Semiring</th><th>mul</th><th>add</th><th>identity</th></tr>
     *   <tr><td>PLUS_TIMES</td><td>&times;</td><td>+</td><td>0</td></tr>
     *   <tr><td>MIN_PLUS</td><td>+</td><td>min</td><td>+&infin;</td></tr>
     *   <tr><td>MAX_PLUS</td><td>+</td><td>max</td><td>-&infin;</td></tr>
     *   <tr><td>OR_AND</td><td>&and;</td><td>&or;</td><td>0</td></tr>
     *   <tr><td>MIN_TIMES</td><td>&times;</td><td>min</td><td>+&infin;</td></tr>
     * </table>
     *
     * @param x        dense vector [cols]
     * @param semiring the algebraic semiring to use for the multiply-reduce
     * @return dense result [rows]
     * @throws IllegalStateException if this array is not in CSR format
     */
    public INDArray mvSemiring(INDArray x, Semiring semiring) {
        Preconditions.checkState(format == SparseFormat.CSR,
                "mvSemiring() requires CSR format; call toCsr() first. Current format: " + format);
        CsrSpmvSemiring op = new CsrSpmvSemiring(values, colIdx, rowPtr, x, rows(), cols(), semiring);
        return Nd4j.exec(op)[0];
    }

    /**
     * Semiring sparse matrix-matrix product:
     * {@code C[i,j] = REDUCE_{k in row i}( SR.add(acc, SR.mul(values[k], B[colIdx[k],j])) )}
     * starting from the semiring's additive identity.
     *
     * <p>This is the batched (multi-vector) extension of {@link #mvSemiring}: each column of
     * {@code B} is processed independently with the same semiring.  For graph algorithms this
     * is useful when propagating multi-dimensional feature vectors (e.g. multi-source BFS
     * or multi-commodity flow distances).
     *
     * @param B        dense matrix [cols, n]
     * @param semiring the algebraic semiring to use for the multiply-reduce
     * @return dense result [rows, n]
     * @throws IllegalStateException if this array is not in CSR format
     */
    public INDArray mmulSemiring(INDArray B, Semiring semiring) {
        Preconditions.checkState(format == SparseFormat.CSR,
                "mmulSemiring() requires CSR format; call toCsr() first. Current format: " + format);
        CsrSpmmSemiring op = new CsrSpmmSemiring(values, colIdx, rowPtr, B, rows(), cols(), semiring);
        return Nd4j.exec(op)[0];
    }

    // -----------------------------------------------------------------------
    // Elementwise operations (CSR only)
    // -----------------------------------------------------------------------

    /**
     * Elementwise sparse matrix addition: {@code C = A + B}.
     *
     * <p>Executes the {@code csr_add} native op and wraps the three result arrays
     * (cValues, cColIdx, cRowPtr) as a new CSR {@link SparseNDArray} with the same
     * logical shape as {@code this}.
     *
     * <p><b>Requirements:</b>
     * <ul>
     *   <li>Both {@code this} and {@code other} must be in {@link SparseFormat#CSR} format.</li>
     *   <li>{@code this.getShape()} must equal {@code other.getShape()} (same [m, n]).</li>
     * </ul>
     *
     * @param other the right-hand sparse matrix B in CSR format
     * @return a new CSR {@link SparseNDArray} C = A + B of the same logical shape
     * @throws IllegalStateException    if either matrix is not in CSR format
     * @throws IllegalArgumentException if the shapes do not match
     */
    public SparseNDArray add(SparseNDArray other) {
        Preconditions.checkState(format == SparseFormat.CSR,
                "add() requires this to be CSR format; call toCsr() first. Current format: " + format);
        Preconditions.checkState(other.getFormat() == SparseFormat.CSR,
                "add() requires other to be CSR format; call other.toCsr() first. Other format: " + other.getFormat());
        Preconditions.checkArgument(java.util.Arrays.equals(this.shape, other.getShape()),
                "Both matrices must have the same shape for add(); this=%s, other=%s",
                java.util.Arrays.toString(this.shape), java.util.Arrays.toString(other.getShape()));

        long m = rows();
        long n = cols();

        CsrAdd op = new CsrAdd(
                this.values,           this.colIdx,           this.rowPtr,
                other.getValues(),     other.getColIdx(),     other.getRowPtr(),
                m, n);
        INDArray[] results = Nd4j.exec(op);
        // results[0] = cValues, results[1] = cColIdx, results[2] = cRowPtr
        return new SparseNDArray(results[0], results[1], results[2],
                new long[]{m, n}, SparseFormat.CSR);
    }

    /**
     * Scalar scale: returns a new CSR {@link SparseNDArray} whose non-zero values are
     * multiplied by {@code s}.  The sparsity structure (colIdx, rowPtr) is unchanged.
     *
     * @param s the scalar multiplier
     * @return a new CSR SparseNDArray with values = this.values * s
     * @throws IllegalStateException if this array is not in CSR format
     */
    public SparseNDArray scale(double s) {
        Preconditions.checkState(format == SparseFormat.CSR,
                "scale() requires CSR format; call toCsr() first. Current format: " + format);
        return new SparseNDArray(values.mul(s), colIdx, rowPtr, shape.clone(), SparseFormat.CSR);
    }

    // -----------------------------------------------------------------------
    // Reductions (CSR only)
    // -----------------------------------------------------------------------

    /**
     * Row sums: returns a 1D dense array of length {@code rows()} where each element is the
     * sum of the corresponding row.
     *
     * <p>Implemented as {@code A · ones[cols]} via {@link CsrSpmv}.
     *
     * @return 1D INDArray of shape [rows] containing per-row sums
     * @throws IllegalStateException if this array is not in CSR format
     */
    public INDArray rowSums() {
        Preconditions.checkState(format == SparseFormat.CSR,
                "rowSums() requires CSR format; call toCsr() first. Current format: " + format);
        INDArray ones = Nd4j.ones(values.dataType(), cols());
        CsrSpmv op = new CsrSpmv(values, colIdx, rowPtr, ones, rows(), cols(), false);
        return Nd4j.exec(op)[0];
    }

    /**
     * Column sums: returns a 1D dense array of length {@code cols()} where each element is
     * the sum of the corresponding column.
     *
     * <p>Implemented as {@code Aᵀ · ones[rows]} via {@link CsrSpmv} with {@code transposeA=true}.
     *
     * @return 1D INDArray of shape [cols] containing per-column sums
     * @throws IllegalStateException if this array is not in CSR format
     */
    public INDArray colSums() {
        Preconditions.checkState(format == SparseFormat.CSR,
                "colSums() requires CSR format; call toCsr() first. Current format: " + format);
        INDArray ones = Nd4j.ones(values.dataType(), rows());
        CsrSpmv op = new CsrSpmv(values, colIdx, rowPtr, ones, rows(), cols(), true);
        return Nd4j.exec(op)[0];
    }

    /**
     * Total sum of all non-zero elements.
     *
     * <p>Because the zero-fill positions contribute zero to the sum, summing only the
     * stored non-zero values is equivalent to summing all elements of the dense matrix.
     *
     * @return the sum of all non-zero values as a {@link Number}
     * @throws IllegalStateException if this array is not in CSR format
     */
    public Number sumNumber() {
        Preconditions.checkState(format == SparseFormat.CSR,
                "sumNumber() requires CSR format; call toCsr() first. Current format: " + format);
        return values.sumNumber();
    }

    // -----------------------------------------------------------------------
    // Graph-preprocessing ergonomics (CSR only)
    // -----------------------------------------------------------------------

    /**
     * Degree vector: per-row sum of all non-zero values.
     *
     * <p>For an unweighted adjacency matrix (all non-zeros are 1), this gives the
     * standard graph degree sequence.  Equivalent to {@link #rowSums()} and provided
     * as a named alias for graph-algorithm readability.
     *
     * @return 1D INDArray of shape [rows] containing per-row sums
     * @throws IllegalStateException if this array is not in CSR format
     */
    public INDArray degree() {
        Preconditions.checkState(format == SparseFormat.CSR,
                "degree() requires CSR format; call toCsr() first. Current format: " + format);
        return rowSums();
    }

    /**
     * Left-diagonal scaling: returns a new CSR {@link SparseNDArray} representing {@code Dl · A}
     * where {@code Dl = diag(dl)}.  Each non-zero entry {@code A[i, j]} is multiplied by
     * {@code dl[i]}.  The sparsity structure is unchanged.
     *
     * <p>Implemented via the {@code csr_diag_mm} native op with {@code dr = ones[cols]}.
     *
     * @param dl 1D [rows] left diagonal vector; must have the same dtype as the values
     * @return a new CSR SparseNDArray with values scaled by {@code dl}
     * @throws IllegalStateException if this array is not in CSR format
     */
    public SparseNDArray scaleRows(INDArray dl) {
        Preconditions.checkState(format == SparseFormat.CSR,
                "scaleRows() requires CSR format; call toCsr() first. Current format: " + format);
        INDArray dr = Nd4j.ones(values.dataType(), cols());
        CsrDiagMm op = new CsrDiagMm(values, colIdx, rowPtr, dl, dr, rows(), cols());
        INDArray[] results = Nd4j.exec(op);
        return new SparseNDArray(results[0], colIdx, rowPtr, shape.clone(), SparseFormat.CSR);
    }

    /**
     * Right-diagonal scaling: returns a new CSR {@link SparseNDArray} representing {@code A · Dr}
     * where {@code Dr = diag(dr)}.  Each non-zero entry {@code A[i, j]} is multiplied by
     * {@code dr[j]}.  The sparsity structure is unchanged.
     *
     * <p>Implemented via the {@code csr_diag_mm} native op with {@code dl = ones[rows]}.
     *
     * @param dr 1D [cols] right diagonal vector; must have the same dtype as the values
     * @return a new CSR SparseNDArray with values scaled by {@code dr}
     * @throws IllegalStateException if this array is not in CSR format
     */
    public SparseNDArray scaleCols(INDArray dr) {
        Preconditions.checkState(format == SparseFormat.CSR,
                "scaleCols() requires CSR format; call toCsr() first. Current format: " + format);
        INDArray dl = Nd4j.ones(values.dataType(), rows());
        CsrDiagMm op = new CsrDiagMm(values, colIdx, rowPtr, dl, dr, rows(), cols());
        INDArray[] results = Nd4j.exec(op);
        return new SparseNDArray(results[0], colIdx, rowPtr, shape.clone(), SparseFormat.CSR);
    }

    /**
     * Add self-loops: returns a new CSR {@link SparseNDArray} representing {@code A + I}
     * where {@code I} is the sparse {@code n × n} identity matrix.
     *
     * <p>The identity is built with {@link Spdiags} (ones on the diagonal) and added via
     * {@link CsrAdd}.  Duplicate entries on the diagonal (if A already has non-zeros there)
     * are summed by {@code csr_add}.
     *
     * <p>The matrix must be square: {@code rows() == cols()}.
     *
     * @return a new CSR SparseNDArray equal to {@code A + I}
     * @throws IllegalStateException    if this array is not in CSR format
     * @throws IllegalArgumentException if the matrix is not square
     */
    public SparseNDArray addSelfLoops() {
        Preconditions.checkState(format == SparseFormat.CSR,
                "addSelfLoops() requires CSR format; call toCsr() first. Current format: " + format);
        long n = rows();
        Preconditions.checkArgument(n == cols(),
                "addSelfLoops() requires a square matrix; got shape [%d, %d]", n, cols());
        // Build sparse n×n identity via spdiags(ones[n], n)
        INDArray diagOnes = Nd4j.ones(values.dataType(), n);
        Spdiags spdiags = new Spdiags(diagOnes, n);
        INDArray[] idResults = Nd4j.exec(spdiags);
        SparseNDArray identity = new SparseNDArray(
                idResults[0], idResults[1], idResults[2], new long[]{n, n}, SparseFormat.CSR);
        return this.add(identity);
    }

    /**
     * Graph Laplacian: {@code L = D - A} where {@code D = diag(rowSums(A))}.
     *
     * <p>Implemented as {@code spdiags(rowSums()).add(this.scale(-1.0))}.
     *
     * <p>The matrix must be square.
     *
     * @return a new CSR SparseNDArray representing the graph Laplacian
     * @throws IllegalStateException    if this array is not in CSR format
     * @throws IllegalArgumentException if the matrix is not square
     */
    public SparseNDArray laplacian() {
        Preconditions.checkState(format == SparseFormat.CSR,
                "laplacian() requires CSR format; call toCsr() first. Current format: " + format);
        long n = rows();
        Preconditions.checkArgument(n == cols(),
                "laplacian() requires a square matrix; got shape [%d, %d]", n, cols());
        // D = diag(rowSums(A))
        INDArray d = rowSums();
        Spdiags spdiags = new Spdiags(d, n);
        INDArray[] dResults = Nd4j.exec(spdiags);
        SparseNDArray D = new SparseNDArray(
                dResults[0], dResults[1], dResults[2], new long[]{n, n}, SparseFormat.CSR);
        // L = D + (-1)*A = D + (-A)
        return D.add(this.scale(-1.0));
    }

    /**
     * Symmetric normalisation (GCN-style): {@code Â = D̃^{-1/2} (A+I) D̃^{-1/2}}
     * where {@code D̃ = diag(rowSums(A+I))}.
     *
     * <p>Algorithm:
     * <ol>
     *   <li>{@code Ã = addSelfLoops()}  — add the identity</li>
     *   <li>{@code d̃ = Ã.rowSums()}    — degree of the self-looped graph</li>
     *   <li>{@code dInv = d̃^{-1/2}}    — with zero-guard: degree-0 nodes get 0</li>
     *   <li>Return {@code Ã.scaleRows(dInv).scaleCols(dInv)}</li>
     * </ol>
     *
     * <p>The matrix must be square.
     *
     * @return a new CSR SparseNDArray representing the symmetrically normalised adjacency
     * @throws IllegalStateException    if this array is not in CSR format
     * @throws IllegalArgumentException if the matrix is not square
     */
    public SparseNDArray normalizeSymmetric() {
        Preconditions.checkState(format == SparseFormat.CSR,
                "normalizeSymmetric() requires CSR format; call toCsr() first. Current format: " + format);
        long n = rows();
        Preconditions.checkArgument(n == cols(),
                "normalizeSymmetric() requires a square matrix; got shape [%d, %d]", n, cols());
        SparseNDArray as = addSelfLoops();           // Ã = A + I
        INDArray d = as.rowSums();                   // D̃ diagonal
        // dInv = d^{-0.5}, guarding zeros: 0^{-0.5} = Inf → replace with 0
        INDArray dInv = Transforms.pow(d, -0.5, true);
        BooleanIndexing.replaceWhere(dInv, 0.0, Conditions.isNan());
        BooleanIndexing.replaceWhere(dInv, 0.0, Conditions.isInfinite());
        // Â = D̃^{-1/2} Ã D̃^{-1/2}
        return as.scaleRows(dInv).scaleCols(dInv);
    }

    /**
     * Row-normalisation: {@code D^{-1} A} where {@code D = diag(rowSums(A))}.
     *
     * <p>Each row of A is divided by its row-sum (out-degree).  Rows with zero sum are
     * left as zeros (guarded).
     *
     * <p>The matrix must be square (rows == cols is NOT enforced here — D^{-1}A is valid
     * for rectangular matrices too — but the result will be rectangular).
     *
     * @return a new CSR SparseNDArray representing the row-normalised matrix
     * @throws IllegalStateException if this array is not in CSR format
     */
    public SparseNDArray normalizeRow() {
        Preconditions.checkState(format == SparseFormat.CSR,
                "normalizeRow() requires CSR format; call toCsr() first. Current format: " + format);
        INDArray d = rowSums();
        // Guard zero-degree rows: replace 0 → 1 before reciprocal, then mask back
        INDArray mask = d.neq(0.0).castTo(d.dataType());
        // Avoid 1/0: temporarily set 0 entries to 1 so rdiv is safe
        INDArray dSafe = d.add(mask.rsub(1.0));   // zeros→1, non-zeros unchanged
        INDArray dInv  = dSafe.rdiv(1.0);         // 1/d (safe)
        dInv.muli(mask);                          // restore zero-degree rows to 0
        return scaleRows(dInv);
    }

    // -----------------------------------------------------------------------
    // GNN primitives (CSR only)
    // -----------------------------------------------------------------------

    /**
     * Per-row softmax over edge attention logits: the GAT edge-softmax primitive.
     *
     * <p>For each row (source node), normalises the stored non-zero values across all
     * neighbour edges in that row using softmax.  This is the core attention-coefficient
     * normalisation step in Graph Attention Networks (GAT, Veličković et al. 2018):
     * <pre>
     *   alpha[k] = exp(values[k]) / sum_{k' in row i} exp(values[k'])
     * </pre>
     * The sparsity structure (colIdx, rowPtr) is unchanged — only the values are replaced
     * by the softmax-normalised attention weights.
     *
     * <p>Executes the {@code csr_row_softmax} native op.
     *
     * @return a new CSR SparseNDArray with per-row-softmax attention values; same
     *         colIdx and rowPtr as this array
     * @throws IllegalStateException if this array is not in CSR format
     */
    public SparseNDArray edgeSoftmax() {
        Preconditions.checkState(format == SparseFormat.CSR,
                "edgeSoftmax() requires CSR format; call toCsr() first. Current format: " + format);
        CsrRowSoftmax op = new CsrRowSoftmax(values, rowPtr, rows());
        INDArray[] results = Nd4j.exec(op);
        // results[0] = alpha[nnz]: per-row-softmax weights; structure unchanged
        return new SparseNDArray(results[0], colIdx, rowPtr, shape.clone(), SparseFormat.CSR);
    }

    /**
     * Max-aggregation over neighbours: the GraphSAGE-max aggregator.
     *
     * <p>For each row {@code i} (target node), gathers the feature vectors of all source
     * neighbours {@code j} identified by the CSR column indices, and returns the
     * element-wise maximum:
     * <pre>
     *   out[i, f] = max_{k: rowPtr[i] &lt;= k &lt; rowPtr[i+1]} nodeFeatures[colIdx[k], f]
     * </pre>
     * This corresponds to the max-aggregator in GraphSAGE (Hamilton et al. 2017), which
     * selects the most salient feature signal from each node's neighbourhood for inductive
     * graph representation learning.
     *
     * <p>Executes the {@code csr_segment_max} native op.
     *
     * @param nodeFeatures 2D [n, f] dense matrix of per-node feature vectors (n = number
     *                     of nodes; f = feature dimension); must have a floating dtype
     * @return a new dense INDArray of shape [rows, f] where rows = this.rows()
     * @throws IllegalStateException if this array is not in CSR format
     */
    public INDArray segmentMax(INDArray nodeFeatures) {
        Preconditions.checkState(format == SparseFormat.CSR,
                "segmentMax() requires CSR format; call toCsr() first. Current format: " + format);
        CsrSegmentMax op = new CsrSegmentMax(colIdx, rowPtr, nodeFeatures, rows());
        return Nd4j.exec(op)[0];
    }

    /**
     * Edge-gather primitive: pull node-feature vectors onto edges.
     *
     * <p>For each edge {@code e} determined by this matrix's CSR {@code colIdx}:
     * <pre>
     *   edgeFeat[e, f] = X[colIdx[e], f]
     * </pre>
     * This is the E-step of MPNN / edge-conditioned GNNs: map source-node embeddings
     * onto edge tensors so that a per-edge message function can operate purely on edges.
     *
     * <p>Executes the {@code csr_edge_gather} native op.
     *
     * @param X  2D [n, F] dense node-feature matrix (n = number of nodes, F = feature dim);
     *           must have a floating dtype
     * @return a new dense INDArray of shape [nnz, F] where nnz = this.nnz()
     * @throws IllegalStateException if this array is not in CSR format
     */
    public INDArray gatherEdges(INDArray X) {
        Preconditions.checkState(format == SparseFormat.CSR,
                "gatherEdges() requires CSR format; call toCsr() first. Current format: " + format);
        CsrEdgeGather op = new CsrEdgeGather(colIdx, X);
        return Nd4j.exec(op)[0];
    }

    /**
     * Segment scatter-reduce: aggregate per-edge messages to per-node outputs.
     *
     * <p>For each target node {@code i} (CSR row) and feature dimension {@code f}:
     * <ul>
     *   <li>mode 0 — SUM:  {@code out[i, f] = Σ_{e in row i} edgeMsg[e, f]}</li>
     *   <li>mode 1 — MEAN: {@code out[i, f] = Σ / degree(i)} (empty row → 0)</li>
     *   <li>mode 2 — MAX:  {@code out[i, f] = max_{e in row i} edgeMsg[e, f]} (empty → 0)</li>
     * </ul>
     * This is the N-step of MPNN: reduce incoming edge messages into a per-node summary.
     *
     * <p>Executes the {@code csr_edge_aggregate} native op.
     *
     * @param edgeMsg  2D [nnz, F] per-edge message vectors (must have a floating dtype)
     * @param mode     0=SUM, 1=MEAN, 2=MAX
     * @return a new dense INDArray of shape [rows, F] where rows = this.rows()
     * @throws IllegalStateException if this array is not in CSR format
     */
    public INDArray aggregateEdges(INDArray edgeMsg, int mode) {
        Preconditions.checkState(format == SparseFormat.CSR,
                "aggregateEdges() requires CSR format; call toCsr() first. Current format: " + format);
        CsrEdgeAggregate op = new CsrEdgeAggregate(rowPtr, edgeMsg, rows(), mode);
        return Nd4j.exec(op)[0];
    }

    @Override
    public String toString() {
        return "SparseNDArray(" + format + ", shape=" + java.util.Arrays.toString(shape)
                + ", nnz=" + nnz() + ", dtype=" + dataType + ")";
    }
}

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

/**
 * Supported sparse tensor storage formats.
 */
public enum SparseFormat {
    /** Compressed Sparse Row — row-major indexed sparse matrix. */
    CSR,
    /** Coordinate format — list of (row, col, value) triples. */
    COO,
    /** Compressed Sparse Column — column-major indexed sparse matrix. */
    CSC,
    /**
     * Block Sparse Row — groups non-zeros into fixed-size square blocks of dimension
     * {@code blockDim × blockDim}.  Only block-rows and block-columns that contain at
     * least one non-zero element are stored.  Both the row count and the column count of
     * the logical dense matrix must be exact multiples of {@code blockDim}.
     *
     * <p>The three BSR component arrays are:
     * <ul>
     *   <li>{@code bsrValues}  – 1D [nnzb * blockDim * blockDim] — the non-zero block
     *       values, each block stored in row-major element order</li>
     *   <li>{@code bsrColIdx}  – 1D [nnzb], INT32 — block-column index for each
     *       stored block (nnzb is data-dependent)</li>
     *   <li>{@code bsrRowPtr}  – 1D [mb+1], INT32 — block-row pointers where
     *       {@code mb = rows / blockDim}</li>
     * </ul>
     */
    BSR
}

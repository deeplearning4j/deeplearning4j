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

package org.nd4j.linalg.api.blas;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.SparseNDArray;

/**
 * Sparse BLAS Level-3 interface: sparse-matrix × dense-matrix operations.
 *
 * <p>Only CSR (Compressed Sparse Row) format is currently supported. Callers
 * holding a COO matrix must first convert it via {@code SparseNDArray.toCsr()}
 * (or equivalent) before invoking these methods.
 *
 * <p>Implementations are backend-agnostic: the underlying {@code csr_spmm} op
 * dispatches to CPU or CUDA natively depending on the active backend.
 *
 * @author nd4j
 */
public interface SparseLevel3 {

    /**
     * Sparse matrix–dense matrix product: {@code C = op(A) · B}.
     *
     * @param a          CSR sparse matrix (must have {@code SparseFormat.CSR})
     * @param b          dense matrix of shape {@code [a.cols(), n]} (or {@code [a.rows(), n]} when transposeA is true)
     * @param transposeA if {@code true} compute {@code A^T · B}
     * @return dense result matrix of shape {@code [a.rows(), n]} (or {@code [a.cols(), n]} when transposeA is true)
     * @throws IllegalArgumentException if {@code a} is not in CSR format
     */
    INDArray spmm(SparseNDArray a, INDArray b, boolean transposeA);

    /**
     * Sparse matrix–dense matrix product with {@code transposeA = false}: {@code C = A · B}.
     *
     * @param a CSR sparse matrix
     * @param b dense matrix of shape {@code [a.cols(), n]}
     * @return dense result matrix of shape {@code [a.rows(), n]}
     * @throws IllegalArgumentException if {@code a} is not in CSR format
     */
    default INDArray spmm(SparseNDArray a, INDArray b) {
        return spmm(a, b, false);
    }
}

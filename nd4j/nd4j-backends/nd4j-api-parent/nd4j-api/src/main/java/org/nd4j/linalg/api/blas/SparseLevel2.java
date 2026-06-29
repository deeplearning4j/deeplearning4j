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
 * Sparse BLAS Level-2 interface: sparse-matrix × dense-vector operations.
 *
 * <p>Only CSR (Compressed Sparse Row) format is currently supported. Callers
 * holding a COO matrix must first convert it via {@code SparseNDArray.toCsr()}
 * (or equivalent) before invoking these methods.
 *
 * <p>Implementations are backend-agnostic: the underlying {@code csr_spmv} op
 * dispatches to CPU or CUDA natively depending on the active backend.
 *
 * @author nd4j
 */
public interface SparseLevel2 {

    /**
     * Sparse matrix–vector product: {@code y = op(A) · x}.
     *
     * @param a          CSR sparse matrix (must have {@code SparseFormat.CSR})
     * @param x          dense column vector of length {@code a.cols()} (or {@code a.rows()} when transposeA is true)
     * @param transposeA if {@code true} compute {@code A^T · x}
     * @return dense result vector of length {@code a.rows()} (or {@code a.cols()} when transposeA is true)
     * @throws IllegalArgumentException if {@code a} is not in CSR format
     */
    INDArray spmv(SparseNDArray a, INDArray x, boolean transposeA);

    /**
     * Sparse matrix–vector product with {@code transposeA = false}: {@code y = A · x}.
     *
     * @param a CSR sparse matrix
     * @param x dense vector of length {@code a.cols()}
     * @return dense result vector of length {@code a.rows()}
     * @throws IllegalArgumentException if {@code a} is not in CSR format
     */
    default INDArray spmv(SparseNDArray a, INDArray x) {
        return spmv(a, x, false);
    }
}

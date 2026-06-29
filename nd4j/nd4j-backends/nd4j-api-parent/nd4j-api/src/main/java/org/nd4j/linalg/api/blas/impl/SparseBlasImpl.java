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

package org.nd4j.linalg.api.blas.impl;

import org.nd4j.linalg.api.blas.SparseLevel2;
import org.nd4j.linalg.api.blas.SparseLevel3;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.SparseFormat;
import org.nd4j.linalg.api.ndarray.SparseNDArray;
import org.nd4j.linalg.api.ops.impl.sparse.CsrSpmm;
import org.nd4j.linalg.api.ops.impl.sparse.CsrSpmv;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Backend-agnostic implementation of {@link SparseLevel2} and {@link SparseLevel3}.
 *
 * <p>Both {@code spmv} and {@code spmm} delegate to the existing
 * {@code csr_spmv} / {@code csr_spmm} {@link org.nd4j.linalg.api.ops.DynamicCustomOp}
 * classes, which in turn dispatch to the active native backend (CPU or CUDA) via
 * the standard op-execution pipeline.  No per-backend subclassing is needed.
 *
 * <p>Only CSR format is supported.  Callers holding COO tensors should call
 * {@code toCsr()} (or construct a {@link SparseNDArray} directly in CSR form)
 * before using this class.
 *
 * <p>This class is a stateless singleton; retrieve it via
 * {@link org.nd4j.linalg.factory.BaseBlasWrapper#sparseLevel2()} /
 * {@link org.nd4j.linalg.factory.BaseBlasWrapper#sparseLevel3()}.
 *
 * @author nd4j
 */
public class SparseBlasImpl implements SparseLevel2, SparseLevel3 {

    /** Singleton — all methods are stateless. */
    public static final SparseBlasImpl INSTANCE = new SparseBlasImpl();

    private SparseBlasImpl() {}

    // -----------------------------------------------------------------------
    // SparseLevel2
    // -----------------------------------------------------------------------

    /**
     * {@inheritDoc}
     *
     * <p>Builds a {@link CsrSpmv} op from the {@link SparseNDArray} components
     * and executes it via {@link Nd4j#exec(org.nd4j.linalg.api.ops.CustomOp)}.
     */
    @Override
    public INDArray spmv(SparseNDArray a, INDArray x, boolean transposeA) {
        requireCsr(a);
        CsrSpmv op = new CsrSpmv(
                a.getValues(),
                a.getColIdx(),
                a.getRowPtr(),
                x,
                a.rows(),
                a.cols(),
                transposeA);
        return Nd4j.exec(op)[0];
    }

    // -----------------------------------------------------------------------
    // SparseLevel3
    // -----------------------------------------------------------------------

    /**
     * {@inheritDoc}
     *
     * <p>Builds a {@link CsrSpmm} op from the {@link SparseNDArray} components
     * and executes it via {@link Nd4j#exec(org.nd4j.linalg.api.ops.CustomOp)}.
     */
    @Override
    public INDArray spmm(SparseNDArray a, INDArray b, boolean transposeA) {
        requireCsr(a);
        CsrSpmm op = new CsrSpmm(
                a.getValues(),
                a.getColIdx(),
                a.getRowPtr(),
                b,
                a.rows(),
                a.cols(),
                transposeA);
        return Nd4j.exec(op)[0];
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    private static void requireCsr(SparseNDArray a) {
        if (a.getFormat() != SparseFormat.CSR) {
            throw new IllegalArgumentException(
                    "SparseBlasImpl requires CSR format; got " + a.getFormat()
                    + ". Convert via toCsr() first.");
        }
    }
}

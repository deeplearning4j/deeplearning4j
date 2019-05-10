/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.api.blas.impl;

import org.nd4j.linalg.api.blas.Level3;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Audrey Loeffel
 */
public class SparseBaseLevel3 extends SparseBaseLevel implements Level3 {
    @Override
    public void gemm(char Order, char TransA, char TransB, double alpha, INDArray A, INDArray B, double beta,
                    INDArray C) {

    }

    @Override
    public void gemm(INDArray A, INDArray B, INDArray C, boolean transposeA, boolean transposeB, double alpha,
                    double beta) {

    }

    @Override
    public void symm(char Order, char Side, char Uplo, double alpha, INDArray A, INDArray B, double beta, INDArray C) {

    }

    @Override
    public void syrk(char Order, char Uplo, char Trans, double alpha, INDArray A, double beta, INDArray C) {

    }

    @Override
    public void syr2k(char Order, char Uplo, char Trans, double alpha, INDArray A, INDArray B, double beta,
                    INDArray C) {

    }

    @Override
    public void trmm(char Order, char Side, char Uplo, char TransA, char Diag, double alpha, INDArray A, INDArray B,
                    INDArray C) {

    }

    @Override
    public void trsm(char Order, char Side, char Uplo, char TransA, char Diag, double alpha, INDArray A, INDArray B) {

    }
}

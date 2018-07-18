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

import org.nd4j.linalg.api.blas.Lapack;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Audrey Loeffel
 */
public class SparseBaseLapack implements Lapack {
    @Override
    public INDArray getrf(INDArray A) {
        return null;
    }

    @Override
    public INDArray getPFactor(int M, INDArray ipiv) {
        return null;
    }

    @Override
    public INDArray getLFactor(INDArray A) {
        return null;
    }

    @Override
    public INDArray getUFactor(INDArray A) {
        return null;
    }

    @Override
    public void getri(int N, INDArray A, int lda, int[] IPIV, INDArray WORK, int lwork, int INFO) {

    }

    @Override
    public void geqrf(INDArray A, INDArray R) {

    }

    @Override
    public void potrf(INDArray A, boolean lower) {

    }

    @Override
    public int syev(char jobz, char uplo, INDArray A, INDArray V) {
        return 0;
    }

    @Override
    public void gesvd(INDArray A, INDArray S, INDArray U, INDArray VT) {

    }
}

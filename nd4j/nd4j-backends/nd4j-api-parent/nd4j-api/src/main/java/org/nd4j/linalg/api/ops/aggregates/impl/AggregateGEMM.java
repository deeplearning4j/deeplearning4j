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

package org.nd4j.linalg.api.ops.aggregates.impl;

import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.aggregates.BaseAggregate;

/**
 *
 * PLEASE NOTE: This op is available for CPU only, and should NOT be ever called manually, unless you know why you're using it
 *
 * @author raver119@gmail.com
 */
public class AggregateGEMM extends BaseAggregate {

    public AggregateGEMM() {
        // no-op
    }

    public AggregateGEMM(int Order, int TransA, int TransB, int M, int N, int K, double alpha, @NonNull INDArray A,
                    int lda, @NonNull INDArray B, int ldb, double beta, @NonNull INDArray C, int ldc) {
        this.arguments.add(A);
        this.arguments.add(B);
        this.arguments.add(C);

        this.indexingArguments.add(M);
        this.indexingArguments.add(N);
        this.indexingArguments.add(K);
        this.indexingArguments.add(lda);
        this.indexingArguments.add(ldb);
        this.indexingArguments.add(ldc);
        this.indexingArguments.add(TransA);
        this.indexingArguments.add(TransB);
        this.indexingArguments.add(Order);

        this.realArguments.add(alpha);
        this.realArguments.add(beta);
    }

    @Override
    public String name() {
        return "aggregate_gemm";
    }

    @Override
    public int opNum() {
        return 5;
    }

    @Override
    public int maxArguments() {
        return 3;
    }

    @Override
    public int maxShapes() {
        return 0;
    }

    @Override
    public int maxIntArrays() {
        return 0;
    }

    @Override
    public int maxIntArraySize() {
        return 0;
    }

    @Override
    public int maxIndexArguments() {
        return 9;
    }

    @Override
    public int maxRealArguments() {
        return 2;
    }

    @Override
    public int getSharedMemorySize() {
        return 0;
    }

    @Override
    public int getThreadsPerInstance() {
        return 0;
    }
}

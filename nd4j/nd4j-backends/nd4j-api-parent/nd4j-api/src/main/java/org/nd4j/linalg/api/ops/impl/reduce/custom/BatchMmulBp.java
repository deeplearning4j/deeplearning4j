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
package org.nd4j.linalg.api.ops.impl.reduce.custom;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

public class BatchMmulBp extends DynamicCustomOp {
    protected int transposeA;
    protected int transposeB;

    protected int batchSize;

    protected int M;
    protected int N;
    protected int K;

    protected int lda,ldb,ldc;
    public BatchMmulBp(SameDiff sameDiff, SDVariable[] matricesA, SDVariable[] matricesB,SDVariable[] eps, boolean transposeA, boolean transposeB) {
        this(sameDiff, ArrayUtil.concat(SDVariable.class,matricesA, matricesB,eps), transposeA, transposeB);
    }

    public BatchMmulBp(SameDiff sameDiff,
                       SDVariable[] matrices,
                       boolean transposeA,
                       boolean transposeB) {
        super(null, sameDiff,
                ArrayUtil.concat(SDVariable.class,
                        new SDVariable[]{
                                sameDiff.var(Nd4j.ones(matrices[0].dataType(), matrices.length / 2)), // alphas
                                sameDiff.var(Nd4j.zeros(matrices[1].dataType(), matrices.length / 2))
                        }, // betas
                        matrices));
        this.transposeA = transposeA ? 1 : 0;
        this.transposeB = transposeB ? 1 : 0;
        this.batchSize = matrices.length / 2;

    }



    public BatchMmulBp(SameDiff sd,
                       SDVariable alphas,
                       SDVariable betas,
                       SDVariable[] inputsA,
                       SDVariable[] inputsB,
                       SDVariable[] eps,
                       boolean transposeA,
                       boolean transposeB) {
        super(sd, ArrayUtil.concat(SDVariable.class,
                new SDVariable[]{alphas,betas},
                inputsA,inputsB,eps
        ));

        this.transposeA = transposeA ? 1 : 0;
        this.transposeB = transposeB ? 1 : 0;
    }

    public BatchMmulBp(INDArray alphas,
                       INDArray betas,
                       INDArray[] inputsA,
                       INDArray[] inputsB,
                       INDArray[] eps,
                       boolean transposeA,
                       boolean transposeB) {
        super(ArrayUtil.concat(
                INDArray.class,
                new INDArray[]{alphas,betas},
                inputsA,inputsB,eps
        ),null);
        this.batchSize = inputsA.length;

        this.transposeA = transposeA ? 1 : 0;
        this.transposeB = transposeB ? 1 : 0;

        long[] firstShape = inputsA[0].shape();
        long[] lastShape = inputsB[0].shape();

        this.M = transposeA ? (int) firstShape[1]: (int) firstShape[0];
        this.N = transposeB ? (int) lastShape[0]: (int) lastShape[1];
        this.K = transposeB ? (int) lastShape[1]: (int) lastShape[0];
        this.lda = (int) firstShape[0];
        this.ldb = (int) lastShape[0];
        this.ldc = (int) firstShape[0];
        addArgs();
    }

    @Override
    public void configureWithSameDiff(SameDiff sameDiff) {
        super.configureWithSameDiff(sameDiff);
        SDVariable[] matrices = args();
        Preconditions.checkState(matrices.length % 2 == 0, "The number of provided matrices needs" +
                "to be divisible by two.");
        this.batchSize = (matrices.length - 2) / 2;

        SDVariable firstMatrix = matrices[2];
        long[] firstShape = firstMatrix.getShape();

        SDVariable lastMatrix = matrices[matrices.length - 1];
        long[] lastShape = lastMatrix.getShape();
        /**/

        if(firstShape != null) {
            this.M = transposeA > 0 ? (int) firstShape[1]: (int) firstShape[0];
            this.lda = (int) firstShape[0];
        }

        if(lastShape != null) {
            this.N = transposeB > 0? (int) lastShape[0]: (int) lastShape[1];
            this.K = transposeB > 0 ? (int) lastShape[1]: (int) lastShape[0];
            this.ldb = (int) lastShape[0];
            this.ldc = this.M;
        }



        //only add arguments when fully initialized
        if(M > 0 && N > 0 && K > 0 && firstShape != null && lastShape != null) {
            addArgs();

        }
    }

    @Override
    public int getNumOutputs() {
        return batchSize;
    }

    public void addArgs() {
        if(iArguments.isEmpty())
            addIArgument(transposeA, transposeB,
                    M, N, K, // K and N are swapped in libnd4j
                    lda,ldb,ldc, // these three are LDA, LDB and LDC (leading dims / strides) from blas. set to matrix dims here
                    batchSize);
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        List<DataType> out = new ArrayList<>();
        for(int i = 0; i < batchSize; i++) {  //-2 for the alpha and beta params
            Preconditions.checkState(dataTypes.get(i).isFPType(), "Inputs to batch mmul op must all be a floating point type: got %s", dataTypes);
            if(i % 2 == 0) {
                out.add(dataTypes.get(i));
            }
        }

        return out;
    }

    public BatchMmulBp() {
    }

    @Override
    public String opName() {
        return "batched_gemm_bp";
    }
}

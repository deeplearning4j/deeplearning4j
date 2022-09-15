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

import lombok.EqualsAndHashCode;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

/**
 * Batched matrix multiplication.
 *
 * Matrix multiply a batch of matrices. First and second batch of matrices have to be arrays of same
 * length and each pair taken from these sets has to have dimensions (M, N) and (N, K),
 * respectively. The result of this operation will be a batch of multiplied matrices. The
 * result has the same length as both input batches and each output matrix is of shape (M, K).
 *
 * @author Max Pumperla
 */
@EqualsAndHashCode
public class BatchMmul extends DynamicCustomOp {

    protected int transposeA;
    protected int transposeB;

    protected int batchSize;

    protected int M;
    protected int N;
    protected int K;

    public BatchMmul(SameDiff sameDiff, SDVariable[] matricesA, SDVariable[] matricesB, boolean transposeA, boolean transposeB) {
        this(sameDiff, ArrayUtils.addAll(matricesA, matricesB), transposeA, transposeB);
    }

    public BatchMmul(SameDiff sameDiff,
                     SDVariable[] matrices,
                     boolean transposeA,
                     boolean transposeB) {
        super(null, sameDiff, ArrayUtils.addAll(
                new SDVariable[]{
                        sameDiff.var(Nd4j.ones(matrices[0].dataType(), matrices.length / 2)), // alphas
                        sameDiff.var(Nd4j.zeros(matrices[1].dataType(), matrices.length / 2))}, // betas
                matrices));
        this.transposeA = transposeA ? 1 : 0;
        this.transposeB = transposeB ? 1 : 0;
        this.batchSize = matrices.length / 2;

    }



    public BatchMmul(SameDiff sd, SDVariable alphas, SDVariable betas, SDVariable[] inputsA, SDVariable[] inputsB, boolean transposeA, boolean transposeB) {
        super(sd, ArrayUtil.concat(SDVariable.class,
                new SDVariable[]{alphas,betas},
                inputsA,inputsB
        ));

    }

    public BatchMmul(INDArray alphas, INDArray betas, INDArray[] inputsA, INDArray[] inputsB, boolean transposeA, boolean transposeB) {
        super(ArrayUtil.concat(
                INDArray.class,
                new INDArray[]{alphas,betas},
                inputsA,inputsB
        ),null);
        this.batchSize = inputsA.length;

        this.transposeA = transposeA ? 1 : 0;
        this.transposeB = transposeB ? 1 : 0;

        long[] firstShape = inputsA[0].shape();
        long[] lastShape = inputsB[0].shape();

        this.M = transposeA ? (int) firstShape[1]: (int) firstShape[0];
        this.N = transposeB ? (int) lastShape[0]: (int) lastShape[1];
        this.K = transposeB ? (int) lastShape[1]: (int) lastShape[0];
        addArgs();
    }

    @Override
    public void configureWithSameDiff(SameDiff sameDiff) {
        super.configureWithSameDiff(sameDiff);
        SDVariable[] matrices = args();
        Preconditions.checkState(matrices.length % 2 == 0, "The number of provided matrices needs" +
                "to be divisible by two.");
        this.batchSize = (matrices.length - 2)/ 2;

        SDVariable firstMatrix = matrices[2];
        long[] firstShape = firstMatrix.getShape();

        SDVariable lastMatrix = matrices[matrices.length - 1];
        long[] lastShape = lastMatrix.getShape();
        /**/

        if(firstShape != null) {
            this.M = transposeA > 0 ? (int) firstShape[1]: (int) firstShape[0];
        }

        if(lastShape != null) {
            this.N = transposeB > 0? (int) lastShape[0]: (int) lastShape[1];
            this.K = transposeB > 0 ? (int) lastShape[1]: (int) lastShape[0];
        }


        //only add arguments when fully initialized
        if(M > 0 && N > 0 && K > 0 && firstShape != null && lastShape != null)
            addArgs();
    }

    @Override
    public int getNumOutputs() {
        return batchSize;
    }

    public void addArgs() {
        if(iArguments.isEmpty())
            addIArgument(transposeA, transposeB,
                    M, N, K, // K and N are swapped in libnd4j
                    M, N, M, // these three are LDA, LDB and LDC (leading dims / strides) from blas. set to matrix dims here
                    batchSize);
    }


    public BatchMmul() {
    }

    @Override
    public String opName() {
        return "batched_gemm";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> grads) {
        SDVariable[] dLdOut = grads.toArray(new SDVariable[grads.size()]);

        SDVariable[] allArgs = args();
        SDVariable[] matricesA = Arrays.copyOfRange(allArgs,2, batchSize);
        SDVariable[] matricesB = Arrays.copyOfRange(allArgs, batchSize, 2 * batchSize);

        SDVariable[] dLdx = sameDiff.batchMmul(allArgs[0],allArgs[1],dLdOut, matricesB, false, transposeB == 1);
        SDVariable[] dLdy = sameDiff.batchMmul(allArgs[0],allArgs[1],matricesA, dLdOut, transposeA == 1, false);

        List<SDVariable> ret = new ArrayList<>();
        Collections.addAll(ret, dLdx);
        Collections.addAll(ret, dLdy);
        return ret;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        List<DataType> out = new ArrayList<>();
        for(int i = 0; i < dataTypes.size() - 2; i++ ) {  //-2 for the alpha and beta params
            Preconditions.checkState(dataTypes.get(i).isFPType(), "Inputs to batch mmul op must all be a floating point type: got %s", dataTypes);
            if(i % 2 == 0) {
                out.add(dataTypes.get(i));
            }
        }

        return out;
    }

    @Override
    public boolean needsConfigure() {
        return true;
    }

}


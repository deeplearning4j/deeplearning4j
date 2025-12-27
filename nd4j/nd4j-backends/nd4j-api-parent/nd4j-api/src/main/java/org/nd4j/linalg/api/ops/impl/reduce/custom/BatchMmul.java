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
    private SDVariable[] matricesA,matricesB;
    private SDVariable alphas,betas;
    
    public BatchMmul(SameDiff sameDiff,
                     SDVariable[] matricesA,
                     SDVariable[] matricesB,
                     boolean transposeA,
                     boolean transposeB) {
        this(sameDiff, ArrayUtils.addAll(matricesA, matricesB), transposeA, transposeB);
        this.matricesA = matricesA;
        this.matricesB = matricesB;
        this.alphas = sameDiff.constant(Nd4j.scalar(matricesA[0].dataType(),1.0));
        this.betas = sameDiff.constant(Nd4j.scalar(matricesB[0].dataType(),0.0));
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
        this.alphas = arg(0);
        this.betas = arg(1);
        this.matricesA = new SDVariable[batchSize];
        this.matricesB = new SDVariable[batchSize];
        for(int i = 0 ; i < batchSize; i++) {
            matricesA[i] = arg(i + 2);
            matricesB[i] = arg(i + 2 + batchSize);
        }
        // Only add transpose flags - dimensions will be inferred at runtime
        addIArgument(this.transposeA, this.transposeB);
    }

    public BatchMmul(SameDiff sd, SDVariable alphas,
                     SDVariable betas,
                     SDVariable[] inputsA,
                     SDVariable[] inputsB, boolean transposeA, boolean transposeB) {
        super(sd, ArrayUtil.concat(SDVariable.class,
                new SDVariable[]{alphas,betas},
                inputsA,inputsB
        ));

        this.batchSize = inputsA.length;
        this.transposeA = transposeA ? 1 : 0;
        this.transposeB = transposeB ? 1 : 0;
        this.alphas = alphas;
        this.betas = betas;
        this.matricesA = inputsA;
        this.matricesB = inputsB;
        
        // Only add transpose flags - dimensions will be inferred at runtime
        addIArgument(this.transposeA, this.transposeB);
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
        
        // Only add transpose flags - dimensions will be inferred at runtime
        addIArgument(this.transposeA, this.transposeB);
    }

    @Override
    public void configureWithSameDiff(SameDiff sameDiff) {
        super.configureWithSameDiff(sameDiff);
        SDVariable[] matrices = args();
        Preconditions.checkState(matrices.length >= 2, "BatchMmul requires at least 2 arguments (alphas and betas)");
        Preconditions.checkState((matrices.length - 2) % 2 == 0, "The number of provided matrices needs" +
                "to be divisible by two (after alphas and betas).");
        this.batchSize = (matrices.length - 2) / 2;

        if(batchSize > 0) {
            this.alphas = arg(0);
            this.betas = arg(1);
            this.matricesA = new SDVariable[batchSize];
            this.matricesB = new SDVariable[batchSize];
            for(int i = 0 ; i < batchSize; i++) {
                matricesA[i] = arg(i + 2);
                matricesB[i] = arg(i + 2 + batchSize);
            }
        }
        
        // Ensure IArgs are set (only transpose flags needed)
        if(iArguments.isEmpty()) {
            addIArgument(transposeA, transposeB);
        }
    }

    @Override
    public int getNumOutputs() {
        return batchSize;
    }

    public BatchMmul() {
    }

    @Override
    public String opName() {
        return "batched_gemm";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grads) {
        SDVariable[] eps = grads.toArray(new SDVariable[0]);
        return new BatchMmulBp(sameDiff,
                alphas,
                betas,
                matricesA,
                matricesB,
                eps,
                transposeA == 1,
                transposeB == 1).outputs();
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        List<DataType> out = new ArrayList<>();
        for(int i = 0; i < batchSize; i++) {
            // Use the datatype of the first matrix (alpha) or first input matrix
            DataType dt = dataTypes.size() > 2 ? dataTypes.get(2) : dataTypes.get(0);
            Preconditions.checkState(dt.isFPType(), "Inputs to batch mmul op must all be a floating point type: got %s", dt);
            out.add(dt);
        }
        return out;
    }

    @Override
    public boolean needsConfigure() {
        return false;  // We don't need special configuration since dimensions are inferred at runtime
    }
}
/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.api.ops.impl.accum;

import lombok.EqualsAndHashCode;
import lombok.val;
import onnx.OnnxProto3;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

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


    public BatchMmul(SameDiff sameDiff,
                     SDVariable[] matrices,
                     boolean transposeA,
                     boolean transposeB) {
        super(null, sameDiff, ArrayUtils.addAll(
                new SDVariable[]{
                        sameDiff.var(Nd4j.ones(matrices.length / 2)), // alphas
                        sameDiff.var(Nd4j.zeros(matrices.length / 2))}, // betas
                matrices));

        Preconditions.checkState(matrices.length % 2 == 0, "The number of provided matrices needs" +
                "to be divisible by two.");
        this.batchSize = matrices.length / 2;

        SDVariable firstMatrix = matrices[0];
        long[] firstShape = firstMatrix.getShape();
        for (int i = 0; i < batchSize; i++) {
            Preconditions.checkState(Arrays.equals(firstShape, matrices[i].getShape()));
        }
        SDVariable lastMatrix = matrices[2 * batchSize - 1];
        long[] lastShape = lastMatrix.getShape();
        for (int i = batchSize; i < 2 * batchSize; i++) {
            Preconditions.checkState(Arrays.equals(lastShape, matrices[i].getShape()));
        }

        this.transposeA = transposeA ? 1 : 0;
        this.transposeB = transposeB ? 1 : 0;

        this.M = transposeA ? (int) firstShape[1]: (int) firstShape[0];
        this.N = transposeA ? (int) firstShape[0]: (int) firstShape[1];
        this.K = transposeB ? (int) lastShape[0]: (int) lastShape[1];

        addArgs();
    }

    @Override
    public int getNumOutputs(){
        return batchSize;
    }

    public void addArgs() {
        addIArgument(transposeA, transposeB,
                M, K, N, // K and N are swapped in libnd4j
                M, K, N, // these three are LDA, LDB and LDC (leading dims / strides) from blas. set to matrix dims here
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
        SDVariable[] matricesA = Arrays.copyOfRange(allArgs,0, batchSize);
        SDVariable[] matricesB = Arrays.copyOfRange(allArgs, batchSize, 2 * batchSize);

        SDVariable[] dLdx = sameDiff.batchMmul(dLdOut, matricesB, false, transposeB == 1);
        SDVariable[] dLdy = sameDiff.batchMmul(matricesA, dLdOut, transposeA == 1, false);

        List<SDVariable> ret = new ArrayList<>();
        Collections.addAll(ret, dLdx);
        Collections.addAll(ret, dLdy);
        return ret;
    }
}


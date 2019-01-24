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

package org.nd4j.linalg.api.ops.impl.nlp;

import lombok.NonNull;
import lombok.val;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

public class CbowRound extends DynamicCustomOp {

    /**
     * hs round
     *
     * @param target
     * @param context
     * @param syn0
     * @param syn1
     * @param expTable
     * @param alpha
     * @param nextRandom
     * @param inferenceVector
     */
    public CbowRound(int target, @NonNull int[] context, @NonNull INDArray syn0, @NonNull INDArray syn1, @NonNull INDArray expTable, @NonNull int[] indices, @NonNull byte[] codes, double alpha, long nextRandom, @NonNull INDArray inferenceVector) {
        this(target, context, 0, syn0, syn1, Nd4j.empty(syn1.dataType()), expTable, Nd4j.empty(syn1.dataType()), indices, codes, 0, alpha, nextRandom, inferenceVector);
    }

    /**
     * ns round
     *
     * @param target
     * @param context
     * @param ngStarter
     * @param syn0
     * @param syn1Neg
     * @param expTable
     * @param negTable
     * @param alpha
     * @param nextRandom
     * @param inferenceVector
     */
    public CbowRound(int target, @NonNull int[] context, int ngStarter, @NonNull INDArray syn0, @NonNull INDArray syn1Neg, @NonNull INDArray expTable, @NonNull INDArray negTable, int nsRounds, double alpha, long nextRandom, @NonNull INDArray inferenceVector) {
        this(target, context, ngStarter, syn0, Nd4j.empty(syn0.dataType()), syn1Neg, expTable, negTable, null, null, nsRounds, alpha, nextRandom, inferenceVector);
    }

    /**
     * full constructor
     *
     * @param target
     * @param context
     * @param ngStarter
     * @param syn0
     * @param syn1
     * @param syn1Neg
     * @param expTable
     * @param negTable
     * @param alpha
     * @param nextRandom
     * @param inferenceVector
     */
    public CbowRound(int target, @NonNull int[] context, int ngStarter, @NonNull INDArray syn0, @NonNull INDArray syn1, @NonNull INDArray syn1Neg, @NonNull INDArray expTable, @NonNull INDArray negTable, int[] indices, byte[] codes, int nsRounds, double alpha, long nextRandom, @NonNull INDArray inferenceVector) {
        if (indices != null)
            Preconditions.checkArgument(indices.length == codes.length, "Indices length should be equal to codes length");

        val ctx = Nd4j.createFromArray(context);
        val idx = indices == null ? Nd4j.empty(DataType.INT) : Nd4j.createFromArray(indices);
        val code = codes == null ? Nd4j.empty(DataType.BYTE) : Nd4j.createFromArray(codes);
        val lr = Nd4j.scalar(alpha);
        val nr = Nd4j.scalar(nextRandom);

        inputArguments.add(Nd4j.scalar(target));
        inputArguments.add(Nd4j.scalar(ngStarter));
        inputArguments.add(ctx);
        inputArguments.add(idx);
        inputArguments.add(code);
        inputArguments.add(syn0);
        inputArguments.add(syn1);
        inputArguments.add(syn1Neg);
        inputArguments.add(expTable);
        inputArguments.add(negTable);
        inputArguments.add(lr);
        inputArguments.add(nr);
        inputArguments.add(inferenceVector);

        // couple of options
        iArguments.add((long) 0);
        iArguments.add((long) nsRounds);

        bArguments.add(true);
        bArguments.add(!inferenceVector.isEmpty());

        // this op is always inplace
        setInPlace(true);
        setInplaceCall(true);

        for (val in:inputArguments)
            outputArguments.add(in);
    }

    @Override
    public String opName() {
        return "cbow";
    }
}

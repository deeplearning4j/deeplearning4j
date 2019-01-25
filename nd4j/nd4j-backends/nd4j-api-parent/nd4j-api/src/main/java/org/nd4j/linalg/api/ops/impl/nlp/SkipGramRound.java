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

public class SkipGramRound extends DynamicCustomOp {
    @Override
    public String opName() {
        return "skipgram";
    }

    /**
     * sg hs round
     *
     * @param target
     * @param syn0
     * @param syn1
     * @param expTable
     * @param indices
     * @param codes
     * @param alpha
     * @param randomValue
     */
    public SkipGramRound(int target, @NonNull INDArray syn0, @NonNull INDArray syn1, @NonNull INDArray expTable, int[] indices, byte[] codes, double alpha, long randomValue, INDArray inferenceVector) {
        this(target, 0, syn0, syn1, Nd4j.empty(syn1.dataType()), expTable, Nd4j.empty(syn1.dataType()), 0, indices, codes, alpha, randomValue, inferenceVector);
    }

    /**
     * sg ns round
     *
     * @param target
     * @param ngStarter
     * @param syn0
     * @param syn1Neg
     * @param expTable
     * @param negTable
     */
    public SkipGramRound(int target, int ngStarter, @NonNull INDArray syn0, @NonNull INDArray syn1Neg, @NonNull INDArray expTable, @NonNull INDArray negTable, int nsRounds, double alpha, long randomValue, INDArray inferenceVector) {
        this(target, ngStarter, syn0, Nd4j.empty(syn0.dataType()), syn1Neg, expTable, negTable, nsRounds, null, null, alpha, randomValue, inferenceVector);
    }

    /**
     * full constructor
     */
    public SkipGramRound(int target, int ngStarter, @NonNull INDArray syn0, @NonNull INDArray syn1, @NonNull INDArray syn1Neg, @NonNull INDArray expTable, @NonNull INDArray negTable, int nsRounds, int[] indices, byte[] codes, double alpha, long randomValue, INDArray inferenceVector) {
        if (indices != null)
            Preconditions.checkArgument(indices.length == codes.length, "Indices length should be equal to codes length");

        val idx = indices == null ? Nd4j.empty(DataType.INT) : Nd4j.createFromArray(indices);
        val code = codes == null ? Nd4j.empty(DataType.BYTE) : Nd4j.createFromArray(codes);
        val lr = Nd4j.scalar(alpha);
        val nr = Nd4j.scalar(randomValue);

        inputArguments.add(Nd4j.createFromArray(target));
        inputArguments.add(Nd4j.createFromArray(ngStarter));
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
        iArguments.add((long) nsRounds);
        bArguments.add(!inferenceVector.isEmpty());

        // this op is always inplace
        setInPlace(true);
        setInplaceCall(true);

        for (val in:inputArguments)
            outputArguments.add(in);
    }
}

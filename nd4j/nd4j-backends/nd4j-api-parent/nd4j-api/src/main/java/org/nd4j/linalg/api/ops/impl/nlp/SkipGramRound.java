/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops.impl.nlp;

import lombok.NonNull;
import lombok.val;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

public class SkipGramRound extends DynamicCustomOp {
    @Override
    public String opName() {
        return "skipgram";
    }

    public SkipGramRound(){ }

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
        this(Nd4j.scalar(target), Nd4j.scalar(-1), syn0, syn1, Nd4j.empty(syn1.dataType()), expTable, Nd4j.empty(syn1.dataType()), 0, Nd4j.createFromArray(indices), Nd4j.createFromArray(codes), Nd4j.scalar(alpha), Nd4j.scalar(randomValue), inferenceVector, false, 1);
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
        this(Nd4j.scalar(target), Nd4j.scalar(ngStarter), syn0, Nd4j.empty(syn0.dataType()), syn1Neg, expTable, negTable, nsRounds, Nd4j.empty(DataType.INT), Nd4j.empty(DataType.BYTE), Nd4j.scalar(alpha), Nd4j.scalar(randomValue), inferenceVector, false, 1);
    }

    /**
     * full constructor
     */
    public SkipGramRound(@NonNull INDArray target, @NonNull INDArray ngStarter, @NonNull INDArray syn0, @NonNull INDArray syn1, @NonNull INDArray syn1Neg, @NonNull INDArray expTable, @NonNull INDArray negTable, int nsRounds, @NonNull INDArray indices, @NonNull INDArray codes, @NonNull INDArray alpha, @NonNull INDArray randomValue, INDArray inferenceVector, boolean preciseMode, int numWorkers) {
//        if (indices != null)
//            Preconditions.checkArgument(indices.length == codes.length, "Indices length should be equal to codes length");

//        val idx = indices == null ? Nd4j.empty(DataType.INT) : Nd4j.createFromArray(indices);
//        val code = codes == null ? Nd4j.empty(DataType.BYTE) : Nd4j.createFromArray(codes);

        inputArguments.add(target);
        inputArguments.add(ngStarter);
        inputArguments.add(indices);
        inputArguments.add(codes);
        inputArguments.add(syn0);
        inputArguments.add(syn1);
        inputArguments.add(syn1Neg);
        inputArguments.add(expTable);
        inputArguments.add(negTable);
        inputArguments.add(alpha);
        inputArguments.add(randomValue);
        inputArguments.add(inferenceVector);

        // temporary arrays for neu1e
        //inputArguments.add(Nd4j.create(syn0.dataType(), new long[]{target.isVector() ? target.size(0) : 1, syn0.columns()}));

        // couple of options
        iArguments.add((long) numWorkers);
        iArguments.add((long) nsRounds);

        bArguments.add(!inferenceVector.isEmpty());
        bArguments.add(preciseMode);

        // this op is always inplace
        setInPlace(true);
        setInplaceCall(true);

        for (val in:inputArguments)
            outputArguments.add(in);
    }
}

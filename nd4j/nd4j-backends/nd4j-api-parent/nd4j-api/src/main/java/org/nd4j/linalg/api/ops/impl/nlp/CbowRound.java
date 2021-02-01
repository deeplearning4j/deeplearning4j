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

public class CbowRound extends DynamicCustomOp {

    public CbowRound(){ }

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
    public CbowRound(int target, @NonNull int[] context, @NonNull int[] lockedWords, @NonNull INDArray syn0, @NonNull INDArray syn1, @NonNull INDArray expTable, @NonNull int[] indices, @NonNull byte[] codes, double alpha, long nextRandom, @NonNull INDArray inferenceVector, int numLabels) {
        this(Nd4j.scalar(target), Nd4j.createFromArray(context), Nd4j.createFromArray(lockedWords), Nd4j.empty(DataType.INT), syn0, syn1, Nd4j.empty(syn1.dataType()), expTable, Nd4j.empty(syn1.dataType()), Nd4j.createFromArray(indices), Nd4j.createFromArray(codes), 0, Nd4j.scalar(alpha), Nd4j.scalar(nextRandom), inferenceVector, Nd4j.scalar(numLabels), inferenceVector.isEmpty(), 1);
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
    public CbowRound(int target, @NonNull int[] context, @NonNull int[] lockedWords, int ngStarter, @NonNull INDArray syn0, @NonNull INDArray syn1Neg, @NonNull INDArray expTable, @NonNull INDArray negTable, int nsRounds, double alpha, long nextRandom, @NonNull INDArray inferenceVector, int numLabels) {
        this(Nd4j.scalar(target), Nd4j.createFromArray(context), Nd4j.createFromArray(lockedWords), Nd4j.scalar(ngStarter), syn0, Nd4j.empty(syn0.dataType()), syn1Neg, expTable, negTable, Nd4j.empty(DataType.INT), Nd4j.empty(DataType.BYTE), nsRounds, Nd4j.scalar(alpha), Nd4j.scalar(nextRandom), inferenceVector, Nd4j.scalar(numLabels), inferenceVector.isEmpty(), 1);
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
    public CbowRound(@NonNull INDArray target, @NonNull INDArray context, @NonNull INDArray lockedWords, @NonNull INDArray ngStarter, @NonNull INDArray syn0, @NonNull INDArray syn1, @NonNull INDArray syn1Neg, @NonNull INDArray expTable, @NonNull INDArray negTable, @NonNull INDArray indices, @NonNull INDArray codes, int nsRounds, @NonNull INDArray alpha, @NonNull INDArray nextRandom, @NonNull INDArray inferenceVector, @NonNull INDArray numLabels, boolean trainWords, int numWorkers) {

        inputArguments.add(target);
        inputArguments.add(ngStarter);
        inputArguments.add(context);
        inputArguments.add(indices);
        inputArguments.add(codes);
        inputArguments.add(syn0);
        inputArguments.add(syn1);
        inputArguments.add(syn1Neg);
        inputArguments.add(expTable);
        inputArguments.add(negTable);
        inputArguments.add(alpha);
        inputArguments.add(nextRandom);
        inputArguments.add(numLabels);
        inputArguments.add(lockedWords);
        inputArguments.add(inferenceVector);

        // couple of options
        iArguments.add((long) numWorkers);
        iArguments.add((long) nsRounds);


        bArguments.add(trainWords);
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

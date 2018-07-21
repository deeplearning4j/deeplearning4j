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
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.aggregates.BaseAggregate;
import org.nd4j.linalg.factory.Nd4j;

/**
 * This aggregate encapsulates AggregateSkipGram training round for a given word and context
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class AggregateSkipGram extends BaseAggregate {
    private int vectorLength;

    public AggregateSkipGram(INDArray syn0, INDArray syn1, INDArray syn1Neg, INDArray expTable, INDArray negTable,
                    int idxSyn0, int[] idxSyn1, int[] codes, int negativeRounds, int ngStarter, int vectorLength,
                    double alpha, long nextRandom, int vocabSize, INDArray inferenceVector) {
        this(syn0, syn1, syn1Neg, expTable, negTable, idxSyn0, idxSyn1, codes, negativeRounds, ngStarter, vectorLength,
                        alpha, nextRandom, vocabSize);

        arguments.set(5, inferenceVector);

        indexingArguments.set(8, inferenceVector == null ? 0 : 1); // set isInference to true
    }

    public AggregateSkipGram(@NonNull INDArray syn0, INDArray syn1, INDArray syn1Neg, @NonNull INDArray expTable,
                    INDArray negTable, int idxSyn0, int[] idxSyn1, int[] codes, int negativeRounds, int ngStarter,
                    int vectorLength, double alpha, long nextRandom, int vocabSize) {
        indexingArguments.add(idxSyn0);
        indexingArguments.add(vectorLength);
        indexingArguments.add(idxSyn1.length);
        indexingArguments.add(negativeRounds);

        // FIXME: int cast
        indexingArguments.add((int) expTable.length());
        indexingArguments.add(vocabSize);
        indexingArguments.add(ngStarter);

        indexingArguments.add(negTable == null ? 0 : (int) negTable.length());
        indexingArguments.add(0);

        arguments.add(syn0);
        arguments.add(syn1);
        arguments.add(expTable);
        arguments.add(syn1Neg);
        arguments.add(negTable);
        arguments.add(null);

        intArrayArguments.add(idxSyn1);
        intArrayArguments.add(codes);

        realArguments.add(alpha);
        realArguments.add((double) nextRandom);

        this.vectorLength = vectorLength;
    }

    /**
     * This is special signature suitable for use with VoidParameterServer, never ever use it outside of spark-nlp
     *
     * @param w1
     * @param w2
     * @param lr
     * @param vectorLength
     */
    // TODO: probably this signature should be removed?
    public AggregateSkipGram(int w1, int w2, int[] codes, int[] points, int negSamples, double lr, int vectorLength) {
        indexingArguments.add(w1);
        indexingArguments.add(w2);
        indexingArguments.add(vectorLength);

        intArrayArguments.add(codes);
        intArrayArguments.add(points);

        realArguments.add(lr);
    }


    /**
     * This method returns amount of shared memory required for this specific Aggregate.
     * PLEASE NOTE: this method is especially important for CUDA backend. On CPU backend it might be ignored, depending on Aggregate.
     *
     * @return
     */
    @Override
    public int getSharedMemorySize() {
        return (vectorLength * Nd4j.sizeOfDataType()) + 512;
    }

    /**
     * This method returns desired number of threads per Aggregate instance
     * PLEASE NOTE: this method is especially important for CUDA backend. On CPU backend it might be ignored, depending on Aggregate.
     *
     * @return
     */
    @Override
    public int getThreadsPerInstance() {
        if (vectorLength > 768)
            return 768;

        return vectorLength;
    }

    @Override
    public String name() {
        return "aggregate_skipgram";
    }

    @Override
    public int opNum() {
        return 3;
    }

    @Override
    public int maxArguments() {
        return 6;
    }

    @Override
    public int maxShapes() {
        return 0;
    }

    @Override
    public int maxIntArrays() {
        return 2;
    }

    @Override
    public int maxIntArraySize() {
        // we hardcode 40 here, due to w2v codeLength mechanics
        // TODO: make sure this limitation doesn't bother with spark environment
        return 40;
    }

    @Override
    public int maxIndexArguments() {
        return 10;
    }

    @Override
    public int maxRealArguments() {
        return 2;
    }
}

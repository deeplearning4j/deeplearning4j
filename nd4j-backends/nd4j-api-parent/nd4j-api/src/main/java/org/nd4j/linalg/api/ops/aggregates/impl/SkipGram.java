package org.nd4j.linalg.api.ops.aggregates.impl;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.aggregates.BaseAggregate;
import org.nd4j.linalg.factory.Nd4j;

/**
 * This aggregate encapsulates SkipGram training round for a given word and context
 *
 * @author raver119@gmail.com
 */
public class SkipGram extends BaseAggregate {
    private int vectorLength;

    public SkipGram(INDArray syn0, INDArray syn1, INDArray syn1Neg, INDArray expTable, INDArray negTable, int idxSyn0, int[] idxSyn1, int[] codes, int negativeRounds, int ngStarter, int vectorLength, double alpha, long nextRandom, int vocabSize) {
        indexingArguments.add(idxSyn0);
        indexingArguments.add(vectorLength);
        indexingArguments.add(idxSyn1.length);
        indexingArguments.add(negativeRounds);
        indexingArguments.add(expTable.length());
        indexingArguments.add(vocabSize);
        indexingArguments.add(ngStarter);
        indexingArguments.add(negTable == null ? 0 : negTable.length());

        arguments.add(syn0);
        arguments.add(syn1);
        arguments.add(expTable);
        arguments.add(syn1Neg);
        arguments.add(negTable);

        intArrayArguments.add(idxSyn1);
        intArrayArguments.add(codes);

        realArguments.add(alpha);
        realArguments.add((double) nextRandom);

        this.vectorLength = vectorLength;
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
        return 5;
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
        return 40;
    }

    @Override
    public int maxIndexArguments() {
        return 8;
    }

    @Override
    public int maxRealArguments() {
        return 2;
    }
}

package org.nd4j.linalg.api.ops.aggregates.impl;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.aggregates.BaseAggregate;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author raver119@gmail.com
 */
public class AggregateCBOW extends BaseAggregate {
    private int vectorLength;

    public AggregateCBOW(INDArray syn0, INDArray syn1, INDArray syn1Neg, INDArray expTable, INDArray negTable, int wordIdx, int[] idxSyn0, int[] idxSyn1, int[] codes, int negativeRounds, int ngStarter, int vectorLength, double alpha, long nextRandom, int vocabSize) {
        indexingArguments.add(vectorLength);
        indexingArguments.add(idxSyn1.length);
        indexingArguments.add(negativeRounds);
        indexingArguments.add(expTable.length());
        indexingArguments.add(vocabSize);
        indexingArguments.add(ngStarter);
        indexingArguments.add(negTable == null ? 0 : negTable.length());
        indexingArguments.add(idxSyn0.length);
        indexingArguments.add(wordIdx);

        arguments.add(syn0);
        arguments.add(syn1);
        arguments.add(expTable);
        arguments.add(syn1Neg);
        arguments.add(negTable);

        intArrayArguments.add(idxSyn0);
        intArrayArguments.add(idxSyn1);
        intArrayArguments.add(codes);

        realArguments.add(alpha);
        realArguments.add((double) nextRandom);

        this.vectorLength = vectorLength;
    }

    @Override
    public String name() {
        return "aggregate_cbow";
    }

    @Override
    public int opNum() {
        return 4;
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
        return 3;
    }

    @Override
    public int maxIntArraySize() {
        return 40;
    }

    @Override
    public int maxIndexArguments() {
        return 9;
    }

    @Override
    public int maxRealArguments() {
        return 2;
    }

    @Override
    public int getSharedMemorySize() {
        return (vectorLength * Nd4j.sizeOfDataType() * 2) + 512;
    }

    @Override
    public int getThreadsPerInstance() {
        if (vectorLength > 768)
            return 768;

        return vectorLength;
    }
}

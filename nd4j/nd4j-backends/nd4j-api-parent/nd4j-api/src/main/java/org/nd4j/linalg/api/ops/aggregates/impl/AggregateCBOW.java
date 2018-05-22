package org.nd4j.linalg.api.ops.aggregates.impl;

import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.aggregates.BaseAggregate;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author raver119@gmail.com
 */
public class AggregateCBOW extends BaseAggregate {
    private int vectorLength;

    /**
     * Optional constructor for ParagraphVectors PV-DM implementation
     *
     * @param syn0
     * @param syn1
     * @param syn1Neg
     * @param expTable
     * @param negTable
     * @param wordIdx
     * @param idxSyn0
     * @param idxSyn1
     * @param codes
     * @param negativeRounds
     * @param ngStarter
     * @param vectorLength
     * @param alpha
     * @param nextRandom
     * @param vocabSize
     * @param numLabels
     * @param trainWords
     */
    public AggregateCBOW(@NonNull INDArray syn0, INDArray syn1, INDArray syn1Neg, @NonNull INDArray expTable,
                    INDArray negTable, int wordIdx, int[] idxSyn0, int[] idxSyn1, int[] codes, int negativeRounds,
                    int ngStarter, int vectorLength, double alpha, long nextRandom, int vocabSize, int numLabels,
                    boolean trainWords, INDArray inferenceVector) {
        this(syn0, syn1, syn1Neg, expTable, negTable, wordIdx, idxSyn0, idxSyn1, codes, negativeRounds, ngStarter,
                        vectorLength, alpha, nextRandom, vocabSize);

        indexingArguments.set(9, numLabels);
        indexingArguments.set(10, trainWords ? 1 : 0);
        indexingArguments.set(11, inferenceVector == null ? 0 : 1); // set inference to true

        arguments.set(5, inferenceVector);
    }

    /**
     * Default constructor for CBOW implementation wrapper
     * @param syn0
     * @param syn1
     * @param syn1Neg
     * @param expTable
     * @param negTable
     * @param wordIdx
     * @param idxSyn0
     * @param idxSyn1
     * @param codes
     * @param negativeRounds
     * @param ngStarter
     * @param vectorLength
     * @param alpha
     * @param nextRandom
     * @param vocabSize
     */
    public AggregateCBOW(@NonNull INDArray syn0, INDArray syn1, INDArray syn1Neg, @NonNull INDArray expTable,
                    INDArray negTable, int wordIdx, int[] idxSyn0, int[] idxSyn1, int[] codes, int negativeRounds,
                    int ngStarter, int vectorLength, double alpha, long nextRandom, int vocabSize) {
        indexingArguments.add(vectorLength);
        indexingArguments.add(idxSyn1.length);
        indexingArguments.add(negativeRounds);

        // FIXME: int cast
        indexingArguments.add((int) expTable.length());
        indexingArguments.add(vocabSize);
        indexingArguments.add(ngStarter);
        indexingArguments.add(negTable == null ? 0 : (int) negTable.length());
        indexingArguments.add(idxSyn0.length);
        indexingArguments.add(wordIdx);
        indexingArguments.add(0); // number of labels. 0 by default
        indexingArguments.add(1); // trainWords? true by default
        indexingArguments.add(0); // is inference? false by default


        arguments.add(syn0);
        arguments.add(syn1);
        arguments.add(expTable);
        arguments.add(syn1Neg);
        arguments.add(negTable);
        arguments.add(null);

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
        return 6;
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
        return 12;
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

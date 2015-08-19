package org.deeplearning4j.spark.text.accumulators;

import org.apache.spark.AccumulatorParam;
import org.deeplearning4j.berkeley.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author jeffreytang
 */
public class Syn0Accumulator implements AccumulatorParam< Pair<Integer, INDArray> > {

    private int vectorLength;
    private int vocabSize;

    public Syn0Accumulator(int vectorLength, int vocabSize) {
        this.vectorLength = vectorLength;
        this.vocabSize = vocabSize;
    }

    @Override
    public Pair<Integer, INDArray> addInPlace(Pair<Integer, INDArray> pair1, Pair<Integer, INDArray> pair2) {
        Integer randNum = pair1.getFirst();
        INDArray syn0 = pair1.getSecond();
        int nthRow = pair2.getFirst();
        INDArray syn0VecUpdate = pair2.getSecond();
        syn0.getRow(nthRow).addi(syn0VecUpdate);
        return new Pair<>(randNum, syn0);
    }

    @Override
    public Pair<Integer, INDArray> zero(Pair<Integer, INDArray> pair) {
        return new Pair<>(0, Nd4j.zeros(vocabSize, vectorLength));
    }

    @Override
    public Pair<Integer, INDArray> addAccumulator(Pair<Integer, INDArray> pair1,
                                                      Pair<Integer, INDArray> pair2) {
        if (pair1 == null) {
            return new Pair<>(0, Nd4j.zeros(vocabSize, vectorLength));
        }
        addInPlace(pair1, pair2);
        return pair1;
    }

}

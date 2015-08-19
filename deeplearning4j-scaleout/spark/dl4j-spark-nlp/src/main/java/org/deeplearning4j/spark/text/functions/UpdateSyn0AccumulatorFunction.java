package org.deeplearning4j.spark.text.functions;

import org.apache.spark.Accumulator;
import org.apache.spark.api.java.function.VoidFunction;
import org.deeplearning4j.berkeley.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author jeffreytang
 */
public class UpdateSyn0AccumulatorFunction implements VoidFunction< Pair<Integer, INDArray> > {

    private Accumulator<Pair<Integer, INDArray>> syn0Acc;

    public UpdateSyn0AccumulatorFunction(Accumulator<Pair<Integer, INDArray>> syn0Acc) {
        this.syn0Acc = syn0Acc;
    }

    @Override
    public void call(Pair<Integer, INDArray> pair) {
        syn0Acc.add(pair);
    }
}

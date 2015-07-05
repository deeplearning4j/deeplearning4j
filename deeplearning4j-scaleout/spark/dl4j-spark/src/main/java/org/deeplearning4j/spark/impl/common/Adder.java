package org.deeplearning4j.spark.impl.common;

import org.apache.spark.api.java.function.VoidFunction;
import org.nd4j.linalg.api.ndarray.INDArray;


/**
 * @author Adam Gibson
 */
public class Adder implements VoidFunction<INDArray> {
    private SumAccum accumulator;

    public Adder(int length) {
        accumulator = new SumAccum(length);
    }

    @Override
    public void call(INDArray indArrayIterator) throws Exception {
        accumulator.add(indArrayIterator);
    }

    public SumAccum getAccumulator() {
        return accumulator;
    }

    public void setAccumulator(SumAccum accumulator) {
        this.accumulator = accumulator;
    }
}

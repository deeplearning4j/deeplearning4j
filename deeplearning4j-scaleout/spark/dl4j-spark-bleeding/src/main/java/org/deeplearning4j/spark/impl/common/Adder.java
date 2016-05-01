package org.deeplearning4j.spark.impl.common;

import org.apache.spark.api.java.function.VoidFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * @author Adam Gibson
 */
public class Adder implements VoidFunction<INDArray> {
    private SumAccum accumulator;
    private static Logger log = LoggerFactory.getLogger(Adder.class);

    public Adder(int length) {
        accumulator = new SumAccum(length);
    }

    @Override
    public void call(INDArray indArrayIterator) throws Exception {
        log.info("Invoking add operation ");
        accumulator.add(indArrayIterator);
        log.info("Invoked add operation ");

    }

    public SumAccum getAccumulator() {
        return accumulator;
    }

    public void setAccumulator(SumAccum accumulator) {
        this.accumulator = accumulator;
    }
}

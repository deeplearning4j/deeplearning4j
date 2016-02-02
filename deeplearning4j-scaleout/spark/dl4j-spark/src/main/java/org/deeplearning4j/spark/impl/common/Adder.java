package org.deeplearning4j.spark.impl.common;

import org.apache.spark.Accumulator;
import org.apache.spark.api.java.function.VoidFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * @author Adam Gibson
 */
public class Adder implements VoidFunction<INDArray> {
    private SumAccum accumulator;
    private Accumulator<Integer> counter;
    private static Logger log = LoggerFactory.getLogger(Adder.class);

    public Adder(int length, Accumulator<Integer> counter) {
        accumulator = new SumAccum(length);
        this.counter = counter;
    }

    @Override
    public void call(INDArray indArrayIterator) throws Exception {
        if(indArrayIterator != null){
            accumulator.add(indArrayIterator);
            counter.add(1);
        }
        log.info("Invoked add operation ");
    }

    public SumAccum getAccumulator() {
        return accumulator;
    }

    public void setAccumulator(SumAccum accumulator) {
        this.accumulator = accumulator;
    }

    public Accumulator<Integer> getCounter(){
        return counter;
    }

    public void setCounter(Accumulator<Integer> counter){
        this.counter = counter;
    }
}

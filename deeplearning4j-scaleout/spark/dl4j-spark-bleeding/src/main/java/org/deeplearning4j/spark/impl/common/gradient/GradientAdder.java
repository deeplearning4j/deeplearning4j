package org.deeplearning4j.spark.impl.common.gradient;

import org.apache.spark.api.java.function.VoidFunction;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.spark.impl.common.SumAccum;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by agibsonccc on 9/27/15.
 */
public class GradientAdder implements VoidFunction<Gradient> {
    private SumAccum accumulator;
    private static Logger log = LoggerFactory.getLogger(GradientAdder.class);

    public GradientAdder(int length) {
        accumulator = new SumAccum(length);
    }

    @Override
    public void call(Gradient indArrayIterator) throws Exception {
        log.info("Invoking add operation ");
        if(indArrayIterator != null && indArrayIterator.gradient() != null)
            accumulator.add(indArrayIterator.gradient());
        log.info("Invoked add operation ");

    }

    public SumAccum getAccumulator() {
        return accumulator;
    }

    public void setAccumulator(SumAccum accumulator) {
        this.accumulator = accumulator;
    }
}

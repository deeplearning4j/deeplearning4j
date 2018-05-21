package org.deeplearning4j.spark.models.sequencevectors.functions;

import org.apache.spark.AccumulatorParam;
import org.nd4j.linalg.primitives.Counter;

/**
 * Accumulator for elements count
 *
 * @author raver119@gmail.com
 */
public class ElementsFrequenciesAccumulator implements AccumulatorParam<Counter<Long>> {
    @Override
    public Counter<Long> addAccumulator(Counter<Long> c1, Counter<Long> c2) {
        if (c1 == null) {
            return new Counter<>();
        }
        addInPlace(c1, c2);
        return c1;
    }

    @Override
    public Counter<Long> addInPlace(Counter<Long> r1, Counter<Long> r2) {
        r1.incrementAll(r2);
        return r1;
    }

    @Override
    public Counter<Long> zero(Counter<Long> initialValue) {
        return new Counter<>();
    }
}

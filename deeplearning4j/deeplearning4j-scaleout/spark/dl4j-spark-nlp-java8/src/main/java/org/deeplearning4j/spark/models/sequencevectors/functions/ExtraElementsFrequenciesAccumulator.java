package org.deeplearning4j.spark.models.sequencevectors.functions;

import org.apache.spark.AccumulatorParam;
import org.deeplearning4j.spark.models.sequencevectors.primitives.ExtraCounter;

/**
 * Accumulator for elements count
 *
 * @author raver119@gmail.com
 */
public class ExtraElementsFrequenciesAccumulator implements AccumulatorParam<ExtraCounter<Long>> {
    @Override
    public ExtraCounter<Long> addAccumulator(ExtraCounter<Long> c1, ExtraCounter<Long> c2) {
        if (c1 == null) {
            return new ExtraCounter<>();
        }
        addInPlace(c1, c2);
        return c1;
    }

    @Override
    public ExtraCounter<Long> addInPlace(ExtraCounter<Long> r1, ExtraCounter<Long> r2) {
        r1.incrementAll(r2);
        return r1;
    }

    @Override
    public ExtraCounter<Long> zero(ExtraCounter<Long> initialValue) {
        return new ExtraCounter<>();
    }
}

package org.deeplearning4j.spark.text.accumulators;

import org.apache.spark.AccumulatorParam;
import org.nd4j.linalg.primitives.Counter;

/**
 * @author jeffreytang
 */
public class WordFreqAccumulator implements AccumulatorParam<Counter<String>> {

    @Override
    public Counter<String> addInPlace(Counter<String> c1, Counter<String> c2) {
        c1.incrementAll(c2);
        return c1;
    }

    @Override
    public Counter<String> zero(Counter<String> initialCounter) {
        return new Counter<>();
    }

    @Override
    public Counter<String> addAccumulator(Counter<String> c1, Counter<String> c2) {
        if (c1 == null) {
            return new Counter<>();
        }
        addInPlace(c1, c2);
        return c1;
    }
}

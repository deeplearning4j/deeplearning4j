package org.deeplearning4j.spark.text.functions;

import org.apache.spark.api.java.function.Function;
import org.nd4j.linalg.primitives.Pair;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author jeffreytang
 */
public class GetSentenceCountFunction implements Function<Pair<List<String>, AtomicLong>, AtomicLong> {

    @Override
    public AtomicLong call(Pair<List<String>, AtomicLong> pair) throws Exception {
        return pair.getSecond();
    }
}

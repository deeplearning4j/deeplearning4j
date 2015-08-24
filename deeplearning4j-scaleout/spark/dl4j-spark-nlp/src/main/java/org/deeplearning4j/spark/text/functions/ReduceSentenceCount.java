package org.deeplearning4j.spark.text.functions;

import org.apache.spark.api.java.function.Function2;

import java.util.concurrent.atomic.AtomicLong;

/**
 * @author jeffreytang
 */
public class ReduceSentenceCount implements Function2<AtomicLong, AtomicLong, AtomicLong> {
    public AtomicLong call(AtomicLong a, AtomicLong b) {
        return new AtomicLong(a.get() + b.get());
    }
}

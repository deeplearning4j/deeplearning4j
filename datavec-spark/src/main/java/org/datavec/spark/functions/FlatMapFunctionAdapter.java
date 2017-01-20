package org.datavec.spark.functions;

import java.io.Serializable;

/**
 *
 * A function that returns zero or more output records from each input record.
 *
 * Adapter for Spark interface in order to freeze interface changes between spark versions
 */
public interface FlatMapFunctionAdapter<T, R> extends Serializable {
    Iterable<R> call(T t) throws Exception;
}

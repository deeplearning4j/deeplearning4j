package org.deeplearning4j.spark.util;

import java.io.Serializable;

/**
 *
 * A function that returns zero or more records of type Double from each input record.
 *
 * Adapter for Spark interface in order to freeze interface changes between spark versions
 */
public interface DoubleFlatMapFunctionAdapter<T> extends Serializable {
    Iterable<Double> call(T t) throws Exception;
}

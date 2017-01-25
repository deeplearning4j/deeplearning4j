package org.deeplearning4j.spark.util;

import scala.Tuple2;

import java.io.Serializable;

/**
 *
 * A function that returns zero or more key-value pair records from each input record. The
 * key-value pairs are represented as scala.Tuple2 objects.
 *
 * Adapter for Spark interface in order to freeze interface changes between spark versions
 */
public interface PairFlatMapFunctionAdapter <T, K, V> extends Serializable {
    Iterable<Tuple2<K, V>> call(T t) throws Exception;
}

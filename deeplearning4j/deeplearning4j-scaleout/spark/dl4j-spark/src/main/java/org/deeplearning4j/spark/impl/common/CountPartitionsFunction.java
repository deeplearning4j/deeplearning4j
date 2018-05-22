package org.deeplearning4j.spark.impl.common;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function2;
import org.deeplearning4j.spark.api.Repartition;
import scala.Tuple2;

import java.util.Collections;
import java.util.Iterator;

/**
 * This is a function use to count the number of elements in each partition.
 * It is used as part of {@link org.deeplearning4j.spark.util.SparkUtils#repartitionBalanceIfRequired(JavaRDD, Repartition, int, int)}
 *
 * @author Alex Black
 */
public class CountPartitionsFunction<T> implements Function2<Integer, Iterator<T>, Iterator<Tuple2<Integer, Integer>>> {
    @Override
    public Iterator<Tuple2<Integer, Integer>> call(Integer v1, Iterator<T> v2) throws Exception {

        int count = 0;
        while (v2.hasNext()) {
            v2.next();
            count++;
        }

        return Collections.singletonList(new Tuple2<>(v1, count)).iterator();
    }
}

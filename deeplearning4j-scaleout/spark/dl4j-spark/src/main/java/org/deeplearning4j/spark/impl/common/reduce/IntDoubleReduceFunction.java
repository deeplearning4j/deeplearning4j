package org.deeplearning4j.spark.impl.common.reduce;

import org.apache.spark.api.java.function.Function2;
import scala.Tuple2;

/**
 * Add both elements of a {@code Tuple2<Integer,Double>}
 */
public class IntDoubleReduceFunction
                implements Function2<Tuple2<Integer, Double>, Tuple2<Integer, Double>, Tuple2<Integer, Double>> {
    @Override
    public Tuple2<Integer, Double> call(Tuple2<Integer, Double> f, Tuple2<Integer, Double> s) throws Exception {
        return new Tuple2<>(f._1() + s._1(), f._2() + s._2());
    }
}

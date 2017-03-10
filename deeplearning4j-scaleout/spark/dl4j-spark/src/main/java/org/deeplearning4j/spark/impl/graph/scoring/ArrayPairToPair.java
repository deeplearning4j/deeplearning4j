package org.deeplearning4j.spark.impl.graph.scoring;

import org.apache.spark.api.java.function.PairFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import scala.Tuple2;

/**
 * Simple conversion function for SparkComputationGraph
 *
 * @author Alex Black
 */
public class ArrayPairToPair<K> implements PairFunction<Tuple2<K, INDArray[]>, K, INDArray> {
    @Override
    public Tuple2<K, INDArray> call(Tuple2<K, INDArray[]> v1) throws Exception {
        INDArray arr = (v1._2() == null ? null : v1._2()[0]);
        return new Tuple2<>(v1._1(), arr);
    }
}

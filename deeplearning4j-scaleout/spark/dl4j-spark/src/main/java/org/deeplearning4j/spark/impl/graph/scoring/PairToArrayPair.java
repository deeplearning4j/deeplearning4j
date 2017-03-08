package org.deeplearning4j.spark.impl.graph.scoring;

import org.apache.spark.api.java.function.PairFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import scala.Tuple2;

/**
 * Simple conversion function for SparkComputationGraph
 *
 * @author Alex Black
 */
public class PairToArrayPair<K> implements PairFunction<Tuple2<K, INDArray>, K, INDArray[]> {
    @Override
    public Tuple2<K, INDArray[]> call(Tuple2<K, INDArray> v1) throws Exception {
        return new Tuple2<>(v1._1(), new INDArray[] {v1._2()});
    }
}

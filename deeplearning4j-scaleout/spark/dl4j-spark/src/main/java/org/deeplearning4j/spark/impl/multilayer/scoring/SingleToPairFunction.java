package org.deeplearning4j.spark.impl.multilayer.scoring;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import scala.Tuple2;

public class SingleToPairFunction<T> implements PairFunction<Tuple2<T,INDArray>, T,Pair<INDArray,INDArray>> {


    @Override
    public Tuple2<T, Pair<INDArray, INDArray>> call(Tuple2<T, INDArray> t2) throws Exception {
        return new Tuple2<>(t2._1(), new Pair<INDArray, INDArray>(t2._2(), null));
    }
}

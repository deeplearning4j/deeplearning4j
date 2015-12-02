package org.deeplearning4j.spark.impl.common.misc;

import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;
import scala.Tuple2;

public class GradientFromTupleFunction implements Function<Tuple2<Gradient,Updater>,Gradient> {
    @Override
    public Gradient call(Tuple2<Gradient, Updater> tuple2) throws Exception {
        return tuple2._1();
    }
}

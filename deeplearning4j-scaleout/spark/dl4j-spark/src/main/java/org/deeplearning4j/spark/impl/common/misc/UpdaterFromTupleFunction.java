package org.deeplearning4j.spark.impl.common.misc;

import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import scala.Tuple2;
import scala.Tuple3;

public class UpdaterFromTupleFunction implements Function<Tuple2<MultiLayerNetwork, Double>,Updater> {
    @Override
    public Updater call(Tuple2<MultiLayerNetwork, Double> t2) throws Exception {
        return t2._1().getUpdater();
    }
}

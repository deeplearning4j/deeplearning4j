package org.deeplearning4j.spark.impl.common.misc;

import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.nn.api.Updater;
import org.nd4j.linalg.api.ndarray.INDArray;
import scala.Tuple3;


public class UpdaterFromTupleFunction implements Function<Tuple3<INDArray,Updater,Double>,Updater> {
    @Override
    public Updater call(Tuple3<INDArray, Updater, Double> indArrayTuple2) throws Exception {
        return indArrayTuple2._2();
    }
}
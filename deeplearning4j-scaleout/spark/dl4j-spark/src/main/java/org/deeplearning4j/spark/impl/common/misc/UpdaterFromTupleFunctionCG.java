package org.deeplearning4j.spark.impl.common.misc;

import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.nd4j.linalg.api.ndarray.INDArray;
import scala.Tuple3;


public class UpdaterFromTupleFunctionCG implements Function<Tuple3<INDArray,ComputationGraphUpdater,Double>,ComputationGraphUpdater> {
    @Override
    public ComputationGraphUpdater call(Tuple3<INDArray, ComputationGraphUpdater, Double> indArrayTuple2) throws Exception {
        return indArrayTuple2._2();
    }
}
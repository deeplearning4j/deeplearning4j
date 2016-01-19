package org.deeplearning4j.spark.impl.common.misc;

import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import scala.Tuple3;

public class UpdaterFromGradientTupleFunctionCG implements Function<Tuple3<Gradient,ComputationGraphUpdater,Double>,ComputationGraphUpdater> {
    @Override
    public ComputationGraphUpdater call(Tuple3<Gradient, ComputationGraphUpdater, Double> indArrayTuple2) throws Exception {
        return indArrayTuple2._2();
    }
}

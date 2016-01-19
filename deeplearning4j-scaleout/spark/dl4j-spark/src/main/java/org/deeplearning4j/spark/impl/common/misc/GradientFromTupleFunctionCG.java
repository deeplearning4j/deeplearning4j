package org.deeplearning4j.spark.impl.common.misc;

import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import scala.Tuple2;
import scala.Tuple3;

public class GradientFromTupleFunctionCG implements Function<Tuple3<Gradient,ComputationGraphUpdater,Double>,Gradient> {
    @Override
    public Gradient call(Tuple3<Gradient, ComputationGraphUpdater,Double> tuple3) throws Exception {
        return tuple3._1();
    }
}

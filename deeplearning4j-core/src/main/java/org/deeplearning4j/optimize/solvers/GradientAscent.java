package org.deeplearning4j.optimize.solvers;


import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.api.StepFunction;
import org.deeplearning4j.optimize.api.TerminationCondition;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;

/**
 * Vectorized Stochastic Gradient Ascent
 * @author Adam Gibson
 *
 */
public class GradientAscent extends BaseOptimizer {


    public GradientAscent(NeuralNetConfiguration conf, StepFunction stepFunction, Collection<IterationListener> iterationListeners, Model model) {
        super(conf, stepFunction, iterationListeners, model);
    }

    public GradientAscent(NeuralNetConfiguration conf, StepFunction stepFunction, Collection<IterationListener> iterationListeners, Collection<TerminationCondition> terminationConditions, Model model) {
        super(conf, stepFunction, iterationListeners, terminationConditions, model);
    }



    @Override
    public void preProcessLine(INDArray line) {
        double norm2 = line.norm2(Integer.MAX_VALUE).getDouble(0);
        if(norm2 > stpMax)
            line.muli(stpMax / norm2);



    }

    @Override
    public void postStep() {
        //no-op
    }

    @Override
    public void setupSearchState(Pair<Gradient, Double> pair) {
        super.setupSearchState(pair);
    }


}

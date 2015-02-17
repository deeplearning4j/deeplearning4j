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
 * No line search gradient descent
 * @author Adam Gibson
 */
public class IterationGradientDescent extends BaseOptimizer {


    public IterationGradientDescent(NeuralNetConfiguration conf, StepFunction stepFunction, Collection<IterationListener> iterationListeners, Model model) {
        super(conf, stepFunction, iterationListeners, model);
    }

    public IterationGradientDescent(NeuralNetConfiguration conf, StepFunction stepFunction, Collection<IterationListener> iterationListeners, Collection<TerminationCondition> terminationConditions, Model model) {
        super(conf, stepFunction, iterationListeners, terminationConditions, model);
    }


    @Override
    public boolean optimize() {
        for(int i = 0; i < conf.getNumIterations(); i++) {
            model.setScore();
            Pair<Gradient,Double> score = gradientAndScore();
            updateGradientAccordingToParams(score.getFirst(), model, batchSize());
            model.update(score.getFirst());
            for(IterationListener listener : conf.getListeners())
                listener.iterationDone(model,i);

        }
        return true;
    }

    @Override
    public void preProcessLine(INDArray line) {
          if(conf.isConstrainGradientToUnitNorm())
              line.divi(line.norm2(Integer.MAX_VALUE));
    }

    @Override
    public void postStep() {

    }
}

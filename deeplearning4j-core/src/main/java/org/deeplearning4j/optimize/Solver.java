package org.deeplearning4j.optimize;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.solvers.*;
import java.util.Collection;

/**
 * Generic purpose solver
 * @author Adam Gibson
 */
public class Solver {
    private NeuralNetConfiguration conf;
    private Collection<IterationListener> listeners;
    private Model model;
    private BaseOptimizer optimizer;


    public void optimize() {
        if(optimizer == null)
            optimizer = getOptimizer();
        optimizer.optimize();

    }

    public BaseOptimizer getOptimizer() {
        NeuralNetwork.OptimizationAlgorithm algo = conf.getOptimizationAlgo();
        switch(algo) {
            case LBFGS:
                return new LBFGS(conf,conf.getStepFunction(),listeners,model);
            case GRADIENT_DESCENT:
                return new GradientAscent(conf,conf.getStepFunction(),listeners,model);
            case HESSIAN_FREE:
                return new StochasticHessianFree(conf,conf.getStepFunction(),listeners,model);
            case CONJUGATE_GRADIENT:
                return new ConjugateGradient(conf,conf.getStepFunction(),listeners,model);
            case ITERATION_GRADIENT_DESCENT:
                return new IterationGradientDescent(conf,conf.getStepFunction(),listeners,model);
        }

        throw new IllegalStateException("No optimizer found");
    }

    public static class Builder {
        private NeuralNetConfiguration conf;
        private Collection<IterationListener> listeners;
        private Model model;


        public Builder configure(NeuralNetConfiguration conf) {
            this.conf = conf;
            return this;
        }



        public Builder listeners(Collection<IterationListener> listeners) {
            this.listeners = listeners;
            return this;
        }

        public Builder model(Model model) {
            this.model = model;
            return this;
        }

        public Solver build() {
            Solver solver = new Solver();
            solver.conf = conf;
            solver.listeners = listeners;
            solver.model = model;
            return solver;
        }
    }


}

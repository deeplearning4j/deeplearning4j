package org.deeplearning4j.optimize.solvers;

import org.deeplearning4j.optimize.api.OptimizableByGradientValueMatrix;
import org.deeplearning4j.optimize.api.TrainingEvaluator;
import org.deeplearning4j.util.OptimizerMatrix;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Normal iteration based gradient descent with basic early stopping
 * @author Adam Gibson
 */
public class IterationGradientDescent implements OptimizerMatrix {
    private OptimizableByGradientValueMatrix optimizable;
    private int iterations = 100;
    private double score = 0.0;
    private static Logger log = LoggerFactory.getLogger(IterationGradientDescent.class);

    public IterationGradientDescent(OptimizableByGradientValueMatrix optimizable,int iterations) {
        this.optimizable = optimizable;
        this.iterations = iterations;
    }

    @Override
    public boolean optimize() {

        for(int i = 0; i < iterations; i++) {
            INDArray params = optimizable.getParameters();
            optimizable.setParameters(params.addi(optimizable.getValueGradient(0)));
            double score = optimizable.getValue();
            log.info("Score at iteration " + i + " is "  + score);
            if(this.score != 0) {
                double diff = Math.abs(score - this.score);
                if(diff < Nd4j.EPS_THRESHOLD) {
                    log.info("Breaking early...no change");
                    break;
                }
            }
        }
        return true;
    }

    @Override
    public boolean optimize(int numIterations) {
        return optimize();
    }

    @Override
    public boolean isConverged() {
        return true;
    }

    @Override
    public void setMaxIterations(int maxIterations) {

    }

    @Override
    public void setTolerance(double tolerance) {

    }

    @Override
    public void setTrainingEvaluator(TrainingEvaluator eval) {

    }
}

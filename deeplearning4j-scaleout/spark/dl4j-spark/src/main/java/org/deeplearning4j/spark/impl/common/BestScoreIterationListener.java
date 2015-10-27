package org.deeplearning4j.spark.impl.common;

import org.apache.spark.Accumulator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;

/**
 * Iteration listener which sends score to BestScoreAcumulator
 *
 * Created by mlapan on 27/10/15.
 */
public class BestScoreIterationListener implements IterationListener {
    private Accumulator<Double> best_score_acc = null;

    public BestScoreIterationListener(Accumulator<Double> best_score_acc) {
        this.best_score_acc = best_score_acc;
    }

    @Override
    public boolean invoked() {
        return false;
    }

    @Override
    public void invoke() {
    }

    @Override
    public void iterationDone(Model model, int iteration) {
        if (best_score_acc != null)
            best_score_acc.add(model.score());
    }
}

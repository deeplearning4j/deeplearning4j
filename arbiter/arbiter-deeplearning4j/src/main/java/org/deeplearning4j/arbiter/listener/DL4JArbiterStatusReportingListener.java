package org.deeplearning4j.arbiter.listener;

import lombok.AllArgsConstructor;
import org.deeplearning4j.arbiter.optimize.runner.CandidateInfo;
import org.deeplearning4j.arbiter.optimize.runner.listener.StatusListener;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.deeplearning4j.optimize.api.IterationListener;

import java.util.List;

/**
 * A simple DL4J Iteration listener that calls Arbiter's status listeners
 *
 * @author Alex Black
 */
@AllArgsConstructor
public class DL4JArbiterStatusReportingListener extends BaseTrainingListener {

    private List<StatusListener> statusListeners;
    private CandidateInfo candidateInfo;

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        if (statusListeners == null) {
            return;
        }

        for (StatusListener sl : statusListeners) {
            sl.onCandidateIteration(candidateInfo, model, iteration);
        }
    }
}

package org.deeplearning4j.arbiter.listener;

import lombok.AllArgsConstructor;
import org.deeplearning4j.arbiter.optimize.runner.CandidateInfo;
import org.deeplearning4j.arbiter.optimize.runner.listener.StatusListener;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;

import java.util.List;

/**
 * Created by Alex on 21/07/2017.
 */
@AllArgsConstructor
public class DL4JArbiterStatusReportingListener implements IterationListener {

    private List<StatusListener> statusListeners;
    private CandidateInfo candidateInfo;


    @Override
    public boolean invoked() {
        return false;
    }

    @Override
    public void invoke() {

    }

    @Override
    public void iterationDone(Model model, int iteration) {
        if(statusListeners == null){
            return;
        }

        for(StatusListener sl : statusListeners){
            sl.onCandidateIteration(candidateInfo, model, iteration);
        }
    }
}

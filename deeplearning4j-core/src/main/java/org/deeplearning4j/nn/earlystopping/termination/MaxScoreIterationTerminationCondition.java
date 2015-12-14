package org.deeplearning4j.nn.earlystopping.termination;

import lombok.AllArgsConstructor;

@AllArgsConstructor
public class MaxScoreIterationTerminationCondition implements IterationTerminationCondition {

    private double maxScore;

    @Override
    public void initialize() {
        //no op
    }

    @Override
    public boolean terminate(double lastMiniBatchScore) {
        return lastMiniBatchScore > maxScore;
    }

    @Override
    public String toString(){
        return "MaxScoreIterationTerminationCondition("+maxScore+")";
    }
}

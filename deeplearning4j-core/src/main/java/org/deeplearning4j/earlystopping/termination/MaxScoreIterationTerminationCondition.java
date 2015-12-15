package org.deeplearning4j.earlystopping.termination;

/** Iteration termination condition for terminating training if the minibatch score exceeds a certain value.
 * This can occur for example with a poorly tuned (too high) learning rate
 */
public class MaxScoreIterationTerminationCondition implements IterationTerminationCondition {

    private double maxScore;

    public MaxScoreIterationTerminationCondition(double maxScore) {
        this.maxScore = maxScore;
    }

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

package org.deeplearning4j.earlystopping.termination;

/**
 * Created by Sadat Anwar on 3/26/16.
 *
 * Stop the training once we achieved an expected score. Normally this will stop if the current score is lower than
 * the initialized score. If you want to stop the training once the score increases the defined score set the
 * lesserBetter flag to false (feel free to give the flag a better name)
 */
public class BestScoreEpochTerminationCondition  implements EpochTerminationCondition{
    private final double bestExpectedScore;
    private boolean lesserBetter = true;

    public BestScoreEpochTerminationCondition(double bestExpectedScore){
        this.bestExpectedScore = bestExpectedScore;
    }

    public BestScoreEpochTerminationCondition(double bestExpectedScore, boolean lesserBetter){
        this(bestExpectedScore);
        this.lesserBetter = lesserBetter;
    }

    @Override
    public void initialize() {
        /* No OP */
    }

    @Override
    public boolean terminate(int epochNum, double score) {
        if (lesserBetter) {
            return score < bestExpectedScore;
        } else{
            return bestExpectedScore < score;
        }
    }

    @Override
    public String toString(){
        return "BestScoreEpochTerminationCondition("+bestExpectedScore+")";
    }
}

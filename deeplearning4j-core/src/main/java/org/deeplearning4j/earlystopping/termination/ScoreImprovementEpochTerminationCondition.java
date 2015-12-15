package org.deeplearning4j.earlystopping.termination;

/** Terminate training if best model score does not improve for N epochs*/
public class ScoreImprovementEpochTerminationCondition implements EpochTerminationCondition {

    private int maxEpochsWithNoImprovement;
    private int bestEpoch = -1;
    private double bestScore;

    public ScoreImprovementEpochTerminationCondition(int maxEpochsWithNoImprovement) {
        this.maxEpochsWithNoImprovement = maxEpochsWithNoImprovement;
    }

    @Override
    public void initialize() {
        //No op
    }

    @Override
    public boolean terminate(int epochNum, double score) {
        if(bestEpoch == -1){
            bestEpoch = epochNum;
            bestScore = score;
            return false;
        } else {
            if(score < bestScore){
                bestScore = score;
                bestEpoch = epochNum;
                return false;
            }

            return epochNum >= bestEpoch + maxEpochsWithNoImprovement;
        }
    }

    @Override
    public String toString(){
        return "ScoreImprovementEpochTerminationCondition(maxEpochsWithNoImprovement="+maxEpochsWithNoImprovement+")";
    }
}

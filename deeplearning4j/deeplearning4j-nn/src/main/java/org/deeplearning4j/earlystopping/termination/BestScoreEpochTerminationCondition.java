package org.deeplearning4j.earlystopping.termination;

import lombok.Data;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Created by Sadat Anwar on 3/26/16.
 *
 * Stop the training once we achieved an expected score. Normally this will stop if the current score is lower than
 * the initialized score. If you want to stop the training once the score increases the defined score set the
 * lesserBetter flag to false (feel free to give the flag a better name)
 */
@Data
public class BestScoreEpochTerminationCondition implements EpochTerminationCondition {
    @JsonProperty
    private final double bestExpectedScore;

    @JsonProperty
    private boolean lesserBetter = true;

    public BestScoreEpochTerminationCondition(double bestExpectedScore) {
        this.bestExpectedScore = bestExpectedScore;
    }

    public BestScoreEpochTerminationCondition(double bestExpectedScore, boolean lesserBetter) {
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
        } else {
            return bestExpectedScore < score;
        }
    }

    @Override
    public String toString() {
        return "BestScoreEpochTerminationCondition(" + bestExpectedScore + ")";
    }
}

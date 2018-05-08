package org.deeplearning4j.optimize.listeners;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.eclipse.collections.impl.list.mutable.primitive.DoubleArrayList;
import org.eclipse.collections.impl.list.mutable.primitive.IntArrayList;

import java.io.Serializable;

/**
 * A simple listener that collects scores to a list every N iterations. Can also optionally log the score.
 *
 * @author Alex Black
 */
@Data
@Slf4j
public class CollectScoresListener extends BaseTrainingListener implements Serializable {

    private final int frequency;
    private final boolean logScore;
    private final IntArrayList listIteration;
    private final DoubleArrayList listScore;

    public CollectScoresListener(int frequency) {
        this(frequency, false);
    }

    public CollectScoresListener(int frequency, boolean logScore){
        this.frequency = frequency;
        this.logScore = logScore;
        listIteration = new IntArrayList();
        listScore = new DoubleArrayList();
    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        if(iteration % frequency == 0){
            double score = model.score();
            listIteration.add(iteration);
            listScore.add(score);
            if(logScore) {
                log.info("Score at iteration {} is {}", iteration, score);
            }
        }
    }
}

package org.deeplearning4j.spark.models.glove;

import org.deeplearning4j.scaleout.api.statetracker.StateTracker;
import org.deeplearning4j.scaleout.perform.BaseWorkPerformerFactory;
import org.deeplearning4j.scaleout.perform.WorkerPerformer;

/**
 * Work performer factory for word2vec
 * @author Adam Gibson
 */
public class GlovePerformerFactory extends BaseWorkPerformerFactory {

    private StateTracker stateTracker;

    public GlovePerformerFactory() {}

    public GlovePerformerFactory(StateTracker stateTracker) {
        this.stateTracker = stateTracker;
    }

    @Override
    public WorkerPerformer instantiate() {
        if(stateTracker != null)
            return new org.deeplearning4j.spark.models.glove.GlovePerformer(stateTracker);
        return new org.deeplearning4j.spark.models.glove.GlovePerformer();
    }
}

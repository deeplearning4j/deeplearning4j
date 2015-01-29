package org.deeplearning4j.models.word2vec;

import org.deeplearning4j.scaleout.api.statetracker.StateTracker;
import org.deeplearning4j.scaleout.perform.BaseWorkPerformerFactory;
import org.deeplearning4j.scaleout.perform.WorkerPerformer;

/**
 * Work performer factory for word2vec
 * @author Adam Gibson
 */
public class Word2VecPerformerFactory extends BaseWorkPerformerFactory {

    private StateTracker stateTracker;

    public Word2VecPerformerFactory() {}

    public Word2VecPerformerFactory(StateTracker stateTracker) {
        this.stateTracker = stateTracker;
    }

    @Override
    public WorkerPerformer instantiate() {
        if(stateTracker != null)
            return new Word2VecPerformer(stateTracker);
        return new Word2VecPerformer();
    }
}

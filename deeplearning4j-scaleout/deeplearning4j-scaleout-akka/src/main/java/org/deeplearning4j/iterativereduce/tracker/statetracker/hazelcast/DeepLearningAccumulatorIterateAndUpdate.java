package org.deeplearning4j.iterativereduce.tracker.statetracker.hazelcast;

import org.deeplearning4j.iterativereduce.akka.DeepLearningAccumulator;
import org.deeplearning4j.iterativereduce.tracker.statetracker.IterateAndUpdate;
import org.deeplearning4j.iterativereduce.tracker.statetracker.UpdateSaver;
import org.deeplearning4j.scaleout.iterativereduce.multi.UpdateableImpl;

import java.util.Collection;

/**
 * This takes in the accumulator, the update saver, and the ids of
 * the workers and handles iterating and loading over each one of them at a time.
 * @author  Adam Gibson
 */
public class DeepLearningAccumulatorIterateAndUpdate implements IterateAndUpdate<UpdateableImpl> {


    private DeepLearningAccumulator accumulator;
    private UpdateSaver<UpdateableImpl> updateSaver;
    private Collection<String> ids;


    public DeepLearningAccumulatorIterateAndUpdate(DeepLearningAccumulator accumulator, UpdateSaver<UpdateableImpl> updateSaver, Collection<String> ids) {
        this.accumulator = accumulator;
        this.updateSaver = updateSaver;
        this.ids = ids;
    }

    /**
     * The accumulated result
     *
     * @return the accumulated result
     */
    @Override
    public UpdateableImpl accumulated() {
        return new UpdateableImpl(accumulator.averaged());
    }

    @Override
    public void accumulate() throws Exception {
        for(String s : ids)
            accumulator.accumulate(updateSaver.load(s).get());

    }


}

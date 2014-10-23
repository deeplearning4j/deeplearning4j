package org.deeplearning4j.iterativereduce.tracker.statetracker.hazelcast.deepautoencoder;

import org.deeplearning4j.iterativereduce.akka.DeepAutoEncoderAccumulator;
import org.deeplearning4j.iterativereduce.akka.DeepLearningAccumulator;
import org.deeplearning4j.iterativereduce.tracker.statetracker.IterateAndUpdate;
import org.deeplearning4j.iterativereduce.tracker.statetracker.UpdateSaver;
import org.deeplearning4j.scaleout.iterativereduce.deepautoencoder.UpdateableEncoderImpl;
import org.deeplearning4j.scaleout.iterativereduce.multi.UpdateableImpl;

import java.util.Collection;

/**
 * This takes in the accumulator, the update saver, and the ids of
 * the workers and handles iterating and loading over each one of them at a time.
 * @author  Adam Gibson
 */
public class DeepAutoEncoderAccumulatorIterateAndUpdate implements IterateAndUpdate<UpdateableEncoderImpl> {


    private DeepAutoEncoderAccumulator accumulator;
    private UpdateSaver<UpdateableEncoderImpl> updateSaver;
    private Collection<String> ids;


    public DeepAutoEncoderAccumulatorIterateAndUpdate(DeepAutoEncoderAccumulator accumulator, UpdateSaver<UpdateableEncoderImpl> updateSaver, Collection<String> ids) {
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
    public UpdateableEncoderImpl accumulated() {
        return new UpdateableEncoderImpl(accumulator.averaged());
    }

    @Override
    public void accumulate() throws Exception {
        for(String s : ids)
            accumulator.accumulate(updateSaver.load(s).get());

    }


}

package org.deeplearning4j.scaleout.statetracker.hazelcast;

import org.deeplearning4j.scaleout.aggregator.JobAggregator;
import org.deeplearning4j.scaleout.aggregator.WorkAccumulator;
import org.deeplearning4j.scaleout.job.Job;

import org.deeplearning4j.scaleout.statetracker.IterateAndUpdate;
import org.deeplearning4j.scaleout.statetracker.UpdateSaver;

import java.util.Collection;

/**
 * This takes in the accumulator, the update saver, and the ids of
 * the workers and handles iterating and loading over each one of them at a time.
 * @author  Adam Gibson
 */
public class IterateAndUpdateImpl implements IterateAndUpdate {


    private JobAggregator accumulator;
    private UpdateSaver updateSaver;
    private Collection<String> ids;


    public IterateAndUpdateImpl(JobAggregator accumulator, UpdateSaver updateSaver, Collection<String> ids) {
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
    public Job accumulated() {
        return accumulator.aggregate();
    }

    @Override
    public void accumulate() throws Exception {
        for(String s : ids) {
            Job j = updateSaver.load(s);
            if(j == null)
                continue;
            accumulator.accumulate(j);
        }
    }


}

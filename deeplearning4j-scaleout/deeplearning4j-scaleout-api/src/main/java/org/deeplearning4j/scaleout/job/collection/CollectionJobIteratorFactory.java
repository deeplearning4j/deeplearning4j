package org.deeplearning4j.scaleout.job.collection;

import org.deeplearning4j.scaleout.job.Job;
import org.deeplearning4j.scaleout.job.JobIterator;
import org.deeplearning4j.scaleout.job.JobIteratorFactory;

import java.util.Collection;

/**
 * Collection job iterator factory
 * @author Adam Gibson
 */
public class CollectionJobIteratorFactory implements JobIteratorFactory {
    private Collection<Job> jobs;

    public CollectionJobIteratorFactory(Collection<Job> jobs) {
        this.jobs = jobs;
    }


    @Override
    public JobIterator create() {
        return new CollectionJobIterator(jobs);
    }
}

package org.deeplearning4j.scaleout.job.collection;

import org.deeplearning4j.scaleout.job.Job;
import org.deeplearning4j.scaleout.job.JobIterator;

import java.util.Collection;
import java.util.Iterator;

/**
 * Iterate over a collection
 * @author Adam Gibson
 */
public class CollectionJobIterator implements JobIterator {
    protected Iterator<Job> jobs;
    protected Collection<Job> jobCollection;

    /**
     *
     * @param jobs the jobs to iterate over
     *             (note that this WILL be cached for the reset operation)
     */
    public CollectionJobIterator(Collection<Job> jobs) {
        this.jobs = jobs.iterator();
        this.jobCollection = jobs;
    }


    @Override
    public Job next(String workerId) {
        Job next = jobs.next();
        next.setWorkerId(workerId);
        return next;
    }

    @Override
    public Job next() {
        return jobs.next();
    }

    @Override
    public boolean hasNext() {
        return jobs.hasNext();
    }

    @Override
    public void reset() {
       jobs = jobCollection.iterator();
    }
}

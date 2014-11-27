package org.deeplearning4j.scaleout.job;

/**
 * Job iterator
 *
 * @author Adam Gibson
 */
public interface JobIterator {


    /**
     * Assigns a worker id
     * @param workerId
     * @return
     */
    Job next(String workerId);

    /**
     * Get the next job
     * @return
     */
    Job next();


    /**
     * Whether there are anymore jobs
     * @return
     */
    boolean hasNext();

    /**
     * Reset to the beginning
     */
    void reset();

}

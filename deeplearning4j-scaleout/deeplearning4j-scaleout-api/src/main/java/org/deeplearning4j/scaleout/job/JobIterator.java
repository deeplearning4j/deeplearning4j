package org.deeplearning4j.scaleout.job;

/**
 * Job iterator
 *
 * @author Adam Gibson
 */
public interface JobIterator {






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

package org.deeplearning4j.scaleout.job;

/**
 *
 * Create a job iterator
 * @author Adam Gibson
 */
public interface JobIteratorFactory {

    /**
     * Create a job iterator
     * @return
     */
    JobIterator create();

}

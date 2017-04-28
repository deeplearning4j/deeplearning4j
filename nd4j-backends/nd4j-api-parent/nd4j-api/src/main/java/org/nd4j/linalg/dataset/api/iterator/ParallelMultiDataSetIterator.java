package org.nd4j.linalg.dataset.api.iterator;

import org.nd4j.linalg.dataset.api.MultiDataSet;

/**
 * @author raver119@gmail.com
 */
public interface ParallelMultiDataSetIterator extends MultiDataSetIterator {

    /**
     * This method sets consumer affinity to specific producer
     *
     * PLEASE NOTE: this method is optional, and it'll change only nextFor()/hasNextFor() mechanics
     */
    void attachThread(int producer);

    /**
     * Returns true, if attached producer has something in queue, false otherwise
     *
     * @return
     */
    boolean hasNextFor();

    /**
     * Returns true, if attached producer has something in queue, false otherwise
     *
     * @param consumer
     * @return
     */
    boolean hasNextFor(int consumer);

    /**
     * Returns next DataSet for given consumer
     *
     * @param consumer
     * @return
     */
    MultiDataSet nextFor(int consumer);

    /**
     * Returns next DataSet for attached consumer
     *
     * @return
     */
    MultiDataSet nextFor();
}

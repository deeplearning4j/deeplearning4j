package org.nd4j.linalg.api.ops.executioner;

import org.nd4j.linalg.api.ops.aggregates.Aggregate;

/**
 * @author raver119@gmail.com
 */
public interface GridExecutioner extends OpExecutioner {

    /**
     * This method forces all currently enqueued ops to be executed immediately
     *
     * PLEASE NOTE: This call CAN be non-blocking, if specific backend implementation supports that.
     */
    void flushQueue();

    /**
     * This method forces all currently enqueued ops to be executed immediately
     *
     * PLEASE NOTE: This call is always blocking, until all queued operations are finished
     */
    void flushQueueBlocking();


    /**
     * This method returns number of operations currently enqueued for execution
     *
     * @return
     */
    int getQueueLength();


    /**
     * This method enqueues aggregate op for future invocation
     *
     * @param op
     */
    void aggregate(Aggregate op);

    /**
     * This method enqueues aggregate op for future invocation.
     * Key value will be used to batch individual ops
     *
     * @param op
     * @param key
     */
    void aggregate(Aggregate op, long key);
}

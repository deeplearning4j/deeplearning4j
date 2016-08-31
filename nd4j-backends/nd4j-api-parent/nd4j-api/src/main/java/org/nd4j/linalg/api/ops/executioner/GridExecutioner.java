package org.nd4j.linalg.api.ops.executioner;

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


    int getQueueLength();
}

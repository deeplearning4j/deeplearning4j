package org.nd4j.parameterserver.distributed.logic;

import lombok.NonNull;
import org.nd4j.parameterserver.distributed.messages.aggregations.VoidAggregation;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * Since VoidParameterServer assumes nearly endless asynchronous data flow, we'll use Clipboard approach to aggregate
 * different batches of aggregates coming in un-ordered.
 *
 * @author raver119@gmail.com
 */
public class Clipboard {

    protected AtomicInteger completedCounter = new AtomicInteger(0);

    /**
     * This method places incoming VoidAggregation into clipboard, for further tracking
     *
     * @param aggregation
     * @return TRUE, if given VoidAggregation was the last chunk, FALSE otherwise
     */
    public boolean pin(@NonNull VoidAggregation aggregation) {
        // TODO: to be implemented
        return false;
    }

    /**
     * This method removes given VoidAggregation from clipboard, and returns it
     *
     * @param aggregation
     */
    public VoidAggregation unpin(@NonNull VoidAggregation aggregation) {
        // TODO: to be implemented
        return null;
    }

    /**
     * This method removes given VoidAggregation from clipboard, and returns it
     *
     * @param batchId
     */
    public VoidAggregation unpin(long batchId) {
        // TODO: to be implemented
        return null;
    }

    /**
     * This method checks, if clipboard has ready aggregations available
     *
     * @return TRUE, if there's at least 1 candidate ready, FALSE otherwise
     */
    public boolean hasCandidates() {
        return false;
    }

    /**
     * This method returns one of available aggregations, if there's at least 1 ready.
     *
     * @return
     */
    public VoidAggregation nextCandidate() {

        completedCounter.decrementAndGet();

        return null;
    }
}

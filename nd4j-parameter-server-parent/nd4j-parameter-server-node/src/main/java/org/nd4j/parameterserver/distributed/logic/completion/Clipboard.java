package org.nd4j.parameterserver.distributed.logic.completion;

import lombok.NonNull;
import org.nd4j.parameterserver.distributed.messages.VoidAggregation;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Since VoidParameterServer assumes nearly endless asynchronous data flow, we'll use Clipboard approach to aggregate
 * different batches of aggregates coming in un-ordered.
 *
 * @author raver119@gmail.com
 */
public class Clipboard {


    protected Map<Long, VoidAggregation> clipboard = new ConcurrentHashMap<>();

    protected Queue<VoidAggregation> completedQueue = new ConcurrentLinkedQueue<>();

    protected AtomicInteger trackingCounter = new AtomicInteger(0);
    protected AtomicInteger completedCounter = new AtomicInteger(0);

    /**
     * This method places incoming VoidAggregation into clipboard, for further tracking
     *
     * @param aggregation
     * @return TRUE, if given VoidAggregation was the last chunk, FALSE otherwise
     */
    public boolean pin(@NonNull VoidAggregation aggregation) {
        VoidAggregation existing = clipboard.get(aggregation.getTaskId());
        if (existing == null) {
            existing = aggregation;
            trackingCounter.incrementAndGet();
            clipboard.put(aggregation.getTaskId(), aggregation);
        }

        existing.accumulateAggregation(aggregation);

        int missing = existing.getMissingChunks();
        if (missing == 0) {
            completedQueue.add(existing);
            completedCounter.incrementAndGet();

            // TODO: delete it from tracking table probably?

            return true;
        } else return false;
    }

    /**
     * This method removes given VoidAggregation from clipboard, and returns it
     *
     * @param aggregation
     */
    public VoidAggregation unpin(@NonNull VoidAggregation aggregation) {
        return unpin(aggregation.getTaskId());
    }

    /**
     * This method removes given VoidAggregation from clipboard, and returns it
     *
     * @param taskId
     */
    public VoidAggregation unpin(long taskId) {
        VoidAggregation aggregation;
        if ((aggregation = clipboard.get(taskId)) != null) {
            clipboard.remove(taskId);
            trackingCounter.decrementAndGet();
            return aggregation;
        } else return null;
    }

    /**
     * This method checks, if clipboard has ready aggregations available
     *
     * @return TRUE, if there's at least 1 candidate ready, FALSE otherwise
     */
    public boolean hasCandidates() {
        return completedCounter.get() > 0;
    }

    /**
     * This method returns one of available aggregations, if there's at least 1 ready.
     *
     * @return
     */
    public VoidAggregation nextCandidate() {
        completedCounter.decrementAndGet();

        VoidAggregation result = completedQueue.poll();

        // removing aggregation from tracking table
        if (result != null)
            unpin(result.getTaskId());

        return result;
    }

    public boolean isReady(Long taskId) {
        VoidAggregation aggregation = clipboard.get(taskId);
        if (aggregation == null)
            return false;

        return aggregation.getMissingChunks() == 0;
    }

    public boolean isTracking(Long taskId) {
        return clipboard.containsKey(taskId);
    }

    public int getNumberOfPinnedStacks() {
        return trackingCounter.get();
    }

    public int getNumberOfCompleteStacks() {
        return completedCounter.get();
    }

    public VoidAggregation getStackFromClipboard(long taskId) {
        return clipboard.get(taskId);
    }
}

package org.nd4j.parameterserver.distributed.logic.completion;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.parameterserver.distributed.messages.VoidAggregation;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Since VoidParameterServer assumes nearly endless asynchronous data flow, we'll use Clipboard approach to aggregate
 * different batches of aggregates coming in un-ordered.
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class Clipboard {
    protected Map<RequestDescriptor, VoidAggregation> clipboard = new ConcurrentHashMap<>();

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
        RequestDescriptor descriptor =
                        RequestDescriptor.createDescriptor(aggregation.getOriginatorId(), aggregation.getTaskId());
        VoidAggregation existing = clipboard.get(descriptor);
        if (existing == null) {
            existing = aggregation;
            trackingCounter.incrementAndGet();
            clipboard.put(descriptor, aggregation);
        }

        existing.accumulateAggregation(aggregation);

        //if (counter.incrementAndGet() % 10000 == 0)
        //    log.info("Clipboard stats: Totals: {}; Completed: {};", clipboard.size(), completedQueue.size());

        int missing = existing.getMissingChunks();
        if (missing == 0) {
            //  completedQueue.add(existing);
            completedCounter.incrementAndGet();
            return true;
        } else
            return false;
    }

    /**
     * This method removes given VoidAggregation from clipboard, and returns it
     *
     * @param aggregation
     */
    public VoidAggregation unpin(@NonNull VoidAggregation aggregation) {
        return unpin(aggregation.getOriginatorId(), aggregation.getTaskId());
    }

    /**
     * This method removes given VoidAggregation from clipboard, and returns it
     *
     * @param taskId
     */
    public VoidAggregation unpin(long originatorId, long taskId) {
        RequestDescriptor descriptor = RequestDescriptor.createDescriptor(originatorId, taskId);
        VoidAggregation aggregation;
        if ((aggregation = clipboard.get(descriptor)) != null) {
            clipboard.remove(descriptor);
            trackingCounter.decrementAndGet();

            // FIXME: we don't want this here
            //            completedQueue.clear();

            return aggregation;
        } else
            return null;
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
        VoidAggregation result = completedQueue.poll();

        // removing aggregation from tracking table
        if (result != null) {
            completedCounter.decrementAndGet();
            unpin(result.getOriginatorId(), result.getTaskId());
        }

        return result;
    }

    public boolean isReady(VoidAggregation aggregation) {
        return isReady(aggregation.getOriginatorId(), aggregation.getTaskId());
    }

    public boolean isReady(long originatorId, long taskId) {
        RequestDescriptor descriptor = RequestDescriptor.createDescriptor(originatorId, taskId);
        VoidAggregation aggregation = clipboard.get(descriptor);
        if (aggregation == null)
            return false;

        return aggregation.getMissingChunks() == 0;
    }

    public boolean isTracking(long originatorId, long taskId) {
        return clipboard.containsKey(RequestDescriptor.createDescriptor(originatorId, taskId));
    }

    public int getNumberOfPinnedStacks() {
        return trackingCounter.get();
    }

    public int getNumberOfCompleteStacks() {
        return completedCounter.get();
    }

    public VoidAggregation getStackFromClipboard(long originatorId, long taskId) {
        return clipboard.get(RequestDescriptor.createDescriptor(originatorId, taskId));
    }
}

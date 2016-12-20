package org.nd4j.parameterserver.distributed.messages.aggregations;

import lombok.Data;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.agrona.concurrent.UnsafeBuffer;
import org.apache.commons.lang3.SerializationUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author raver119@gmail.com
 */
@Data
@Slf4j
public abstract class BaseAggregation extends BaseVoidMessage implements VoidAggregation, Serializable {
    protected short aggregationType = -1;
    protected short aggregationWidth;
    protected int numberOfElements;
    protected short shardIndex;
    protected Long taskId;


    protected INDArray payload;

    // transient part
    protected transient AtomicInteger chunksCounter = new AtomicInteger(1);
    protected transient Map<Short, INDArray> chunks = new TreeMap<>();

    public BaseAggregation(long taskId, short aggregationWidth, short shardIndex) {
        if (aggregationWidth < 2)
            throw new ND4JIllegalStateException("Aggregations smaller then 2 elements make no sense");

        this.aggregationWidth = aggregationWidth;
        this.taskId = taskId;
        this.shardIndex = shardIndex;
    }

    public void setShardIndex(short shardIndex) {
        if (shardIndex == this.shardIndex)
            return;

        chunks.remove(this.shardIndex);
        chunks.put(shardIndex, payload);

        this.shardIndex = shardIndex;
    }

    protected void addToChunks(INDArray array) {
        chunks.put(this.shardIndex, array);
    }

    public void accumulateAggregation(@NonNull VoidAggregation aggregation) {
        if (aggregation.getAggregationType() != getAggregationType())
            throw new ND4JIllegalStateException("Trying to aggregate different aggregations!");

        // no need to do anything in this case
        if (this.getShardIndex() == aggregation.getShardIndex()) {
            return;
        }

        if (chunks.get(aggregation.getShardIndex()) == null)
            chunksCounter.incrementAndGet();

        chunks.put(aggregation.getShardIndex(), aggregation.getPayload());
    }

    @Override
    public INDArray getAccumulatedResult() {
        return Nd4j.hstack(chunks.values());
    }

    @Override
    public int getMissingChunks() {
        return aggregationWidth - chunksCounter.get();
    }

    @Override
    public int getMessageType() {
        // joint aggregation messageType for all aggregations
        return 21;
    }

    @Override
    public byte[] asBytes() {
        return SerializationUtils.serialize(this);
    }

    @Override
    public UnsafeBuffer asUnsafeBuffer() {
        return new UnsafeBuffer(asBytes());
    }
}

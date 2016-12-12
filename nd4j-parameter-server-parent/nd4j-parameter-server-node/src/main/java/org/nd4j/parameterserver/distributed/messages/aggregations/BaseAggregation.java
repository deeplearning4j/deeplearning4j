package org.nd4j.parameterserver.distributed.messages.aggregations;

import lombok.Data;
import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author raver119@gmail.com
 */
@Data
public abstract class BaseAggregation implements VoidAggregation, Serializable {
    protected short aggregationType = -1;
    protected short aggregationWidth;
    protected int numberOfElements;
    protected short shardIndex;

    protected INDArray payload;

    // transient part
    protected transient AtomicInteger chunksCounter = new AtomicInteger(1);
    protected transient Map<Short, INDArray> chunks = new TreeMap<>();

    public BaseAggregation(short aggregationWidth) {
        if (aggregationWidth < 2)
            throw new ND4JIllegalStateException("Aggregations smaller then 2 elements make no sense");

        this.aggregationWidth = aggregationWidth;
    }


    public void accumulateAggregation(@NonNull VoidAggregation aggregation) {
        if (aggregation.getAggregationType() != getAggregationType())
            throw new ND4JIllegalStateException("Trying to aggregate different aggregations!");

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
}

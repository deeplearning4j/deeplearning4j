/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.parameterserver.distributed.messages.aggregations;

import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.agrona.concurrent.UnsafeBuffer;
import org.apache.commons.lang3.SerializationUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
import org.nd4j.parameterserver.distributed.messages.VoidAggregation;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public abstract class BaseAggregation extends BaseVoidMessage implements VoidAggregation, Serializable {
    @Getter
    @Setter
    protected short aggregationType = -1;
    @Getter
    @Setter
    protected short aggregationWidth;
    @Getter
    @Setter
    protected int numberOfElements;
    @Getter
    protected short shardIndex;


    @Getter
    @Setter
    protected INDArray payload;

    // transient part
    @Getter
    protected transient AtomicInteger chunksCounter;
    @Getter
    protected transient Map<Short, INDArray> chunks;

    protected BaseAggregation() {
        chunksCounter = new AtomicInteger(1);
        chunks = new ConcurrentHashMap<>();
    }

    protected BaseAggregation(long taskId, short aggregationWidth, short shardIndex) {
        this();
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

        if (aggregationWidth == 1) {
            return chunks.get((short) 0);
        } else
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

    @Override
    public short getTargetId() {
        return (short) -1;
    }
}

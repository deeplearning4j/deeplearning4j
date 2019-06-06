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

package org.nd4j.parameterserver.distributed.v2.chunks.impl;

import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.primitives.AtomicBoolean;
import org.nd4j.linalg.util.SerializationUtils;
import org.nd4j.parameterserver.distributed.v2.chunks.ChunksTracker;
import org.nd4j.parameterserver.distributed.v2.chunks.VoidChunk;
import org.nd4j.parameterserver.distributed.v2.messages.VoidMessage;

import java.io.*;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Memory-based implementation of ChunksTracker
 */
@Slf4j
public class InmemoryChunksTracker<T extends VoidMessage> implements ChunksTracker<T> {
    @Getter
    private final String originId;

    private final int numChunks;

    private Map<Integer, AtomicBoolean> map = new ConcurrentHashMap<>();

    private final byte[] buffer;

    private final long size;


    public InmemoryChunksTracker(VoidChunk chunk) {
        originId = chunk.getOriginalId();
        numChunks = chunk.getNumberOfChunks();

        if (chunk.getTotalSize() > Integer.MAX_VALUE)
            throw new ND4JIllegalStateException("Total message size > Integer.MAX_VALUE");

        size = chunk.getTotalSize();

        try {
            buffer = new byte[(int) size];

            // we'll pre-initialize states map
            for (int e = 0; e < numChunks; e++)
                map.put(e, new AtomicBoolean(false));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public long size() {
        return size;
    }

    @Override
    public boolean isComplete() {
        for (val b:map.values())
            if (!b.get())
                return false;

        return true;
    }

    @Override
    public synchronized boolean append(@NonNull VoidChunk chunk) {
        val b = map.get(chunk.getChunkId());

        if (b.get())
            return isComplete();

        val offset = chunk.getChunkId() * chunk.getSplitSize();

        int cnt = 0;
        for (int e = offset; e < offset + chunk.getPayload().length; e++)
            buffer[e] = chunk.getPayload()[cnt++];

        // tagging this chunk as received
        b.set(true);

        return isComplete();
    }

    @Override
    public T getMessage() {
        if (!isComplete())
            throw new ND4JIllegalStateException("Message isn't ready for concatenation");

        try (val bais = new ByteArrayInputStream(buffer)) {
            return SerializationUtils.deserialize(bais);
        } catch (Exception e) {
            log.error("Exception: {}",e);
            throw new RuntimeException(e);
        }
    }

    @Override
    public void release() {
        //
    }
}

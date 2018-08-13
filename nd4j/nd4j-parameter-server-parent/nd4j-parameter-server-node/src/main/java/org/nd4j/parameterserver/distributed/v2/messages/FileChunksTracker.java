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

package org.nd4j.parameterserver.distributed.v2.messages;

import lombok.Getter;
import lombok.NonNull;
import lombok.val;
import org.apache.commons.lang3.SerializationUtils;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.primitives.AtomicBoolean;
import org.nd4j.parameterserver.distributed.v2.messages.ChunksTracker;
import org.nd4j.parameterserver.distributed.v2.messages.VoidChunk;
import org.nd4j.parameterserver.distributed.v2.messages.VoidMessage;

import java.io.*;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * File-based implementation of ChunksTracker
 */
public class FileChunksTracker<T extends VoidMessage> implements ChunksTracker<T> {
    @Getter
    private final String originId;

    private final int numChunks;

    private File holder;

    private Map<Integer, AtomicBoolean> map = new ConcurrentHashMap<>();

    public FileChunksTracker(VoidChunk chunk) {
        originId = chunk.getOriginalId();
        numChunks = chunk.getNumberOfChunks();
        try {
            holder = File.createTempFile("FileChunksTracker", "Message");
            holder.deleteOnExit();

            // fill file with 0s for simplicity
            try (val fos = new FileOutputStream(holder); val bos = new BufferedOutputStream(fos, 32768)) {
                for (int e = 0; e < chunk.getTotalSize(); e++)
                    bos.write(0);
            }

            // we'll pre-initialize states map
            for (int e = 0; e < numChunks; e++)
                map.put(e, new AtomicBoolean(false));

            // and write down this chunk
            append(chunk);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public boolean append(@NonNull VoidChunk chunk) {
        val b = map.get(chunk.getChunkId());

        if (b.get())
            return isComplete();

        // writing out this chunk
        try (val f = new RandomAccessFile(holder, "rw")) {
            f.seek(chunk.getChunkId() * chunk.getSplitSize());

            f.write(chunk.getPayload());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        // tagging this chunk as received
        b.set(true);

        return isComplete();
    }

    @Override
    public boolean isComplete() {
        for (val b:map.values())
            if (!b.get())
                return false;

        return true;
    }

    @Override
    public T getMessage() {
        if (!isComplete())
            throw new ND4JIllegalStateException("Message isn't ready for concatenation");

        try (val fis = new FileInputStream(holder); val bis = new BufferedInputStream(fis)) {
            return SerializationUtils.deserialize(bis);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void release() {
        try {
            holder.delete();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}

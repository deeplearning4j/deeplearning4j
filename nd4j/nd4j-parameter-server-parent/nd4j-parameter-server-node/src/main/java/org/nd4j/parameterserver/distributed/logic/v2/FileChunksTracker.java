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

package org.nd4j.parameterserver.distributed.logic.v2;

import lombok.Getter;
import org.nd4j.parameterserver.distributed.messages.v2.VoidChunk;
import org.nd4j.parameterserver.distributed.messages.v2.VoidMessage_v2;

import java.io.File;

/**
 * File-based implementation of ChunksTracker
 */
public class FileChunksTracker<T extends VoidMessage_v2> implements ChunksTracker<T> {
    @Getter
    private final String originId;

    private final int numChunks;

    private File holder;

    public FileChunksTracker(VoidChunk chunk) {
        originId = chunk.getOriginalId();
        numChunks = chunk.getNumberOfChunks();
        try {
            holder = File.createTempFile("FileChunksTracker", "Message");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public boolean isComplete() {
        return false;
    }

    @Override
    public T getMessage() {
        return null;
    }

    @Override
    public void release() {

    }
}

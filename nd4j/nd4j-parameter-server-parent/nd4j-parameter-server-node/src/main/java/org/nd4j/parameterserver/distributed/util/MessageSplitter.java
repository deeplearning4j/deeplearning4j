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

package org.nd4j.parameterserver.distributed.util;

import lombok.NonNull;
import lombok.val;
import org.apache.commons.lang3.SerializationUtils;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.parameterserver.distributed.messages.v2.VoidChunk;
import org.nd4j.parameterserver.distributed.messages.v2.VoidMessage_v2;

import java.io.*;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Optional;

/**
 * This class provides methods for splitting VoidMessages into chunks, and merging them back again
 *
 * @author raver119@gmail.com
 */
public class MessageSplitter {
    private final MessageSplitter INSTANCE = new MessageSplitter();

    protected MessageSplitter() {
        //
    }

    /**
     * This method splits VoidMessage into chunks, and returns them as Collection
     * @param message
     * @return
     */
    public Collection<VoidChunk> split(@NonNull VoidMessage_v2 message, int maxBytes) throws IOException {
        if (maxBytes <= 0)
            throw new ND4JIllegalStateException("MaxBytes must be > 0");

        val tempFile = File.createTempFile("messageSplitter","temp");
        val result = new ArrayList<VoidChunk>();

        try (val fos = new FileOutputStream(tempFile); val bos = new BufferedOutputStream(fos)) {
            // serializing original message to disc
            SerializationUtils.serialize(message, fos);

            val length = tempFile.length();
            try (val fis = new FileInputStream(tempFile); val bis = new BufferedInputStream(fis)) {
                // now we'll be reading serialized message into
                val bytes = new byte[maxBytes];
                int cnt = 0;
                int id = 0;

                while (cnt < length) {
                    val c = bis.read(bytes);

                    // FIXME: we don't really want UUID used here, it's just a placeholder for now
                    val msg = VoidChunk.builder()
                            .messageId(java.util.UUID.randomUUID().toString())
                            .originalId(message.getMessageId())
                            .chunkId(id++)
                            .payload(bytes.clone())
                            .totalSize(length)
                            .build();

                    result.add(msg);
                    cnt += c;
                }
            }
        }

        tempFile.delete();
        return result;
    }

    /**
     * This method tries to merge
     *
     * @param chunk
     * @param <T>
     * @return
     */
    public <T> Optional<T> merge(@NonNull VoidChunk chunk) {
        val originalId= chunk.getOriginalId();

        return Optional.empty();
    }
}

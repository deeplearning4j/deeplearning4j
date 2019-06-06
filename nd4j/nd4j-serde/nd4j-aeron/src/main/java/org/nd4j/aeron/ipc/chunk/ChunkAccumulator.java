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

package org.nd4j.aeron.ipc.chunk;

import org.nd4j.aeron.ipc.NDArrayMessage;

/**
 * Accumulate {@link NDArrayMessageChunk} and reassemble them in to
 * {@link NDArrayMessage}.
 *
 * @author Aadm Gibson
 */
public interface ChunkAccumulator {

    /**
     * Returns the number of chunks
     * accumulated for a given id so far
     * @param id the id to get the
     *           number of chunks for
     * @return the number of chunks accumulated
     * for a given id so far
     */
    int numChunksSoFar(String id);

    /**
     * Returns true if all chunks are present
     * @param id the id to check for
     * @return true if all the chunks are present,false otherwise
     */
    boolean allPresent(String id);

    /**
     * Reassemble an ndarray message
     * from a set of chunks
     *
     * Note that once reassemble is called,
     * the associated chunk lists will automatically
     * be removed from storage.
     *
     *
     * @param id the id to reassemble
     * @return the reassembled message
     */
    NDArrayMessage reassemble(String id);

    /**
     * Accumulate chunks
     * @param chunk the chunk to accumulate
     */
    void accumulateChunk(NDArrayMessageChunk chunk);
}

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

import org.nd4j.parameterserver.distributed.v2.messages.VoidChunk;
import org.nd4j.parameterserver.distributed.v2.messages.VoidMessage;

/**
 * This interface describes logic for tracking chunks of bigger message
 */
public interface ChunksTracker<T extends VoidMessage> {

    /**
     * This message returns ID of the original message we're tracking here
     * @return
     */
    String getOriginId();

    /**
     * This method checks if all chunks were received
     * @return true if all chunks were received, false otherwise
     */
    boolean isComplete();

    /**
     * This message appends chunk to this tracker
     * @param chunk Chunk to be added
     * @return true if that was last chunk, false otherwise
     */
    boolean append(VoidChunk chunk);

    /**
     * This method returns original message
     * @return
     */
    T getMessage();

    /**
     * This method releases all resources used (if used) for this message
     */
    void release();
}

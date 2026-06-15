/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.memory;

/**
 * Coherence states for hybrid memory buffers using an MSI-like protocol.
 * These states track the validity and ownership of data copies across devices.
 *
 * <p>The state transitions follow these rules:</p>
 * <ul>
 *   <li>Read on owner (EXCLUSIVE/MODIFIED): No change</li>
 *   <li>Read on other device: Source becomes SHARED, copy created as SHARED</li>
 *   <li>Write on owner (EXCLUSIVE): Becomes MODIFIED</li>
 *   <li>Write on owner (SHARED): Other copies INVALID, owner becomes EXCLUSIVE</li>
 *   <li>Write on non-owner: Transfer ownership, old copies INVALID, new becomes EXCLUSIVE</li>
 * </ul>
 */
public enum CoherenceState {

    /**
     * Data is valid and this is the only copy.
     * The device has exclusive ownership and can read/write without synchronization.
     */
    EXCLUSIVE,

    /**
     * Data is valid and shared with other devices.
     * Read operations are allowed, but write requires invalidating other copies.
     */
    SHARED,

    /**
     * Data has been modified and this is the only valid copy.
     * Must be written back before other devices can access.
     * Semantically similar to EXCLUSIVE but indicates pending writeback.
     */
    MODIFIED,

    /**
     * Data is stale or not present.
     * Must be fetched from a valid copy before use.
     */
    INVALID;

    /**
     * Check if data in this state is valid (readable)
     * @return true if data can be read
     */
    public boolean isValid() {
        return this != INVALID;
    }

    /**
     * Check if data in this state can be written without synchronization
     * @return true if write is allowed without sync
     */
    public boolean canWrite() {
        return this == EXCLUSIVE || this == MODIFIED;
    }

    /**
     * Check if data in this state is the authoritative copy
     * @return true if this is the owner copy
     */
    public boolean isOwner() {
        return this == EXCLUSIVE || this == MODIFIED;
    }

    /**
     * Check if data needs to be written back before invalidation
     * @return true if writeback is needed
     */
    public boolean needsWriteback() {
        return this == MODIFIED;
    }

    /**
     * Get the state after a read operation on this copy
     * @return the new state (unchanged for valid states)
     */
    public CoherenceState afterRead() {
        return this;  // Reading doesn't change state
    }

    /**
     * Get the state after a write operation on this copy
     * @return MODIFIED if was valid, INVALID otherwise
     */
    public CoherenceState afterWrite() {
        if (this.isValid()) {
            return MODIFIED;
        }
        return INVALID;  // Can't write to invalid
    }

    /**
     * Get the state after another device reads from a valid copy
     * @return SHARED if was EXCLUSIVE, unchanged otherwise
     */
    public CoherenceState afterRemoteRead() {
        if (this == EXCLUSIVE) {
            return SHARED;
        }
        return this;
    }

    /**
     * Get the state after another device writes
     * @return INVALID (our copy is now stale)
     */
    public CoherenceState afterRemoteWrite() {
        return INVALID;
    }

    /**
     * Get the state for a newly created copy from a valid source
     * @return SHARED
     */
    public static CoherenceState forNewCopy() {
        return SHARED;
    }

    /**
     * Get the state for freshly allocated memory
     * @return EXCLUSIVE (no other copies exist)
     */
    public static CoherenceState forNewAllocation() {
        return EXCLUSIVE;
    }
}

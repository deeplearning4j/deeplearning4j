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

package org.eclipse.deeplearning4j.llm.generation;

import lombok.Getter;

/**
 * Result of a radix prefix cache lookup.
 *
 * <p>Encapsulates the outcome of searching the cross-request prefix cache
 * for a matching token prefix. Contains the number of tokens matched,
 * the shared block IDs that can be reused, and an estimate of memory saved.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see RadixPrefixCache
 */
public class PrefixLookupResult {

    /** Number of tokens matched in the prefix. */
    @Getter
    private final int matchedTokenCount;

    /** Physical block IDs that can be reused from the cache. */
    @Getter
    private final int[] sharedBlockIds;

    /** Estimated memory saved in bytes by sharing these blocks. */
    @Getter
    private final long memorySavedBytes;

    /**
     * Create a prefix lookup result.
     *
     * @param matchedTokenCount number of tokens matched
     * @param sharedBlockIds    block IDs that can be reused
     * @param memorySavedBytes  estimated memory savings in bytes
     */
    public PrefixLookupResult(int matchedTokenCount, int[] sharedBlockIds, long memorySavedBytes) {
        this.matchedTokenCount = matchedTokenCount;
        this.sharedBlockIds = sharedBlockIds;
        this.memorySavedBytes = memorySavedBytes;
    }

    /**
     * Create an empty result (no match found).
     */
    public static PrefixLookupResult noMatch() {
        return new PrefixLookupResult(0, new int[0], 0);
    }

    /**
     * Whether any prefix was matched.
     */
    public boolean hasMatch() {
        return matchedTokenCount > 0 && sharedBlockIds.length > 0;
    }

    @Override
    public String toString() {
        return String.format("PrefixLookupResult[matched=%d tokens, blocks=%d, saved=%d bytes]",
                matchedTokenCount, sharedBlockIds.length, memorySavedBytes);
    }
}

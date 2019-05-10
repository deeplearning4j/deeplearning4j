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

package org.deeplearning4j.datasets.iterator.parallel;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

/**
 * This is utility class, that allows easy handling of multiple joint boolean states.
 *
 * PLEASE NOTE: It's suited for tracking up to 32 states in total.
 * PLEASE NOTE: This class is NOT thread safe
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class MultiBoolean {
    private final int numEntries;
    private int holder = 0;
    private int max = 0;
    private boolean oneTime;
    private MultiBoolean timeTracker;

    public MultiBoolean(int numEntries) {
        this(numEntries, false);
    }

    public MultiBoolean(int numEntries, boolean initialValue) {
        this(numEntries, initialValue, false);
    }

    public MultiBoolean(int numEntries, boolean initialValue, boolean oneTime) {
        if (numEntries > 32)
            throw new UnsupportedOperationException("Up to 32 entries can be tracked at once.");

        this.oneTime = oneTime;
        this.numEntries = numEntries;
        for (int i = 1; i <= numEntries; i++) {
            this.max |= 1 << i;
        }

        if (initialValue)
            this.holder = this.max;

        if (oneTime)
            this.timeTracker = new MultiBoolean(numEntries, false, false);
    }

    /**
     * Sets specified entry to specified state
     *
     * @param value
     * @param entry
     */
    public void set(boolean value, int entry) {
        if (entry > numEntries || entry < 0)
            throw new ND4JIllegalStateException(
                            "Entry index given (" + entry + ")in is higher then configured one (" + numEntries + ")");

        if (oneTime && this.timeTracker.get(entry))
            return;

        if (value)
            this.holder |= 1 << (entry + 1);
        else
            this.holder &= ~(1 << (entry + 1));

        if (oneTime)
            this.timeTracker.set(true, entry);
    }

    /**
     * Gets current state for specified entry
     *
     * @param entry
     * @return
     */
    public boolean get(int entry) {
        if (entry > numEntries || entry < 0)
            throw new ND4JIllegalStateException(
                            "Entry index given (" + entry + ")in is higher then configured one (" + numEntries + ")");

        return (this.holder & 1 << (entry + 1)) != 0;
    }

    /**
     * This method returns true if ALL states are true. False otherwise.
     *
     * @return
     */
    public boolean allTrue() {
        //log.info("Holder: {}; Max: {}", holder, max);
        return holder == max;
    }

    /**
     * This method returns true if ALL states are false. False otherwise
     * @return
     */
    public boolean allFalse() {
        return holder == 0;
    }
}

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

package org.nd4j.profiler;

import lombok.Getter;
import org.nd4j.common.primitives.Counter;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Counter class for tracking memory allocation, deallocation, and GC activity.
 *
 * @author Adam Gibson
 */
public class MemoryCounter {

    @Getter
    private static Counter<String> allocated = new Counter<>();

    @Getter
    private static Counter<String> instanceCounts = new Counter<>();

    @Getter
    private static AtomicLong currMemory = new AtomicLong(0);

    public static void increment(String name, long size) {
        allocated.incrementCount(name, size);
        instanceCounts.incrementCount(name, 1);
    }

    public static void decrement(String name, long size) {
        allocated.incrementCount(name, -size);
        instanceCounts.incrementCount(name, -1);
    }


    /**
     * Record the current amount of memory used when a system.gc runs.
     * This {@link #currMemory} is meant to be sampled.
     * @param currMemory the current amount of memory used
     */
    public static void recordGC(long currMemory) {
        MemoryCounter.currMemory.set(currMemory);
    }
}
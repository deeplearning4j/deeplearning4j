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

package org.nd4j.linalg.profiler.data.primitives;

import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class StackTree {
    protected Map<String, StackNode> basement = new HashMap<>();
    protected AtomicLong eventsCount = new AtomicLong(0);
    protected Map<StackDescriptor, ComparableAtomicLong> branches = new HashMap<>();
    @Getter
    protected StackDescriptor lastDescriptor;

    public StackTree() {
        //
    }

    public String renderTree(boolean displayCounts) {
        StringBuilder builder = new StringBuilder();

        // we'll always have single entry here, but let's keep loop here
        for (StackNode cNode : basement.values()) {
            cNode.traverse(0, displayCounts);
        }

        return builder.toString();
    }

    public void consumeStackTrace(@NonNull StackDescriptor descriptor) {
        consumeStackTrace(descriptor, 1);
    }

    public void consumeStackTrace(@NonNull StackDescriptor descriptor, long increment) {
        eventsCount.incrementAndGet();

        lastDescriptor = descriptor;

        if (!branches.containsKey(descriptor))
            branches.put(descriptor, new ComparableAtomicLong(0));

        branches.get(descriptor).incrementAndGet();

        String entry = descriptor.getEntryName();
        if (!basement.containsKey(entry))
            basement.put(entry, new StackNode(entry));

        // propagate stack trace across tree
        basement.get(entry).consume(descriptor, 0, increment);
    }

    public long getTotalEventsNumber() {
        return eventsCount.get();
    }

    public int getUniqueBranchesNumber() {
        return branches.size();
    }

    public void reset() {
        basement.clear();
        eventsCount.set(0);
        branches.clear();
    }
}

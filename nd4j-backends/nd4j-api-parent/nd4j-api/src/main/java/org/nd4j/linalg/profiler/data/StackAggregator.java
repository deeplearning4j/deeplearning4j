package org.nd4j.linalg.profiler.data;

import org.nd4j.linalg.profiler.data.primitives.ComparableAtomicLong;
import org.nd4j.linalg.profiler.data.primitives.StackDescriptor;
import org.nd4j.linalg.profiler.data.primitives.StackTree;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * This is utility class, provides stack traces collection, used in OpProfiler, to count events occurrences based on their position in code
 *
 *
 * @author raver119@gmail.com
 */
public class StackAggregator {
    private StackTree tree = new StackTree();

    public StackAggregator() {
        // nothing to do here so far
    }

    public void reset() {
        tree.reset();
    }

    public void incrementCount() {
        StackDescriptor descriptor = new StackDescriptor(Thread.currentThread().getStackTrace());
        tree.consumeStackTrace(descriptor);
    }

    public long getTotalEventsNumber() {
        return tree.getTotalEventsNumber();
    }

    public int getUniqueBranchesNumber() {
        return tree.getUniqueBranchesNumber();
    }
}

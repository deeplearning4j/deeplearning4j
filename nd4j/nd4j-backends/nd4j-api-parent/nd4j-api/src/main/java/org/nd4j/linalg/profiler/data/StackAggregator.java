package org.nd4j.linalg.profiler.data;

import org.nd4j.linalg.profiler.data.primitives.StackDescriptor;
import org.nd4j.linalg.profiler.data.primitives.StackTree;

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

    public void renderTree() {
        tree.renderTree(false);
    }

    public void renderTree(boolean displayCounts) {
        tree.renderTree(displayCounts);
    }

    public void reset() {
        tree.reset();
    }

    public void incrementCount() {
        incrementCount(1);
    }

    public void incrementCount(long time) {
        StackDescriptor descriptor = new StackDescriptor(Thread.currentThread().getStackTrace());
        tree.consumeStackTrace(descriptor, time);
    }

    public long getTotalEventsNumber() {
        return tree.getTotalEventsNumber();
    }

    public int getUniqueBranchesNumber() {
        return tree.getUniqueBranchesNumber();
    }

    public StackDescriptor getLastDescriptor() {
        return tree.getLastDescriptor();
    }
}

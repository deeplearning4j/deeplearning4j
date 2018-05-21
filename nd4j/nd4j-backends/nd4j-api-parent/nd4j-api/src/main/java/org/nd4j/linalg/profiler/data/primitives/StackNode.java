package org.nd4j.linalg.profiler.data.primitives;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class StackNode implements Comparable<StackNode> {
    private final String nodeURI;
    protected Map<String, StackNode> entries = new HashMap<>();
    protected AtomicLong counter = new AtomicLong(0);

    public StackNode(@NonNull String uri) {
        this.nodeURI = uri;
    }

    public Collection<StackNode> getNodes() {
        return entries.values();
    }

    public void traverse(int ownLevel, boolean displayCounts) {
        StringBuilder builder = new StringBuilder();

        for (int x = 0; x < ownLevel; x++) {
            builder.append("   ");
        }

        builder.append("").append(nodeURI);

        if (displayCounts)
            builder.append("  ").append(counter.get()).append(" us");

        System.out.println(builder.toString());

        for (StackNode node : entries.values()) {
            node.traverse(ownLevel + 1, displayCounts);
        }
    }

    public void consume(@NonNull StackDescriptor descriptor, int lastLevel) {
        consume(descriptor, lastLevel, 1);
    }

    public void consume(@NonNull StackDescriptor descriptor, int lastLevel, long delta) {
        boolean gotEntry = false;
        for (int e = 0; e < descriptor.size(); e++) {
            String entryName = descriptor.getElementName(e);

            // we look for current entry first
            if (!gotEntry) {
                if (entryName.equalsIgnoreCase(nodeURI) && e >= lastLevel) {
                    gotEntry = true;
                    counter.addAndGet(delta);
                }
            } else {
                // after current entry is found, we just fill first node after it
                if (!entries.containsKey(entryName))
                    entries.put(entryName, new StackNode(entryName));

                entries.get(entryName).consume(descriptor, e);
                break;
            }
        }
    }

    @Override
    public int compareTo(StackNode o) {
        return Long.compare(o.counter.get(), this.counter.get());
    }
}

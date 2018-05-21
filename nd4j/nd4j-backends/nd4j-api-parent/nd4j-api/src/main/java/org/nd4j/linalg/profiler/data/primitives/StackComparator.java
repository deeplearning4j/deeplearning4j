package org.nd4j.linalg.profiler.data.primitives;

import java.util.Comparator;

/**
 * @author raver119@gmail.com
 */
public class StackComparator implements Comparator<StackNode> {

    @Override
    public int compare(StackNode o1, StackNode o2) {
        return o1.compareTo(o2);
    }
}

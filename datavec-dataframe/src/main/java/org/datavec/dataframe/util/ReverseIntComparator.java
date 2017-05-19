package org.datavec.dataframe.util;

import it.unimi.dsi.fastutil.ints.IntComparator;
import it.unimi.dsi.fastutil.longs.LongComparator;

import net.jcip.annotations.Immutable;

/**
 * A Comparator for int primitives for sorting in reverse order, using the given comparator
 */
@Immutable
public final class ReverseIntComparator {

    static final IntComparator reverseIntComparator = new IntComparator() {

        @Override
        public int compare(int o2, int o1) {
            return (o1 < o2 ? -1 : (o1 == o2) ? 0 : 1);
        }

        @Override
        public int compare(Integer o2, Integer o1) {
            return (o1 < o2 ? -1 : (o1.equals(o2) ? 0 : 1));
        }
    };

    public static IntComparator instance() {
        return reverseIntComparator;
    }

}

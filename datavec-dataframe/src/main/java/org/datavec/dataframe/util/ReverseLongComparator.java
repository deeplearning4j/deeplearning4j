package org.datavec.dataframe.util;

import it.unimi.dsi.fastutil.longs.LongComparator;

import net.jcip.annotations.Immutable;

/**
 * A comparator for long primitives for sorting in descending order
 */
@Immutable
public final class ReverseLongComparator {

    static final LongComparator reverseLongComparator = new LongComparator() {

        @Override
        public int compare(Long o2, Long o1) {
            return (o1 < o2 ? -1 : (o1.equals(o2) ? 0 : 1));
        }

        @Override
        public int compare(long o2, long o1) {
            return (o1 < o2 ? -1 : (o1 == o2 ? 0 : 1));
        }
    };

    public static LongComparator instance() {
        return reverseLongComparator;
    }

}

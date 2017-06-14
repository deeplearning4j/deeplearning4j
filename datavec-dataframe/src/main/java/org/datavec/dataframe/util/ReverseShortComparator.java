package org.datavec.dataframe.util;

import it.unimi.dsi.fastutil.shorts.ShortComparator;

import net.jcip.annotations.Immutable;

/**
 * A Comparator for int primitives for sorting in reverse order, using the given comparator
 */
@Immutable
public final class ReverseShortComparator {

    public static ShortComparator reverseShortComparator = new ShortComparator() {

        @Override
        public int compare(Short o2, Short o1) {
            return (o1 < o2 ? -1 : (o1.equals(o2) ? 0 : 1));
        }

        @Override
        public int compare(short o2, short o1) {
            return (o1 < o2 ? -1 : (o1 == o2 ? 0 : 1));
        }
    };

    public static ShortComparator instance() {
        return reverseShortComparator;
    }
}

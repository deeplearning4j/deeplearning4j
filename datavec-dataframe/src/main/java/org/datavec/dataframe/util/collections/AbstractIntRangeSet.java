package org.datavec.dataframe.util.collections;

import edu.umd.cs.findbugs.annotations.Nullable;

/**
 * A skeletal implementation of {@code IntRangeSet}.
 */
abstract class AbstractIntRangeSet implements IntRangeSet {
    AbstractIntRangeSet() {}

    @Override
    public boolean contains(int value) {
        return rangeContaining(value) != null;
    }

    @Override
    public abstract IntRange rangeContaining(int value);

    @Override
    public boolean isEmpty() {
        return asRanges().isEmpty();
    }

    @Override
    public void add(IntRange range) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void remove(IntRange range) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void clear() {
        remove(IntRange.all());
    }

    @Override
    public boolean enclosesAll(IntRangeSet other) {
        for (IntRange range : other.asRanges()) {
            if (!encloses(range)) {
                return false;
            }
        }
        return true;
    }

    @Override
    public void addAll(IntRangeSet other) {
        for (IntRange range : other.asRanges()) {
            add(range);
        }
    }

    @Override
    public void removeAll(IntRangeSet other) {
        for (IntRange range : other.asRanges()) {
            remove(range);
        }
    }

    @Override
    public boolean intersects(IntRange otherRange) {
        return !subRangeSet(otherRange).isEmpty();
    }

    @Override
    public abstract boolean encloses(IntRange otherRange);

    @Override
    public boolean equals(@Nullable Object obj) {
        if (obj == this) {
            return true;
        } else if (obj instanceof IntRangeSet) {
            IntRangeSet other = (IntRangeSet) obj;
            return this.asRanges().equals(other.asRanges());
        }
        return false;
    }

    @Override
    public final int hashCode() {
        return asRanges().hashCode();
    }

    @Override
    public final String toString() {
        return asRanges().toString();
    }
}

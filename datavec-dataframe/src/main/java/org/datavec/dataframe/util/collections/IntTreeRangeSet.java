package org.datavec.dataframe.util.collections;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.MoreObjects;
import com.google.common.collect.AbstractIterator;
import com.google.common.collect.BoundType;
import com.google.common.collect.ForwardingCollection;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Iterators;
import com.google.common.collect.Maps;
import com.google.common.collect.Ordering;
import com.google.common.collect.PeekingIterator;

import java.util.Collection;
import java.util.Comparator;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.NavigableMap;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.TreeMap;

import edu.umd.cs.findbugs.annotations.Nullable;

/**
 * An implementation of {@link IntRangeSet} backed by a {@link TreeMap}.
 */
public class IntTreeRangeSet extends AbstractIntRangeSet {

    final NavigableMap<IntCut, IntRange> rangesByLowerBound;

    /**
     * Creates an empty {@code IntTreeRangeSet} instance.
     */
    public static IntTreeRangeSet create() {
        return new IntTreeRangeSet(new TreeMap<>());
    }

    /**
     * Returns a {@code IntTreeRangeSet} initialized with the ranges in the specified range set.
     */
    public static IntTreeRangeSet create(IntRangeSet rangeSet) {
        IntTreeRangeSet result = create();
        result.addAll(rangeSet);
        return result;
    }

    private IntTreeRangeSet(NavigableMap<IntCut, IntRange> rangesByLowerCut) {
        this.rangesByLowerBound = rangesByLowerCut;
    }

    private transient Set<IntRange> asRanges;

    @Override
    public Set<IntRange> asRanges() {
        Set<IntRange> result = asRanges;
        return (result == null) ? asRanges = new AsRanges() : result;
    }

    final class AsRanges extends ForwardingCollection<IntRange> implements Set<IntRange> {
        @Override
        protected Collection<IntRange> delegate() {
            return rangesByLowerBound.values();
        }
    }

    @Override
    @Nullable
    public IntRange rangeContaining(int value) {
        checkNotNull(value);
        Entry<IntCut, IntRange> floorEntry = rangesByLowerBound.floorEntry(IntCut.belowValue(value));
        if (floorEntry != null && floorEntry.getValue().contains(value)) {
            return floorEntry.getValue();
        } else {
            return null;
        }
    }

    @Override
    public boolean encloses(IntRange range) {
        checkNotNull(range);
        Entry<IntCut, IntRange> floorEntry = rangesByLowerBound.floorEntry(range.lowerBound);
        return floorEntry != null && floorEntry.getValue().encloses(range);
    }

    @Nullable
    private IntRange rangeEnclosing(IntRange range) {
        checkNotNull(range);
        Entry<IntCut, IntRange> floorEntry = rangesByLowerBound.floorEntry(range.lowerBound);
        return (floorEntry != null && floorEntry.getValue().encloses(range)) ? floorEntry.getValue() : null;
    }

    @Override
    public IntRange span() {
        Entry<IntCut, IntRange> firstEntry = rangesByLowerBound.firstEntry();
        Entry<IntCut, IntRange> lastEntry = rangesByLowerBound.lastEntry();
        if (firstEntry == null) {
            throw new NoSuchElementException();
        }
        return IntRange.create(firstEntry.getValue().lowerBound, lastEntry.getValue().upperBound);
    }

    @Override
    public void add(IntRange rangeToAdd) {
        checkNotNull(rangeToAdd);

        if (rangeToAdd.isEmpty()) {
            return;
        }

        // We will use { } to illustrate ranges currently in the range set, and < >
        // to illustrate rangeToAdd.
        IntCut lbToAdd = rangeToAdd.lowerBound;
        IntCut ubToAdd = rangeToAdd.upperBound;

        Entry<IntCut, IntRange> entryBelowLB = rangesByLowerBound.lowerEntry(lbToAdd);
        if (entryBelowLB != null) {
            // { <
            IntRange rangeBelowLB = entryBelowLB.getValue();
            if (rangeBelowLB.upperBound.compareTo(lbToAdd) >= 0) {
                // { < }, and we will need to coalesce
                if (rangeBelowLB.upperBound.compareTo(ubToAdd) >= 0) {
                    // { < > }
                    ubToAdd = rangeBelowLB.upperBound;
                    /*
                     * TODO(cpovirk): can we just "return;" here? Or, can we remove this if() entirely? If
                     * not, add tests to demonstrate the problem with each approach
                     */
                }
                lbToAdd = rangeBelowLB.lowerBound;
            }
        }

        Entry<IntCut, IntRange> entryBelowUB = rangesByLowerBound.floorEntry(ubToAdd);
        if (entryBelowUB != null) {
            // { >
            IntRange rangeBelowUB = entryBelowUB.getValue();
            if (rangeBelowUB.upperBound.compareTo(ubToAdd) >= 0) {
                // { > }, and we need to coalesce
                ubToAdd = rangeBelowUB.upperBound;
            }
        }

        // Remove ranges which are strictly enclosed.
        rangesByLowerBound.subMap(lbToAdd, ubToAdd).clear();

        replaceRangeWithSameLowerBound(IntRange.create(lbToAdd, ubToAdd));
    }

    @Override
    public void remove(IntRange rangeToRemove) {
        checkNotNull(rangeToRemove);

        if (rangeToRemove.isEmpty()) {
            return;
        }

        // We will use { } to illustrate ranges currently in the range set, and < >
        // to illustrate rangeToRemove.

        Entry<IntCut, IntRange> entryBelowLB = rangesByLowerBound.lowerEntry(rangeToRemove.lowerBound);
        if (entryBelowLB != null) {
            // { <
            IntRange rangeBelowLB = entryBelowLB.getValue();
            if (rangeBelowLB.upperBound.compareTo(rangeToRemove.lowerBound) >= 0) {
                // { < }, and we will need to subdivide
                if (rangeToRemove.hasUpperBound() && rangeBelowLB.upperBound.compareTo(rangeToRemove.upperBound) >= 0) {
                    // { < > }
                    replaceRangeWithSameLowerBound(IntRange.create(rangeToRemove.upperBound, rangeBelowLB.upperBound));
                }
                replaceRangeWithSameLowerBound(IntRange.create(rangeBelowLB.lowerBound, rangeToRemove.lowerBound));
            }
        }

        Entry<IntCut, IntRange> entryBelowUB = rangesByLowerBound.floorEntry(rangeToRemove.upperBound);
        if (entryBelowUB != null) {
            // { >
            IntRange rangeBelowUB = entryBelowUB.getValue();
            if (rangeToRemove.hasUpperBound() && rangeBelowUB.upperBound.compareTo(rangeToRemove.upperBound) >= 0) {
                // { > }
                replaceRangeWithSameLowerBound(IntRange.create(rangeToRemove.upperBound, rangeBelowUB.upperBound));
            }
        }

        rangesByLowerBound.subMap(rangeToRemove.lowerBound, rangeToRemove.upperBound).clear();
    }

    private void replaceRangeWithSameLowerBound(IntRange range) {
        if (range.isEmpty()) {
            rangesByLowerBound.remove(range.lowerBound);
        } else {
            rangesByLowerBound.put(range.lowerBound, range);
        }
    }

    private transient IntRangeSet complement;

    @Override
    public IntRangeSet complement() {
        IntRangeSet result = complement;
        return (result == null) ? complement = new Complement() : result;
    }

    private static abstract class BaseRangeByBound extends TreeMap<IntCut, IntRange> {
        protected NavigableMap<IntCut, IntRange> rangesByLowerBound;
        /**
         * upperBoundWindow represents the headMap/subMap/tailMap view of the entire "ranges by upper
         * bound" map; it's a constraint on the *keys*, and does not affect the values.
         */
        protected IntRange upperBoundWindow;

        private NavigableMap<IntCut, IntRange> subMap(IntRange window) {
            if (window.isConnected(upperBoundWindow)) {
                return new IntTreeRangeSet.RangesByUpperBound(rangesByLowerBound,
                                window.intersection(upperBoundWindow));
            } else {
                return ImmutableSortedMap.of();
            }
        }

        @Override
        public NavigableMap<IntCut, IntRange> subMap(IntCut fromKey, boolean fromInclusive, IntCut toKey,
                        boolean toInclusive) {
            return subMap(IntRange.range(fromKey.endpoint(), IntTreeRangeSet.boundTypeForBoolean(fromInclusive),
                            toKey.endpoint(), IntTreeRangeSet.boundTypeForBoolean(toInclusive)));
        }

        @Override
        public NavigableMap<IntCut, IntRange> headMap(IntCut toKey, boolean inclusive) {
            return subMap(IntRange.upTo(toKey.endpoint(), IntTreeRangeSet.boundTypeForBoolean(inclusive)));
        }

        @Override
        public NavigableMap<IntCut, IntRange> tailMap(IntCut fromKey, boolean inclusive) {
            return subMap(IntRange.downTo(fromKey.endpoint(), IntTreeRangeSet.boundTypeForBoolean(inclusive)));
        }

        @Override
        public Comparator<? super IntCut> comparator() {
            return Ordering.<IntCut>natural();
        }

        @Override
        public boolean containsKey(@Nullable Object key) {
            return get(key) != null;
        }

        @Override
        public abstract IntRange get(@Nullable Object key);
    }

    static final class RangesByUpperBound extends BaseRangeByBound {

        RangesByUpperBound(NavigableMap<IntCut, IntRange> rangesByLowerBound) {
            this.rangesByLowerBound = rangesByLowerBound;
            this.upperBoundWindow = IntRange.all();
        }

        private RangesByUpperBound(NavigableMap<IntCut, IntRange> rangesByLowerBound, IntRange upperBoundWindow) {
            this.rangesByLowerBound = rangesByLowerBound;
            this.upperBoundWindow = upperBoundWindow;
        }

        @Override
        public IntRange get(@Nullable Object key) {
            if (key instanceof IntCut) {
                try {
                    @SuppressWarnings("unchecked") // we catch CCEs
                    IntCut cut = (IntCut) key;
                    if (!upperBoundWindow.contains(cut.endpoint())) {
                        return null;
                    }
                    Entry<IntCut, IntRange> candidate = rangesByLowerBound.lowerEntry(cut);
                    if (candidate != null && candidate.getValue().upperBound.equals(cut)) {
                        return candidate.getValue();
                    }
                } catch (ClassCastException e) {
                    return null;
                }
            }
            return null;
        }

        Iterator<Entry<IntCut, IntRange>> entryIterator() {
            /*
             * We want to start the iteration at the first range where the upper bound is in
             * upperBoundWindow.
             */
            final Iterator<IntRange> backingItr;
            if (!upperBoundWindow.hasLowerBound()) {
                backingItr = rangesByLowerBound.values().iterator();
            } else {
                Entry<IntCut, IntRange> lowerEntry = rangesByLowerBound.lowerEntry(upperBoundWindow.lowerBound());
                if (lowerEntry == null) {
                    backingItr = rangesByLowerBound.values().iterator();
                } else if (upperBoundWindow.lowerBound.isLessThan(lowerEntry.getValue().upperEndpoint())) {
                    backingItr = rangesByLowerBound.tailMap(lowerEntry.getKey(), true).values().iterator();
                } else {
                    backingItr = rangesByLowerBound.tailMap(upperBoundWindow.lowerBound(), true).values().iterator();
                }
            }
            return new AbstractIterator<Entry<IntCut, IntRange>>() {
                @Override
                protected Entry<IntCut, IntRange> computeNext() {
                    if (!backingItr.hasNext()) {
                        return endOfData();
                    }
                    IntRange range = backingItr.next();
                    if (upperBoundWindow.upperBound.isLessThan(range.upperEndpoint())) {
                        return endOfData();
                    } else {
                        return Maps.immutableEntry(range.upperBound, range);
                    }
                }
            };
        }

        Iterator<Entry<IntCut, IntRange>> descendingEntryIterator() {
            Collection<IntRange> candidates;
            if (upperBoundWindow.hasUpperBound()) {
                candidates = rangesByLowerBound.headMap(upperBoundWindow.upperBound(), false).descendingMap().values();
            } else {
                candidates = rangesByLowerBound.descendingMap().values();
            }
            final PeekingIterator<IntRange> backingItr = Iterators.peekingIterator(candidates.iterator());
            if (backingItr.hasNext() && upperBoundWindow.upperBound.isLessThan(backingItr.peek().upperEndpoint())) {
                backingItr.next();
            }
            return new AbstractIterator<Entry<IntCut, IntRange>>() {
                @Override
                protected Entry<IntCut, IntRange> computeNext() {
                    if (!backingItr.hasNext()) {
                        return endOfData();
                    }
                    IntRange range = backingItr.next();
                    return upperBoundWindow.lowerBound.isLessThan(range.upperEndpoint())
                                    ? Maps.immutableEntry(range.upperBound, range) : endOfData();
                }
            };
        }

        @Override
        public int size() {
            if (upperBoundWindow.equals(IntRange.all())) {
                return rangesByLowerBound.size();
            }
            return Iterators.size(entryIterator());
        }

        @Override
        public boolean isEmpty() {
            return upperBoundWindow.equals(IntRange.all()) ? rangesByLowerBound.isEmpty() : !entryIterator().hasNext();
        }
    }

    private static final class ComplementRangesByLowerBound extends BaseRangeByBound {
        private final NavigableMap<IntCut, IntRange> positiveRangesByLowerBound;
        private final NavigableMap<IntCut, IntRange> positiveRangesByUpperBound;

        /**
         * complementLowerBoundWindow represents the headMap/subMap/tailMap view of the entire
         * "complement ranges by lower bound" map; it's a constraint on the *keys*, and does not affect
         * the values.
         */
        private final IntRange complementLowerBoundWindow;

        ComplementRangesByLowerBound(NavigableMap<IntCut, IntRange> positiveRangesByLowerBound) {
            this(positiveRangesByLowerBound, IntRange.all());
        }

        private ComplementRangesByLowerBound(NavigableMap<IntCut, IntRange> positiveRangesByLowerBound,
                        IntRange window) {
            this.positiveRangesByLowerBound = positiveRangesByLowerBound;
            this.positiveRangesByUpperBound = new RangesByUpperBound(positiveRangesByLowerBound);
            this.complementLowerBoundWindow = window;
        }

        private NavigableMap<IntCut, IntRange> subMap(IntRange subWindow) {
            if (!complementLowerBoundWindow.isConnected(subWindow)) {
                return ImmutableSortedMap.of();
            } else {
                subWindow = subWindow.intersection(complementLowerBoundWindow);
                return new ComplementRangesByLowerBound(positiveRangesByLowerBound, subWindow);
            }
        }

        Iterator<Entry<IntCut, IntRange>> entryIterator() {
            /*
             * firstComplementRangeLowerBound is the first complement range lower bound inside
             * complementLowerBoundWindow. Complement range lower bounds are either positive range upper
             * bounds, or IntCut.belowAll().
             *
             * positiveItr starts at the first positive range with lower bound greater than
             * firstComplementRangeLowerBound. (Positive range lower bounds correspond to complement range
             * upper bounds.)
             */
            Collection<IntRange> positiveRanges;
            if (complementLowerBoundWindow.hasLowerBound()) {
                positiveRanges = positiveRangesByUpperBound
                                .tailMap(complementLowerBoundWindow.lowerBound(),
                                                complementLowerBoundWindow.lowerBoundType() == BoundType.CLOSED)
                                .values();
            } else {
                positiveRanges = positiveRangesByUpperBound.values();
            }
            final PeekingIterator<IntRange> positiveItr = Iterators.peekingIterator(positiveRanges.iterator());
            final IntCut firstComplementRangeLowerBound;
            if (complementLowerBoundWindow.contains(IntCut.belowAll().endpoint())
                            && (!positiveItr.hasNext() || positiveItr.peek().lowerBound != IntCut.belowAll())) {
                firstComplementRangeLowerBound = IntCut.belowAll();
            } else if (positiveItr.hasNext()) {
                firstComplementRangeLowerBound = positiveItr.next().upperBound;
            } else {
                return Iterators.emptyIterator();
            }
            return new AbstractIterator<Entry<IntCut, IntRange>>() {
                IntCut nextComplementRangeLowerBound = firstComplementRangeLowerBound;

                @Override
                protected Entry<IntCut, IntRange> computeNext() {
                    if (complementLowerBoundWindow.upperBound.isLessThan(nextComplementRangeLowerBound.endpoint())
                                    || nextComplementRangeLowerBound == IntCut.aboveAll()) {
                        return endOfData();
                    }
                    IntRange negativeRange;
                    if (positiveItr.hasNext()) {
                        IntRange positiveRange = positiveItr.next();
                        negativeRange = IntRange.create(nextComplementRangeLowerBound, positiveRange.lowerBound);
                        nextComplementRangeLowerBound = positiveRange.upperBound;
                    } else {
                        negativeRange = IntRange.create(nextComplementRangeLowerBound, IntCut.aboveAll());
                        nextComplementRangeLowerBound = IntCut.aboveAll();
                    }
                    return Maps.immutableEntry(negativeRange.lowerBound, negativeRange);
                }
            };
        }

        Iterator<Entry<IntCut, IntRange>> descendingEntryIterator() {

            /*
             * firstComplementRangeUpperBound is the upper bound of the last complement range with lower
             * bound inside complementLowerBoundWindow.
             *
             * positiveItr starts at the first positive range with upper bound less than
             * firstComplementRangeUpperBound. (Positive range upper bounds correspond to complement range
             * lower bounds.)
             */
            IntCut startingPoint = complementLowerBoundWindow.hasUpperBound() ? complementLowerBoundWindow.upperBound()
                            : IntCut.aboveAll();
            boolean inclusive = complementLowerBoundWindow.hasUpperBound()
                            && complementLowerBoundWindow.upperBoundType() == BoundType.CLOSED;
            final PeekingIterator<IntRange> positiveItr = Iterators.peekingIterator(positiveRangesByUpperBound
                            .headMap(startingPoint, inclusive).descendingMap().values().iterator());
            IntCut cut;
            if (positiveItr.hasNext()) {
                cut = (positiveItr.peek().upperBound == IntCut.aboveAll()) ? positiveItr.next().lowerBound
                                : positiveRangesByLowerBound.higherKey(positiveItr.peek().upperBound);
            } else if (!complementLowerBoundWindow.contains(IntCut.belowAll().endpoint())
                            || positiveRangesByLowerBound.containsKey(IntCut.belowAll())) {
                return Iterators.emptyIterator();
            } else {
                cut = positiveRangesByLowerBound.higherKey(IntCut.belowAll());
            }
            final IntCut firstComplementRangeUpperBound = MoreObjects.firstNonNull(cut, IntCut.aboveAll());
            return new AbstractIterator<Entry<IntCut, IntRange>>() {
                IntCut nextComplementRangeUpperBound = firstComplementRangeUpperBound;

                @Override
                protected Entry<IntCut, IntRange> computeNext() {
                    if (nextComplementRangeUpperBound == IntCut.belowAll()) {
                        return endOfData();
                    } else if (positiveItr.hasNext()) {
                        IntRange positiveRange = positiveItr.next();
                        IntRange negativeRange =
                                        IntRange.create(positiveRange.upperBound, nextComplementRangeUpperBound);
                        nextComplementRangeUpperBound = positiveRange.lowerBound;
                        if (complementLowerBoundWindow.lowerBound.isLessThan(negativeRange.lowerEndpoint())) {
                            return Maps.immutableEntry(negativeRange.lowerBound, negativeRange);
                        }
                    } else if (complementLowerBoundWindow.lowerBound.isLessThan(IntCut.belowAll().endpoint())) {
                        IntRange negativeRange = IntRange.create(IntCut.belowAll(), nextComplementRangeUpperBound);
                        nextComplementRangeUpperBound = IntCut.belowAll();
                        return Maps.immutableEntry(IntCut.belowAll(), negativeRange);
                    }
                    return endOfData();
                }
            };
        }

        @Override
        public int size() {
            return Iterators.size(entryIterator());
        }

        @Override
        @Nullable
        public IntRange get(Object key) {
            if (key instanceof IntCut) {
                try {
                    @SuppressWarnings("unchecked")
                    IntCut cut = (IntCut) key;
                    // tailMap respects the current window
                    Entry<IntCut, IntRange> firstEntry = tailMap(cut, true).firstEntry();
                    if (firstEntry != null && firstEntry.getKey().equals(cut)) {
                        return firstEntry.getValue();
                    }
                } catch (ClassCastException e) {
                    return null;
                }
            }
            return null;
        }
    }

    private final class Complement extends IntTreeRangeSet {
        Complement() {
            super(new ComplementRangesByLowerBound(IntTreeRangeSet.this.rangesByLowerBound));
        }

        @Override
        public void add(IntRange rangeToAdd) {
            IntTreeRangeSet.this.remove(rangeToAdd);
        }

        @Override
        public void remove(IntRange rangeToRemove) {
            IntTreeRangeSet.this.add(rangeToRemove);
        }

        @Override
        public boolean contains(int value) {
            return !IntTreeRangeSet.this.contains(value);
        }

        @Override
        public IntRangeSet complement() {
            return IntTreeRangeSet.this;
        }
    }

    private static final class SubRangeSetRangesByLowerBound extends BaseRangeByBound {
        /**
         * lowerBoundWindow is the headMap/subMap/tailMap view; it only restricts the keys, and does not
         * affect the values.
         */
        private final IntRange lowerBoundWindow;

        /**
         * restriction is the subRangeSet view; ranges are truncated to their intersection with
         * restriction.
         */
        private final IntRange restriction;

        private final NavigableMap<IntCut, IntRange> rangesByLowerBound;
        private final NavigableMap<IntCut, IntRange> rangesByUpperBound;

        private SubRangeSetRangesByLowerBound(IntRange lowerBoundWindow, IntRange restriction,
                        NavigableMap<IntCut, IntRange> rangesByLowerBound) {
            this.lowerBoundWindow = checkNotNull(lowerBoundWindow);
            this.restriction = checkNotNull(restriction);
            this.rangesByLowerBound = checkNotNull(rangesByLowerBound);
            this.rangesByUpperBound = new RangesByUpperBound(rangesByLowerBound);
        }

        private NavigableMap<IntCut, IntRange> subMap(IntRange window) {
            if (!window.isConnected(lowerBoundWindow)) {
                return ImmutableSortedMap.of();
            } else {
                return new SubRangeSetRangesByLowerBound(lowerBoundWindow.intersection(window), restriction,
                                rangesByLowerBound);
            }
        }

        @Override
        @Nullable
        public IntRange get(@Nullable Object key) {
            if (key instanceof IntCut) {
                try {
                    @SuppressWarnings("unchecked") // we catch CCE's
                    IntCut cut = (IntCut) key;
                    if (!lowerBoundWindow.contains(cut.endpoint()) || cut.compareTo(restriction.lowerBound) < 0
                                    || cut.compareTo(restriction.upperBound) >= 0) {
                        return null;
                    } else if (cut.equals(restriction.lowerBound)) {
                        // it might be present, truncated on the left
                        Map.Entry<IntCut, IntRange> entry = rangesByLowerBound.floorEntry(cut);
                        IntRange candidate = valueOrNull(entry);
                        if (candidate != null && candidate.upperBound.compareTo(restriction.lowerBound) > 0) {
                            return candidate.intersection(restriction);
                        }
                    } else {
                        IntRange result = rangesByLowerBound.get(cut);
                        if (result != null) {
                            return result.intersection(restriction);
                        }
                    }
                } catch (ClassCastException e) {
                    return null;
                }
            }
            return null;
        }

        Iterator<Entry<IntCut, IntRange>> entryIterator() {
            if (restriction.isEmpty()) {
                return Iterators.emptyIterator();
            }
            final Iterator<IntRange> completeRangeItr;
            if (lowerBoundWindow.upperBound.isLessThan(restriction.lowerEndpoint())) {
                return Iterators.emptyIterator();
            } else if (lowerBoundWindow.lowerBound.isLessThan(restriction.lowerEndpoint())) {
                // starts at the first range with upper bound strictly greater than restriction.lowerBound
                completeRangeItr = rangesByUpperBound.tailMap(restriction.lowerBound, false).values().iterator();
            } else {
                // starts at the first range with lower bound above lowerBoundWindow.lowerBound
                completeRangeItr = rangesByLowerBound
                                .tailMap(lowerBoundWindow.lowerBound(),
                                                lowerBoundWindow.lowerBoundType() == BoundType.CLOSED)
                                .values().iterator();
            }
            final IntCut upperBoundOnLowerBounds = Ordering.natural().min(lowerBoundWindow.upperBound,
                            IntCut.belowValue(restriction.upperEndpoint()));
            return new AbstractIterator<Entry<IntCut, IntRange>>() {
                @Override
                protected Entry<IntCut, IntRange> computeNext() {
                    if (!completeRangeItr.hasNext()) {
                        return endOfData();
                    }
                    IntRange nextRange = completeRangeItr.next();
                    if (upperBoundOnLowerBounds.isLessThan(nextRange.lowerEndpoint())) {
                        return endOfData();
                    } else {
                        nextRange = nextRange.intersection(restriction);
                        return Maps.immutableEntry(nextRange.lowerBound, nextRange);
                    }
                }
            };
        }

        Iterator<Entry<IntCut, IntRange>> descendingEntryIterator() {
            if (restriction.isEmpty()) {
                return Iterators.emptyIterator();
            }
            IntCut upperBoundOnLowerBounds = Ordering.natural().min(lowerBoundWindow.upperBound,
                            IntCut.belowValue(restriction.upperEndpoint()));
            final Iterator<IntRange> completeRangeItr =
                            rangesByLowerBound
                                            .headMap(upperBoundOnLowerBounds,
                                                            upperBoundOnLowerBounds
                                                                            .typeAsUpperBound() == BoundType.CLOSED)
                                            .descendingMap().values().iterator();
            return new AbstractIterator<Entry<IntCut, IntRange>>() {
                @Override
                protected Entry<IntCut, IntRange> computeNext() {
                    if (!completeRangeItr.hasNext()) {
                        return endOfData();
                    }
                    IntRange nextRange = completeRangeItr.next();
                    if (restriction.lowerBound.compareTo(nextRange.upperBound) >= 0) {
                        return endOfData();
                    }
                    nextRange = nextRange.intersection(restriction);
                    if (lowerBoundWindow.contains(nextRange.lowerEndpoint())) {
                        return Maps.immutableEntry(nextRange.lowerBound, nextRange);
                    } else {
                        return endOfData();
                    }
                }
            };
        }

        @Override
        public int size() {
            return Iterators.size(entryIterator());
        }
    }

    @Override
    public IntRangeSet subRangeSet(IntRange view) {
        return view.equals(IntRange.all()) ? this : new SubRangeSetInt(view);
    }

    private final class SubRangeSetInt extends IntTreeRangeSet {
        private final IntRange restriction;

        SubRangeSetInt(IntRange restriction) {
            super(new SubRangeSetRangesByLowerBound(IntRange.<IntCut>all(), restriction,
                            IntTreeRangeSet.this.rangesByLowerBound));
            this.restriction = restriction;
        }

        @Override
        public boolean encloses(IntRange range) {
            if (!restriction.isEmpty() && restriction.encloses(range)) {
                IntRange enclosing = IntTreeRangeSet.this.rangeEnclosing(range);
                return enclosing != null && !enclosing.intersection(restriction).isEmpty();
            }
            return false;
        }

        @Override
        @Nullable
        public IntRange rangeContaining(int value) {
            if (!restriction.contains(value)) {
                return null;
            }
            IntRange result = IntTreeRangeSet.this.rangeContaining(value);
            return (result == null) ? null : result.intersection(restriction);
        }

        @Override
        public void add(IntRange rangeToAdd) {
            checkArgument(restriction.encloses(rangeToAdd), "Cannot add range %s to subRangeSet(%s)", rangeToAdd,
                            restriction);
            super.add(rangeToAdd);
        }

        @Override
        public void remove(IntRange rangeToRemove) {
            if (rangeToRemove.isConnected(restriction)) {
                IntTreeRangeSet.this.remove(rangeToRemove.intersection(restriction));
            }
        }

        @Override
        public boolean contains(int value) {
            return restriction.contains(value) && IntTreeRangeSet.this.contains(value);
        }

        @Override
        public void clear() {
            IntTreeRangeSet.this.remove(restriction);
        }

        @Override
        public IntRangeSet subRangeSet(IntRange view) {
            if (view.encloses(restriction)) {
                return this;
            } else if (view.isConnected(restriction)) {
                return new SubRangeSetInt(restriction.intersection(view));
            } else {
                return new IntTreeRangeSet(new TreeMap<>());
            }
        }
    }

    private static BoundType boundTypeForBoolean(boolean inclusive) {
        return inclusive ? BoundType.CLOSED : BoundType.OPEN;
    }

    private static IntRange valueOrNull(@Nullable Map.Entry<IntCut, IntRange> entry) {
        return entry == null ? null : entry.getValue();
    }

}

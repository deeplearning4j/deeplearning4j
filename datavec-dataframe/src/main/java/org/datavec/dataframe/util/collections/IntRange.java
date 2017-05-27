package org.datavec.dataframe.util.collections;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.Equivalence;
import com.google.common.collect.BoundType;
import it.unimi.dsi.fastutil.ints.IntCollection;
import it.unimi.dsi.fastutil.ints.IntIterator;
import it.unimi.dsi.fastutil.ints.IntSortedSet;

import java.util.NoSuchElementException;
import java.util.SortedSet;

import edu.umd.cs.findbugs.annotations.Nullable;

public final class IntRange {

    static IntRange create(IntCut lowerBound, IntCut upperBound) {
        return new IntRange(lowerBound, upperBound);
    }

    /**
     * Returns a range that contains all values strictly greater than {@code
     * lower} and strictly less than {@code upper}.
     *
     * @throws IllegalArgumentException if {@code lower} is greater than <i>or
     *                                  equal to</i> {@code upper}
     * @since 14.0
     */
    public static IntRange open(int lower, int upper) {
        return create(IntCut.aboveValue(lower), IntCut.belowValue(upper));
    }

    /**
     * Returns a range that contains all values greater than or equal to
     * {@code lower} and less than or equal to {@code upper}.
     *
     * @throws IllegalArgumentException if {@code lower} is greater than {@code
     *                                  upper}
     * @since 14.0
     */
    public static IntRange closed(int lower, int upper) {
        return create(IntCut.belowValue(lower), IntCut.aboveValue(upper));
    }

    /**
     * Returns a range that contains all values greater than or equal to
     * {@code lower} and strictly less than {@code upper}.
     *
     * @throws IllegalArgumentException if {@code lower} is greater than {@code
     *                                  upper}
     * @since 14.0
     */
    public static IntRange closedOpen(int lower, int upper) {
        return create(IntCut.belowValue(lower), IntCut.belowValue(upper));
    }

    /**
     * Returns a range that contains all values strictly greater than {@code
     * lower} and less than or equal to {@code upper}.
     *
     * @throws IllegalArgumentException if {@code lower} is greater than {@code
     *                                  upper}
     * @since 14.0
     */
    public static IntRange openClosed(int lower, int upper) {
        return create(IntCut.aboveValue(lower), IntCut.aboveValue(upper));
    }

    /**
     * Returns a range that contains any value from {@code lower} to {@code
     * upper}, where each endpoint may be either inclusive (closed) or exclusive
     * (open).
     *
     * @throws IllegalArgumentException if {@code lower} is greater than {@code
     *                                  upper}
     */
    public static IntRange range(int lower, BoundType lowerType, int upper, BoundType upperType) {
        checkNotNull(lowerType);
        checkNotNull(upperType);

        IntCut lowerBound = (lowerType == BoundType.OPEN) ? IntCut.aboveValue(lower) : IntCut.belowValue(lower);
        IntCut upperBound = (upperType == BoundType.OPEN) ? IntCut.belowValue(upper) : IntCut.aboveValue(upper);
        return create(lowerBound, upperBound);
    }

    /**
     * Returns a range that contains all values strictly less than {@code
     * endpoint}.
     *
     * @since 14.0
     */
    public static IntRange lessThan(int endpoint) {
        return create(IntCut.belowAll(), IntCut.belowValue(endpoint));
    }

    /**
     * Returns a range that contains all values less than or equal to
     * {@code endpoint}.
     *
     * @since 14.0
     */
    public static IntRange atMost(int endpoint) {
        return create(IntCut.belowAll(), IntCut.aboveValue(endpoint));
    }

    /**
     * Returns a range with no lower bound up to the given endpoint, which may be
     * either inclusive (closed) or exclusive (open).
     *
     * @since 14.0
     */
    public static IntRange upTo(int endpoint, BoundType boundType) {
        switch (boundType) {
            case OPEN:
                return lessThan(endpoint);
            case CLOSED:
                return atMost(endpoint);
            default:
                throw new AssertionError();
        }
    }

    /**
     * Returns a range that contains all values strictly greater than {@code
     * endpoint}.
     *
     * @since 14.0
     */
    public static IntRange greaterThan(int endpoint) {
        return create(IntCut.aboveValue(endpoint), IntCut.aboveAll());
    }

    /**
     * Returns a range that contains all values greater than or equal to
     * {@code endpoint}.
     *
     * @since 14.0
     */
    public static IntRange atLeast(int endpoint) {
        return create(IntCut.belowValue(endpoint), IntCut.aboveAll());
    }

    /**
     * Returns a range from the given endpoint, which may be either inclusive
     * (closed) or exclusive (open), with no upper bound.
     *
     * @since 14.0
     */
    public static IntRange downTo(int endpoint, BoundType boundType) {
        switch (boundType) {
            case OPEN:
                return greaterThan(endpoint);
            case CLOSED:
                return atLeast(endpoint);
            default:
                throw new AssertionError();
        }
    }

    private static final IntRange ALL = new IntRange(IntCut.belowAll(), IntCut.aboveAll());

    /**
     * Returns a range that contains every value of type {@code int}.
     */
    public static IntRange all() {
        return ALL;
    }

    /**
     * Returns a range that {@linkplain IntRange#contains(int) contains} only
     * the given value. The returned range is {@linkplain BoundType#CLOSED closed}
     * on both ends.
     *
     * @since 14.0
     */
    public static IntRange singleton(int value) {
        return closed(value, value);
    }

    /**
     * Returns the minimal range that
     * {@linkplain IntRange#contains(int) contains} all of the given values.
     * The returned range is {@linkplain BoundType#CLOSED closed} on both ends.
     *
     * @throws ClassCastException     if the parameters are not <i>mutually
     *                                comparable</i>
     * @throws NoSuchElementException if {@code values} is empty
     * @throws NullPointerException   if any of {@code values} is null
     * @since 14.0
     */
    public static IntRange encloseAll(IntCollection values) {
        checkNotNull(values);
        if (values instanceof IntSortedSet) {
            IntSortedSet setValues = (IntSortedSet) values;
            return closed(setValues.firstInt(), setValues.lastInt());
        }
        IntIterator valueIterator = values.iterator();
        int min = checkNotNull(valueIterator.next());
        int max = min;
        while (valueIterator.hasNext()) {
            int value = checkNotNull(valueIterator.next());
            min = Integer.min(min, value);
            max = Integer.max(max, value);
        }
        return closed(min, max);
    }

    final IntCut lowerBound;
    final IntCut upperBound;

    private IntRange(IntCut lowerBound, IntCut upperBound) {
        this.lowerBound = checkNotNull(lowerBound);
        this.upperBound = checkNotNull(upperBound);
        if (lowerBound.compareTo(upperBound) > 0 || lowerBound == IntCut.aboveAll()
                        || upperBound == IntCut.belowAll()) {
            throw new IllegalArgumentException("Invalid range: " + toString(lowerBound, upperBound));
        }
    }

    public IntCut upperBound() {
        return upperBound;
    }

    public IntCut lowerBound() {
        return lowerBound;
    }

    /**
     * Returns {@code true} if this range has a lower endpoint.
     */
    public boolean hasLowerBound() {
        return lowerBound != IntCut.belowAll();
    }

    /**
     * Returns the lower endpoint of this range.
     *
     * @throws IllegalStateException if this range is unbounded below (that is, {@link
     *                               #hasLowerBound()} returns {@code false})
     */
    public int lowerEndpoint() {
        return lowerBound.endpoint();
    }

    /**
     * Returns the type of this range's lower bound: {@link BoundType#CLOSED} if the range includes
     * its lower endpoint, {@link BoundType#OPEN} if it does not.
     *
     * @throws IllegalStateException if this range is unbounded below (that is, {@link
     *                               #hasLowerBound()} returns {@code false})
     */
    public BoundType lowerBoundType() {
        return lowerBound.typeAsLowerBound();
    }

    /**
     * Returns {@code true} if this range has an upper endpoint.
     */
    public boolean hasUpperBound() {
        return upperBound != IntCut.aboveAll();
    }

    /**
     * Returns the upper endpoint of this range.
     *
     * @throws IllegalStateException if this range is unbounded above (that is, {@link
     *                               #hasUpperBound()} returns {@code false})
     */
    public int upperEndpoint() {
        return upperBound.endpoint();
    }

    /**
     * Returns the type of this range's upper bound: {@link BoundType#CLOSED} if the range includes
     * its upper endpoint, {@link BoundType#OPEN} if it does not.
     *
     * @throws IllegalStateException if this range is unbounded above (that is, {@link
     *                               #hasUpperBound()} returns {@code false})
     */
    public BoundType upperBoundType() {
        return upperBound.typeAsUpperBound();
    }

    /**
     * Returns {@code true} if this range is of the form {@code [v..v)} or {@code (v..v]}. (This does
     * not encompass ranges of the form {@code (v..v)}, because such ranges are <i>invalid</i> and
     * can't be constructed at all.)
     * <p>
     * <p>Note that certain discrete ranges such as the integer range {@code (3..4)} are <b>not</b>
     * considered empty, even though they contain no actual values.  In these cases, it may be
     * helpful to preprocess ranges with {@link #canonical(IntegerDomain)}.
     */
    public boolean isEmpty() {
        return lowerBound.equals(upperBound);
    }

    /**
     * Returns {@code true} if {@code value} is within the bounds of this range. For example, on the
     * range {@code [0..2)}, {@code contains(1)} returns {@code true}, while {@code contains(2)}
     * returns {@code false}.
     */
    public boolean contains(int value) {
        return lowerBound.isLessThan(value) && !upperBound.isLessThan(value);
    }

    /**
     * Returns {@code true} if every element in {@code values} is {@linkplain #contains contained} in
     * this range.
     */
    public boolean containsAll(IntCollection values) {
        if (values.isEmpty()) {
            return true;
        }

        // this optimizes testing equality of two range-backed sets
        if (values instanceof IntSortedSet) {
            IntSortedSet set = (IntSortedSet) values;
            return contains(set.first()) && contains(set.last());
        }

        for (int value : values) {
            if (!contains(value)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Returns {@code true} if the bounds of {@code other} do not extend outside the bounds of this
     * range. Examples:
     * <p>
     * <ul>
     * <li>{@code [3..6]} encloses {@code [4..5]}
     * <li>{@code (3..6)} encloses {@code (3..6)}
     * <li>{@code [3..6]} encloses {@code [4..4)} (even though the latter is empty)
     * <li>{@code (3..6]} does not enclose {@code [3..6]}
     * <li>{@code [4..5]} does not enclose {@code (3..6)} (even though it contains every value
     * contained by the latter range)
     * <li>{@code [3..6]} does not enclose {@code (1..1]} (even though it contains every value
     * contained by the latter range)
     * </ul>
     * <p>
     * <p>Note that if {@code a.encloses(b)}, then {@code b.contains(v)} implies
     * {@code a.contains(v)}, but as the last two examples illustrate, the converse is not always
     * true.
     * <p>
     * <p>Being reflexive, antisymmetric and transitive, the {@code encloses} relation defines a
     * <i>partial order</i> over ranges. There exists a unique {@linkplain IntRange#all maximal} range
     * according to this relation, and also numerous {@linkplain #isEmpty minimal} ranges. Enclosure
     * also implies {@linkplain #isConnected connectedness}.
     */
    public boolean encloses(IntRange other) {
        return lowerBound.compareTo(other.lowerBound) <= 0 && upperBound.compareTo(other.upperBound) >= 0;
    }

    /**
     * Returns {@code true} if there exists a (possibly empty) range which is {@linkplain #encloses
     * enclosed} by both this range and {@code other}.
     * <p>
     * <p>For example,
     * <ul>
     * <li>{@code [2, 4)} and {@code [5, 7)} are not connected
     * <li>{@code [2, 4)} and {@code [3, 5)} are connected, because both enclose {@code [3, 4)}
     * <li>{@code [2, 4)} and {@code [4, 6)} are connected, because both enclose the empty range
     * {@code [4, 4)}
     * </ul>
     * <p>
     * <p>Note that this range and {@code other} have a well-defined {@linkplain #span union} and
     * {@linkplain #intersection intersection} (as a single, possibly-empty range) if and only if this
     * method returns {@code true}.
     * <p>
     * <p>The connectedness relation is both reflexive and symmetric, but does not form an {@linkplain
     * Equivalence equivalence relation} as it is not transitive.
     * <p>
     * <p>Note that certain discrete ranges are not considered connected, even though there are no
     * elements "between them."  For example, {@code [3, 5]} is not considered connected to {@code
     * [6, 10]}.  In these cases, it may be desirable for both input ranges to be preprocessed with
     * {@link #canonical(IntegerDomain)} before testing for connectedness.
     */
    public boolean isConnected(IntRange other) {
        return lowerBound.compareTo(other.upperBound) <= 0 && other.lowerBound.compareTo(upperBound) <= 0;
    }

    /**
     * Returns the maximal range {@linkplain #encloses enclosed} by both this range and {@code
     * connectedRange}, if such a range exists.
     * <p>
     * <p>For example, the intersection of {@code [1..5]} and {@code (3..7)} is {@code (3..5]}. The
     * resulting range may be empty; for example, {@code [1..5)} intersected with {@code [5..7)}
     * yields the empty range {@code [5..5)}.
     * <p>
     * <p>The intersection exists if and only if the two ranges are {@linkplain #isConnected
     * connected}.
     * <p>
     * <p>The intersection operation is commutative, associative and idempotent, and its identity
     * element is {@link IntRange#all}).
     *
     * @throws IllegalArgumentException if {@code isConnected(connectedRange)} is {@code false}
     */
    public IntRange intersection(IntRange connectedRange) {
        int lowerCmp = lowerBound.compareTo(connectedRange.lowerBound);
        int upperCmp = upperBound.compareTo(connectedRange.upperBound);
        if (lowerCmp >= 0 && upperCmp <= 0) {
            return this;
        } else if (lowerCmp <= 0 && upperCmp >= 0) {
            return connectedRange;
        } else {
            IntCut newLower = (lowerCmp >= 0) ? lowerBound : connectedRange.lowerBound;
            IntCut newUpper = (upperCmp <= 0) ? upperBound : connectedRange.upperBound;
            return create(newLower, newUpper);
        }
    }

    /**
     * Returns the minimal range that {@linkplain #encloses encloses} both this range and {@code
     * other}. For example, the span of {@code [1..3]} and {@code (5..7)} is {@code [1..7)}.
     * <p>
     * <p><i>If</i> the input ranges are {@linkplain #isConnected connected}, the returned range can
     * also be called their <i>union</i>. If they are not, note that the span might contain values
     * that are not contained in either input range.
     * <p>
     * <p>Like {@link #intersection(IntRange) intersection}, this operation is commutative, associative
     * and idempotent. Unlike it, it is always well-defined for any two input ranges.
     */
    public IntRange span(IntRange other) {
        int lowerCmp = lowerBound.compareTo(other.lowerBound);
        int upperCmp = upperBound.compareTo(other.upperBound);
        if (lowerCmp <= 0 && upperCmp >= 0) {
            return this;
        } else if (lowerCmp >= 0 && upperCmp <= 0) {
            return other;
        } else {
            IntCut newLower = (lowerCmp <= 0) ? lowerBound : other.lowerBound;
            IntCut newUpper = (upperCmp >= 0) ? upperBound : other.upperBound;
            return create(newLower, newUpper);
        }
    }

    /**
     * Returns the canonical form of this range in the given domain. The canonical form has the
     * following properties:
     * <p>
     * <ul>
     * <li>equivalence: {@code a.canonical().contains(v) == a.contains(v)} for all {@code v} (in other
     * words, {@code ContiguousSet.createFromCsv(a.canonical(domain), domain).equals(
     * ContiguousSet.createFromCsv(a, domain))}
     * <li>uniqueness: unless {@code a.isEmpty()},
     * {@code ContiguousSet.createFromCsv(a, domain).equals(ContiguousSet.createFromCsv(b, domain))} implies
     * {@code a.canonical(domain).equals(b.canonical(domain))}
     * <li>idempotence: {@code a.canonical(domain).canonical(domain).equals(a.canonical(domain))}
     * </ul>
     * <p>
     * <p>Furthermore, this method guarantees that the range returned will be one of the following
     * canonical forms:
     * <p>
     * <ul>
     * <li>[start..end)
     * <li>[start..+∞)
     * <li>(-∞..end) (only if type {@code int} is unbounded below)
     * <li>(-∞..+∞) (only if type {@code int} is unbounded below)
     * </ul>
     */
    public IntRange canonical(IntegerDomain domain) {
        checkNotNull(domain);
        IntCut lower = lowerBound.canonical(domain);
        IntCut upper = upperBound.canonical(domain);
        return (lower == lowerBound && upper == upperBound) ? this : create(lower, upper);
    }

    /**
     * Returns {@code true} if {@code object} is a range having the same endpoints and bound types as
     * this range. Note that discrete ranges such as {@code (1..4)} and {@code [2..3]} are <b>not</b>
     * equal to one another, despite the fact that they each contain precisely the same set of values.
     * Similarly, empty ranges are not equal unless they have exactly the same representation, so
     * {@code [3..3)}, {@code (3..3]}, {@code (4..4]} are all unequal.
     */
    @Override
    public boolean equals(@Nullable Object object) {
        if (object instanceof IntRange) {
            IntRange other = (IntRange) object;
            return lowerBound.equals(other.lowerBound) && upperBound.equals(other.upperBound);
        }
        return false;
    }

    /**
     * Returns a hash code for this range.
     */
    @Override
    public int hashCode() {
        return lowerBound.hashCode() * 31 + upperBound.hashCode();
    }

    /**
     * Returns a string representation of this range, such as {@code "[3..5)"} (other examples are
     * listed in the class documentation).
     */
    @Override
    public String toString() {
        return toString(lowerBound, upperBound);
    }

    private static String toString(IntCut lowerBound, IntCut upperBound) {
        StringBuilder sb = new StringBuilder(16);
        lowerBound.describeAsLowerBound(sb);
        sb.append("..");
        upperBound.describeAsUpperBound(sb);
        return sb.toString();
    }

    /**
     * Used to avoid http://bugs.sun.com/view_bug.do?bug_id=6558557
     */
    private static <T> SortedSet<T> cast(Iterable<T> iterable) {
        return (SortedSet<T>) iterable;
    }

    Object readResolve() {
        if (this.equals(ALL)) {
            return all();
        } else {
            return this;
        }
    }
}

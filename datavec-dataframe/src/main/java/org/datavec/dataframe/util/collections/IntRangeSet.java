package org.datavec.dataframe.util.collections;

import java.util.NoSuchElementException;
import java.util.Set;

import edu.umd.cs.findbugs.annotations.Nullable;

/**
 * A set comprising zero or more {@linkplain IntRange#isEmpty nonempty},
 * {@linkplain IntRange#isConnected(IntRange) disconnected} ranges of type {@code C}.
 * <p>
 * <p>Implementations that choose to support the {@link #add(IntRange)} operation are required to
 * ignore empty ranges and coalesce connected ranges.  For example:  <pre>   {@code
 * <p>
 *   IntRangeSet rangeSet = IntTreeRangeSet.createFromCsv();
 *   rangeSet.add(IntRange.closed(1, 10)); // {[1, 10]}
 *   rangeSet.add(IntRange.closedOpen(11, 15)); // disconnected range; {[1, 10], [11, 15)}
 *   rangeSet.add(IntRange.closedOpen(15, 20)); // connected range; {[1, 10], [11, 20)}
 *   rangeSet.add(IntRange.openClosed(0, 0)); // empty range; {[1, 10], [11, 20)}
 *   rangeSet.remove(IntRange.open(5, 10)); // splits [1, 10]; {[1, 5], [10, 10], [11, 20)}}</pre>
 * <p>
 * <p>Note that the behavior of {@link IntRange#isEmpty()} and {@link IntRange#isConnected(IntRange)} may
 * not be as expected on discrete ranges.  See the Javadoc of those methods for details.
 * <p>
 * <p>For a {@link Set} whose contents are specified by a {@link IntRange}.
 */
public interface IntRangeSet {

    // Query methods

    /**
     * Determines whether any of this range set's member ranges contains {@code value}.
     */
    boolean contains(int value);

    /**
     * Returns the unique range from this range set that {@linkplain IntRange#contains contains}
     * {@code value}, or {@code null} if this range set does not contain {@code value}.
     */
    IntRange rangeContaining(int value);

    /**
     * Returns {@code true} if there exists a non-empty range enclosed by both a member range in this
     * range set and the specified range. This is equivalent to calling
     * {@code subRangeSet(otherRange)} and testing whether the resulting range set is non-empty.
     *
     * @since 20.0
     */
    boolean intersects(IntRange otherRange);

    /**
     * Returns {@code true} if there exists a member range in this range set which
     * {@linkplain IntRange#encloses encloses} the specified range.
     */
    boolean encloses(IntRange otherRange);

    /**
     * Returns {@code true} if for each member range in {@code other} there exists a member range in
     * this range set which {@linkplain IntRange#encloses encloses} it. It follows that
     * {@code this.contains(value)} whenever {@code other.contains(value)}. Returns {@code true} if
     * {@code other} is empty.
     * <p>
     * <p>This is equivalent to checking if this range set {@link #encloses} each of the ranges in
     * {@code other}.
     */
    boolean enclosesAll(IntRangeSet other);

    /**
     * Returns {@code true} if this range set contains no ranges.
     */
    boolean isEmpty();

    /**
     * Returns the minimal range which {@linkplain IntRange#encloses(IntRange) encloses} all ranges
     * in this range set.
     *
     * @throws NoSuchElementException if this range set is {@linkplain #isEmpty() empty}
     */
    IntRange span();

    // Views

    /**
     * Returns a view of the {@linkplain IntRange#isConnected disconnected} ranges that make up this
     * range set.  The returned set may be empty. The iterators returned by its
     * {@link Iterable#iterator} method return the ranges in increasing order of lower bound
     * (equivalently, of upper bound).
     */
    Set<IntRange> asRanges();

    /**
     * Returns a view of the complement of this {@code IntRangeSet}.
     * <p>
     * <p>The returned view supports the {@link #add} operation if this {@code IntRangeSet} supports
     * {@link #remove}, and vice versa.
     */
    IntRangeSet complement();

    /**
     * Returns a view of the intersection of this {@code IntRangeSet} with the specified range.
     * <p>
     * <p>The returned view supports all optional operations supported by this {@code IntRangeSet}, with
     * the caveat that an {@link IllegalArgumentException} is thrown on an attempt to
     * {@linkplain #add(IntRange) add} any range not {@linkplain IntRange#encloses(IntRange) enclosed} by
     * {@code view}.
     */
    IntRangeSet subRangeSet(IntRange view);

    // Modification

    /**
     * Adds the specified range to this {@code IntRangeSet} (optional operation). That is, for equal
     * range sets a and b, the result of {@code a.add(range)} is that {@code a} will be the minimal
     * range set for which both {@code a.enclosesAll(b)} and {@code a.encloses(range)}.
     * <p>
     * <p>Note that {@code range} will be {@linkplain IntRange#span(IntRange) coalesced} with any ranges in
     * the range set that are {@linkplain IntRange#isConnected(IntRange) connected} with it.  Moreover,
     * if {@code range} is empty, this is a no-op.
     *
     * @throws UnsupportedOperationException if this range set does not support the {@code add}
     *                                       operation
     */
    void add(IntRange range);

    /**
     * Removes the specified range from this {@code IntRangeSet} (optional operation). After this
     * operation, if {@code range.contains(c)}, {@code this.contains(c)} will return {@code false}.
     * <p>
     * <p>If {@code range} is empty, this is a no-op.
     *
     * @throws UnsupportedOperationException if this range set does not support the {@code remove}
     *                                       operation
     */
    void remove(IntRange range);

    /**
     * Removes all ranges from this {@code IntRangeSet} (optional operation).  After this operation,
     * {@code this.contains(c)} will return false for all {@code c}.
     * <p>
     * <p>This is equivalent to {@code remove(IntRange.all())}.
     *
     * @throws UnsupportedOperationException if this range set does not support the {@code clear}
     *                                       operation
     */
    void clear();

    /**
     * Adds all of the ranges from the specified range set to this range set (optional operation).
     * After this operation, this range set is the minimal range set that
     * {@linkplain #enclosesAll(IntRangeSet) encloses} both the original range set and {@code other}.
     * <p>
     * <p>This is equivalent to calling {@link #add} on each of the ranges in {@code other} in turn.
     *
     * @throws UnsupportedOperationException if this range set does not support the {@code addAll}
     *                                       operation
     */
    void addAll(IntRangeSet other);

    /**
     * Removes all of the ranges from the specified range set from this range set (optional
     * operation). After this operation, if {@code other.contains(c)}, {@code this.contains(c)} will
     * return {@code false}.
     * <p>
     * <p>This is equivalent to calling {@link #remove} on each of the ranges in {@code other} in
     * turn.
     *
     * @throws UnsupportedOperationException if this range set does not support the {@code removeAll}
     *                                       operation
     */
    void removeAll(IntRangeSet other);

    // Object methods

    /**
     * Returns {@code true} if {@code obj} is another {@code IntRangeSet} that contains the same ranges
     * according to {@link IntRange#equals(Object)}.
     */
    @Override
    boolean equals(@Nullable Object obj);

    /**
     * Returns {@code asRanges().hashCode()}.
     */
    @Override
    int hashCode();

    /**
     * Returns a readable string representation of this range set. For example, if this
     * {@code IntRangeSet} consisted of {@code IntRange.closed(1, 3)} and {@code IntRange.greaterThan(4)},
     * this might return {@code " [1..3](4..+∞)}"}.
     */
    @Override
    String toString();
}

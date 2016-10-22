package org.datavec.dataframe.util.collections;

/**
 * A descriptor for a <i>discrete</i> {@code Comparable} domain such as all
 * {@link Integer} instances. A discrete domain is one that supports the three basic
 * operations: {@link #next}, {@link #previous} and {@link #distance}, according
 * to their specifications. The methods {@link #minValue} and {@link #maxValue}
 * should also be overridden for bounded types.
 * <p>
 * <p>A discrete domain always represents the <i>entire</i> set of values of its
 * type; it cannot represent partial domains such as "prime integers" or
 * "strings of length 5."
 * <p>
 * <p>See the Guava User Guide section on <a href=
 * "https://github.com/google/guava/wiki/RangesExplained#discrete-domains">
 * {@code IntegerDomain}</a>.
 *
 * @author Kevin Bourrillion
 * @since 10.0
 */

final class IntegerDomain {

  private static final IntegerDomain INSTANCE = new IntegerDomain();

  /**
   * Returns the discrete domain for values of type {@code Integer}.
   *
   * @since 14.0 (since 10.0 as {@code DiscreteDomains.integers()})
   */
  public static IntegerDomain integers() {
    return INSTANCE;
  }

  public Integer next(Integer value) {
    int i = value;
    return (i == Integer.MAX_VALUE) ? null : i + 1;
  }

  public Integer previous(Integer value) {
    int i = value;
    return (i == Integer.MIN_VALUE) ? null : i - 1;
  }

  public long distance(Integer start, Integer end) {
    return (long) end - start;
  }

  public Integer minValue() {
    return Integer.MIN_VALUE;
  }

  public Integer maxValue() {
    return Integer.MAX_VALUE;
  }

  private Object readResolve() {
    return INSTANCE;
  }

  @Override
  public String toString() {
    return "IntegerDomain.integers()";
  }
}

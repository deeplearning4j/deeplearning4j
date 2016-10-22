package org.datavec.dataframe.util.collections;

import com.google.common.collect.BoundType;
import com.google.common.primitives.Booleans;

import java.util.NoSuchElementException;

/**
 *
 */
abstract class IntCut implements Comparable<IntCut> {

  protected final int endpoint;

  IntCut(int endpoint) {
    this.endpoint = endpoint;
  }

  abstract boolean isLessThan(int value);

  abstract BoundType typeAsLowerBound();

  abstract BoundType typeAsUpperBound();

  abstract IntCut withLowerBoundType(BoundType boundType, IntegerDomain domain);

  abstract IntCut withUpperBoundType(BoundType boundType, IntegerDomain domain);

  abstract void describeAsLowerBound(StringBuilder sb);

  abstract void describeAsUpperBound(StringBuilder sb);

  abstract int leastValueAbove(IntegerDomain domain);

  abstract int greatestValueBelow(IntegerDomain domain);

  /*
   * The canonical form is a BelowValue cut whenever possible, otherwise ABOVE_ALL, or
   * (only in the case of types that are unbounded below) BELOW_ALL.
  */
  IntCut canonical(IntegerDomain domain) {
    return this;
  }


  @Override
  public int compareTo(IntCut that) {
    if (that == belowAll()) {
      return 1;
    }
    if (that == aboveAll()) {
      return -1;
    }
    int result = Integer.compare(endpoint, that.endpoint);
    if (result != 0) {
      return result;
    }
    // same value. below comes before above
    return Booleans.compare(this instanceof AboveValue, that instanceof AboveValue);
  }

  int endpoint() {
    return this.endpoint;
  }

  @Override
  public boolean equals(Object obj) {
    if (obj instanceof IntCut) {
      IntCut that = (IntCut) obj;
      int compareResult = compareTo(that);
      return compareResult == 0;
    }
    return false;
  }

  static IntCut belowAll() {
    return IntCut.BelowAll.INSTANCE;
  }

  static IntCut aboveAll() {
    return IntCut.AboveAll.INSTANCE;
  }

  static IntCut belowValue(int endpoint) {
    return new IntCut.BelowValue(endpoint);
  }

  static IntCut aboveValue(int endpoint) {
    return new IntCut.AboveValue(endpoint);
  }


  private static final class AboveValue extends IntCut {

    AboveValue(int endpoint) {
      super(endpoint);
    }

    boolean isLessThan(int value) {
      return Integer.compare(this.endpoint, value) < 0;
    }

    BoundType typeAsLowerBound() {
      return BoundType.OPEN;
    }

    BoundType typeAsUpperBound() {
      return BoundType.CLOSED;
    }

    IntCut withLowerBoundType(BoundType boundType, IntegerDomain domain) {
      switch (boundType) {
        case OPEN:
          return this;
        case CLOSED:
          Integer next = domain.next(endpoint);
          return (next == null) ? IntCut.belowAll() : belowValue(next);
        default:
          throw new AssertionError();
      }
    }

    IntCut withUpperBoundType(BoundType boundType, IntegerDomain domain) {
      switch (boundType) {
        case OPEN:
          Integer next = domain.next(endpoint);
          return (next == null) ? IntCut.aboveAll() : belowValue(next);
        case CLOSED:
          return this;
        default:
          throw new AssertionError();
      }
    }

    void describeAsLowerBound(StringBuilder sb) {
      sb.append('(').append(this.endpoint);
    }

    void describeAsUpperBound(StringBuilder sb) {
      sb.append(this.endpoint).append(']');
    }

    int leastValueAbove(IntegerDomain domain) {
      return domain.next(this.endpoint);
    }

    int greatestValueBelow(IntegerDomain domain) {
      return this.endpoint;
    }

    IntCut canonical(IntegerDomain domain) {
      Integer next = this.leastValueAbove(domain);
      return next != null ? belowValue(next) : IntCut.aboveAll();
    }

    public int hashCode() {
      return endpoint;
    }

    public String toString() {
      return "/" + this.endpoint + "\\";
    }
  }


  private static final class BelowValue extends IntCut {

    BelowValue(int endpoint) {
      super(endpoint);
    }

    boolean isLessThan(int value) {
      return Integer.compare(this.endpoint, value) <= 0;
    }

    BoundType typeAsLowerBound() {
      return BoundType.CLOSED;
    }

    BoundType typeAsUpperBound() {
      return BoundType.OPEN;
    }

    IntCut withLowerBoundType(BoundType boundType, IntegerDomain domain) {
      switch (boundType) {
        case CLOSED:
          return this;
        case OPEN:
          Integer previous = domain.previous(endpoint);
          return (previous == null) ? IntCut.belowAll() : new AboveValue(previous);
        default:
          throw new AssertionError();
      }
    }

    IntCut withUpperBoundType(BoundType boundType, IntegerDomain domain) {
      switch (boundType) {
        case CLOSED:
          Integer previous = domain.previous(endpoint);
          return (previous == null) ? IntCut.aboveAll() : new AboveValue(previous);
        case OPEN:
          return this;
        default:
          throw new AssertionError();
      }
    }

    void describeAsLowerBound(StringBuilder sb) {
      sb.append('[').append(this.endpoint);
    }

    void describeAsUpperBound(StringBuilder sb) {
      sb.append(this.endpoint).append(')');
    }

    int leastValueAbove(IntegerDomain domain) {
      return this.endpoint;
    }

    int greatestValueBelow(IntegerDomain domain) {
      return domain.previous(endpoint);
    }

    public int hashCode() {
      return this.endpoint;
    }

    public String toString() {
      return "\\" + this.endpoint + "/";
    }

    static IntCut aboveValue(int endpoint) {
      return new AboveValue(endpoint);
    }
  }


  private static final class AboveAll extends IntCut {

    private static final IntCut.AboveAll INSTANCE = new IntCut.AboveAll();

    private AboveAll() {
      super(Integer.MAX_VALUE);
    }

    int endpoint() {
      throw new IllegalStateException("range unbounded on this side");
    }

    boolean isLessThan(int value) {
      return false;
    }

    BoundType typeAsLowerBound() {
      throw new AssertionError("this statement should be unreachable");
    }

    BoundType typeAsUpperBound() {
      throw new IllegalStateException();
    }

    IntCut withLowerBoundType(BoundType boundType, IntegerDomain domain) {
      throw new AssertionError("this statement should be unreachable");
    }

    IntCut withUpperBoundType(BoundType boundType, IntegerDomain domain) {
      throw new IllegalStateException();
    }

    void describeAsLowerBound(StringBuilder sb) {
      throw new AssertionError();
    }

    void describeAsUpperBound(StringBuilder sb) {
      sb.append("+∞)");
    }

    int leastValueAbove(IntegerDomain domain) {
      throw new AssertionError();
    }

    int greatestValueBelow(IntegerDomain domain) {
      return domain.maxValue();
    }

    public int compareTo(IntCut o) {
      return o == this ? 0 : 1;
    }

    public String toString() {
      return "+∞";
    }

    static IntCut belowValue(int endpoint) {
      return new BelowValue(endpoint);
    }

    private Object readResolve() {
      return INSTANCE;
    }
  }


  private static final class BelowAll extends IntCut {

    private static final IntCut.BelowAll INSTANCE = new IntCut.BelowAll();

    private BelowAll() {
      super(Integer.MIN_VALUE);
    }

    int endpoint() {
      throw new IllegalStateException("range unbounded on this side");
    }

    boolean isLessThan(int value) {
      return true;
    }

    BoundType typeAsLowerBound() {
      throw new IllegalStateException();
    }

    BoundType typeAsUpperBound() {
      throw new AssertionError("this statement should be unreachable");
    }

    IntCut withLowerBoundType(BoundType boundType, IntegerDomain domain) {
      throw new IllegalStateException();
    }

    IntCut withUpperBoundType(BoundType boundType, IntegerDomain domain) {
      throw new AssertionError("this statement should be unreachable");
    }

    void describeAsLowerBound(StringBuilder sb) {
      sb.append("(-∞");
    }

    void describeAsUpperBound(StringBuilder sb) {
      throw new AssertionError();
    }

    int leastValueAbove(IntegerDomain domain) {
      return domain.minValue();
    }

    int greatestValueBelow(IntegerDomain domain) {
      throw new AssertionError();
    }

    IntCut canonical(IntegerDomain domain) {
      try {
        return IntCut.belowValue(domain.minValue());
      } catch (NoSuchElementException var3) {
        return this;
      }
    }

    public int compareTo(IntCut o) {
      return o == this ? 0 : -1;
    }

    public String toString() {
      return "-∞";
    }

    private Object readResolve() {
      return INSTANCE;
    }
  }

}

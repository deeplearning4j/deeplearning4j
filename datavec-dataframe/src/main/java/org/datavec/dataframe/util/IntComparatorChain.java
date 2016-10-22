package org.datavec.dataframe.util;

/**
 *
 */

import it.unimi.dsi.fastutil.ints.IntComparator;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

public class IntComparatorChain implements IntComparator, Serializable {

  private static final long serialVersionUID = -721644942746081630L;
  private final List<IntComparator> comparatorChain;
  private BitSet orderingBits;
  private boolean isLocked;

  public IntComparatorChain() {
    this(new ArrayList<>(), new BitSet());
  }

  public IntComparatorChain(IntComparator comparator) {
    this(comparator, false);
  }

  public IntComparatorChain(IntComparator comparator, boolean reverse) {
    this.orderingBits = null;
    this.isLocked = false;
    this.comparatorChain = new ArrayList<>(1);
    this.comparatorChain.add(comparator);
    this.orderingBits = new BitSet(1);
    if (reverse) {
      this.orderingBits.set(0);
    }

  }

  public IntComparatorChain(List<IntComparator> list) {
    this(list, new BitSet(list.size()));
  }

  public IntComparatorChain(List<IntComparator> list, BitSet bits) {
    this.orderingBits = null;
    this.isLocked = false;
    this.comparatorChain = list;
    this.orderingBits = bits;
  }

  public void addComparator(IntComparator comparator) {
    this.addComparator(comparator, false);
  }

  public void addComparator(IntComparator comparator, boolean reverse) {
    this.checkLocked();
    this.comparatorChain.add(comparator);
    if (reverse) {
      this.orderingBits.set(this.comparatorChain.size() - 1);
    }
  }

  public void setComparator(int index, IntComparator comparator) throws IndexOutOfBoundsException {
    this.setComparator(index, comparator, false);
  }

  public void setComparator(int index, IntComparator comparator, boolean reverse) {
    this.checkLocked();
    this.comparatorChain.set(index, comparator);
    if (reverse) {
      this.orderingBits.set(index);
    } else {
      this.orderingBits.clear(index);
    }
  }

  public void setForwardSort(int index) {
    this.checkLocked();
    this.orderingBits.clear(index);
  }

  public void setReverseSort(int index) {
    this.checkLocked();
    this.orderingBits.set(index);
  }

  public int size() {
    return this.comparatorChain.size();
  }

  public boolean isLocked() {
    return this.isLocked;
  }

  private void checkLocked() {
    if (this.isLocked) {
      throw new UnsupportedOperationException("Comparator ordering cannot be changed after the first comparison is " +
          "performed");
    }
  }

  private void checkChainIntegrity() {
    if (this.comparatorChain.size() == 0) {
      throw new UnsupportedOperationException("ComparatorChains must contain at least one Comparator");
    }
  }

  public int compare(Integer o1, Integer o2) throws UnsupportedOperationException {
    if (!this.isLocked) {
      this.checkChainIntegrity();
      this.isLocked = true;
    }

    Iterator comparators = this.comparatorChain.iterator();

    for (int comparatorIndex = 0; comparators.hasNext(); ++comparatorIndex) {
      Comparator comparator = (Comparator) comparators.next();
      int retval = comparator.compare(o1, o2);
      if (retval != 0) {
        if (this.orderingBits.get(comparatorIndex)) {
          if (retval > 0) {
            retval = -1;
          } else {
            retval = 1;
          }
        }
        return retval;
      }
    }
    return 0;
  }

  public int compare(int o1, int o2) throws UnsupportedOperationException {
    if (!this.isLocked) {
      this.checkChainIntegrity();
      this.isLocked = true;
    }

    Iterator comparators = this.comparatorChain.iterator();

    for (int comparatorIndex = 0; comparators.hasNext(); ++comparatorIndex) {
      IntComparator comparator = (IntComparator) comparators.next();
      int retval = comparator.compare(o1, o2);
      if (retval != 0) {
        if (this.orderingBits.get(comparatorIndex)) {
          if (retval > 0) {
            retval = -1;
          } else {
            retval = 1;
          }
        }
        return retval;
      }
    }
    return 0;
  }

  public int hashCode() {
    int hash = 0;
    if (null != this.comparatorChain) {
      hash ^= this.comparatorChain.hashCode();
    }

    if (null != this.orderingBits) {
      hash ^= this.orderingBits.hashCode();
    }
    return hash;
  }

  public boolean equals(Object object) {
    if (this == object) {
      return true;
    } else if (null == object) {
      return false;
    } else if (!object.getClass().equals(this.getClass())) {
      return false;
    } else {
      boolean var10000;
      label48:
      {
        label32:
        {
          IntComparatorChain chain = (IntComparatorChain) object;
          if (null == this.orderingBits) {
            if (null != chain.orderingBits) {
              break label32;
            }
          } else if (!this.orderingBits.equals(chain.orderingBits)) {
            break label32;
          }

          if (null == this.comparatorChain) {
            if (null == chain.comparatorChain) {
              break label48;
            }
          } else if (this.comparatorChain.equals(chain.comparatorChain)) {
            break label48;
          }
        }

        var10000 = false;
        return var10000;
      }

      var10000 = true;
      return var10000;
    }
  }
}

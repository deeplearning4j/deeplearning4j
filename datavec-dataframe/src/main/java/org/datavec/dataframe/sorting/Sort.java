package org.datavec.dataframe.sorting;

import com.google.common.base.MoreObjects;

import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Provides fine-grained control over sorting.
 * <p>
 * Use:
 * <p>
 * table.sortOn(first("Year", DESCEND).next("State", ASCEND));
 * <p>
 * This sorts table on the column named year in descending order, such that the most recent years
 * appear first, then on State, in ascending order so "AL" will appear before "CA". You can add
 * additional instructions for multi-column sorts by chaining additional calls to next() with the
 * appropriate column names and Order.
 */
public class Sort implements Iterable<Map.Entry<String, Sort.Order>> {

  public enum Order {ASCEND, DESCEND}

  private final LinkedHashMap<String, Order> sortOrder = new LinkedHashMap<>();

  public static Sort on(String columnName, Order order) {
    return new Sort(columnName, order);
  }

  public Sort(String columnName, Order order) {
    next(columnName, order);
  }

  public Sort next(String columnName, Order order) {
    sortOrder.put(columnName, order);
    return this;
  }

  public boolean isEmpty() {
    return sortOrder.isEmpty();
  }

  public int size() {
    return sortOrder.size();
  }

  /**
   * Returns an iterator over elements of type {@code T}.
   *
   * @return an Iterator.
   */
  @Override
  public Iterator<Map.Entry<String, Order>> iterator() {
    return sortOrder.entrySet().iterator();
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("order", sortOrder)
        .toString();
  }
}

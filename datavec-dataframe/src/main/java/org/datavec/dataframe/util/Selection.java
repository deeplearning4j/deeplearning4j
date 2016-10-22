package org.datavec.dataframe.util;

import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntIterable;
import org.roaringbitmap.RoaringBitmap;

/**
 * A selection maintains an ordered set of ints that can be used to select rows from a table or column
 */
public interface Selection extends IntIterable {

  int[] toArray();

  /**
   * Returns an IntArrayList containing the ints in this selection
   */
  RoaringBitmap toBitmap();

  IntArrayList toIntArrayList();

  /**
   * Adds the given integer to the Selection if it is not already present, and does nothing otherwise
   */
  void add(int i);

  int size();

  /**
   * Intersects the receiver and {@code otherSelection}, updating the receiver
   */
  void and(Selection otherSelection);

  /**
   * Implements the union of the receiver and {@code otherSelection}, updating the receiver
   */
  void or(Selection otherSelection);

  /**
   * Implements the set difference operation between the receiver and {@code otherSelection}, updating the receiver
   */
  void andNot(Selection otherSelection);

  boolean isEmpty();

  void clear();

  boolean contains(int i);

  /**
   * Adds to the current bitmap all integers in [rangeStart,rangeEnd)
   *
   * @param start inclusive beginning of range
   * @param end exclusive ending of range
   */
  void addRange(int start, int end);

  int get(int i);
}

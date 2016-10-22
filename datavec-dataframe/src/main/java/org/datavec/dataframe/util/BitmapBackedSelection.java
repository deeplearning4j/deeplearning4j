package org.datavec.dataframe.util;

import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntIterator;
import org.roaringbitmap.RoaringBitmap;

/**
 *
 */
public class BitmapBackedSelection implements Selection {

  private final RoaringBitmap bitmap;

  public BitmapBackedSelection(RoaringBitmap bitmap) {
    this.bitmap = bitmap;
  }

  public BitmapBackedSelection() {
    this.bitmap = new RoaringBitmap();
  }

  public void add(int i) {
    bitmap.add(i);
  }

  @Override
  public int size() {
    return bitmap.getCardinality();
  }

  @Override
  public int[] toArray() {
    return bitmap.toArray();
  }

  @Override
  public RoaringBitmap toBitmap() {
    return bitmap.clone();
  }

  @Override
  public IntArrayList toIntArrayList() {
    return new IntArrayList(bitmap.toArray());
  }


  /**
   * Intersects the receiver and {@code otherSelection}, updating the receiver
   */
  @Override
  public void and(Selection otherSelection) {
    bitmap.and(otherSelection.toBitmap());
  }

  /**
   * Implements the union of the receiver and {@code otherSelection}, updating the receiver
   */
  @Override
  public void or(Selection otherSelection) {
    bitmap.or(otherSelection.toBitmap());
  }

  /**
   * Implements the set difference operation between the receiver and {@code otherSelection}, updating the receiver
   */
  @Override
  public void andNot(Selection otherSelection) {
    bitmap.andNot(otherSelection.toBitmap());
  }

  @Override
  public boolean isEmpty() {
    return size() == 0;
  }

  @Override
  public void clear() {
    bitmap.clear();
  }

  @Override
  public boolean contains(int i) {
    return bitmap.contains(i);
  }

  /**
   * Adds to the current bitmap all integers in [rangeStart,rangeEnd)
   *
   * @param start inclusive beginning of range
   * @param end   exclusive ending of range
   */
  @Override
  public void addRange(int start, int end) {
    bitmap.add(start, end);
  }

  @Override
  public int get(int i) {
    return bitmap.select(i);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;

    BitmapBackedSelection integers = (BitmapBackedSelection) o;

    return bitmap.equals(integers.bitmap);
  }

  @Override
  public int hashCode() {
    return bitmap.hashCode();
  }

  @Override
  public IntIterator iterator() {

    return new IntIterator() {

      private final org.roaringbitmap.IntIterator iterator = bitmap.getIntIterator();

      @Override
      public int nextInt() {
        return iterator.next();
      }

      @Override
      public int skip(int k) {
        throw new UnsupportedOperationException("Views do not support skipping in the iterator");
      }

      @Override
      public boolean hasNext() {
        return iterator.hasNext();
      }

      @Override
      public Integer next() {
        return iterator.next();
      }
    };
  }
}

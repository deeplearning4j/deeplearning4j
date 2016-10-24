package org.datavec.dataframe.api;

import org.datavec.dataframe.columns.AbstractColumn;
import org.datavec.dataframe.columns.Column;
import org.datavec.dataframe.columns.ShortColumnUtils;
import org.datavec.dataframe.filtering.ShortBiPredicate;
import org.datavec.dataframe.filtering.ShortPredicate;
import org.datavec.dataframe.io.TypeUtils;
import org.datavec.dataframe.mapping.ShortMapUtils;
import org.datavec.dataframe.reducing.NumericReduceUtils;
import org.datavec.dataframe.sorting.IntComparisonUtil;
import org.datavec.dataframe.store.ColumnMetadata;
import org.datavec.dataframe.util.BitmapBackedSelection;
import org.datavec.dataframe.util.ReverseShortComparator;
import org.datavec.dataframe.util.Selection;
import org.datavec.dataframe.util.Stats;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import it.unimi.dsi.fastutil.floats.FloatArrayList;
import it.unimi.dsi.fastutil.ints.IntComparator;
import it.unimi.dsi.fastutil.shorts.ShortArrayList;
import it.unimi.dsi.fastutil.shorts.ShortArrays;
import it.unimi.dsi.fastutil.shorts.ShortIterator;
import it.unimi.dsi.fastutil.shorts.ShortOpenHashSet;
import it.unimi.dsi.fastutil.shorts.ShortSet;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A column that contains signed 2 byte integer values
 */
public class ShortColumn extends AbstractColumn implements ShortMapUtils, NumericColumn {

  public static final short MISSING_VALUE = (short) ColumnType.SHORT_INT.getMissingValue();

  private static final int DEFAULT_ARRAY_SIZE = 128;
  private static final int BYTE_SIZE = 2;

  private ShortArrayList data;

  public static ShortColumn create(String name) {
    return new ShortColumn(name, DEFAULT_ARRAY_SIZE);
  }

  public static ShortColumn create(ColumnMetadata metadata) {
    return new ShortColumn(metadata);
  }

  public static ShortColumn create(String name, int arraySize) {
    return new ShortColumn(name, arraySize);
  }

  public static ShortColumn create(String name, ShortArrayList ints) {
    ShortColumn column = new ShortColumn(name, ints.size());
    column.data = ints;
    return column;
  }

  public ShortColumn(String name, int initialSize) {
    super(name);
    data = new ShortArrayList(initialSize);
  }

  public ShortColumn(ColumnMetadata metadata) {
    super(metadata);
    data = new ShortArrayList(metadata.getSize());
  }

  public ShortColumn(String name) {
    super(name);
    data = new ShortArrayList(DEFAULT_ARRAY_SIZE);
  }

  public int size() {
    return data.size();
  }

  @Override
  public ColumnType type() {
    return ColumnType.SHORT_INT;
  }

  public void add(short i) {
    data.add(i);
  }

  public void set(int index, short value) {
    data.set(index, value);
  }

  public Selection isLessThan(int i) {
    return select(ShortColumnUtils.isLessThan, i);
  }

  public Selection isGreaterThan(int i) {
    return select(ShortColumnUtils.isGreaterThan, i);
  }

  public Selection isGreaterThanOrEqualTo(int i) {
    return select(ShortColumnUtils.isGreaterThanOrEqualTo, i);
  }

  public Selection isLessThanOrEqualTo(int i) {
    return select(ShortColumnUtils.isLessThanOrEqualTo, i);
  }

  public Selection isEqualTo(int i) {
    return select(ShortColumnUtils.isEqualTo, i);
  }

  public Selection isEqualTo(ShortColumn f) {
    Selection results = new BitmapBackedSelection();
    int i = 0;
    ShortIterator shortIterator = f.iterator();
    for (int next : data) {
      if (next == shortIterator.next()) {
        results.add(i);
      }
      i++;
    }
    return results;
  }

  @Override
  public Table summary() {
    return stats().asTable();
  }

  /**
   * Returns the count of missing values in this column
   */
  @Override
  public int countMissing() {
    int count = 0;
    for (int i = 0; i < size(); i++) {
      if (get(i) == MISSING_VALUE) {
        count++;
      }
    }
    return count;
  }

  @Override
  public int countUnique() {
    Selection selection = new BitmapBackedSelection();
    for (int i : data) {
      selection.add(i);
    }
    return selection.size();
  }

  @Override
  public ShortColumn unique() {
    Selection selection = new BitmapBackedSelection();
    for (short i : data) {
      selection.add(i);
    }
    int[] ints = selection.toArray();
    short[] shorts = new short[ints.length];
    for (int i = 0; i < ints.length; i++) {
      shorts[i] = (short) ints[i];
    }
    return ShortColumn.create(name() + " Unique values", ShortArrayList.wrap(shorts));
  }

  @Override
  public String getString(int row) {
    return String.valueOf(data.getShort(row));
  }

  @Override
  public ShortColumn emptyCopy() {
    ShortColumn column = new ShortColumn(name(), DEFAULT_ARRAY_SIZE);
    column.setComment(comment());
    return column;
  }

  @Override
  public ShortColumn emptyCopy(int rowSize) {
    ShortColumn column = new ShortColumn(name(), rowSize);
    column.setComment(comment());
    return column;
  }

  @Override
  public void clear() {
    data.clear();
  }

  @Override
  public void sortAscending() {
    Arrays.parallelSort(data.elements());
  }

  @Override
  public void sortDescending() {
    ShortArrays.parallelQuickSort(data.elements(), ReverseShortComparator.instance());
  }

  @Override
  public ShortColumn copy() {
    ShortColumn copy = emptyCopy(size());
    copy.data.addAll(data);
    copy.setComment(comment());
    return copy;
  }

  @Override
  public boolean isEmpty() {
    return data.isEmpty();
  }

  @Override  // TODO(lwhite): Move to AbstractColumn
  public void addCell(String object) {
    try {
      add(convert(object));
    } catch (NumberFormatException nfe) {
      throw new NumberFormatException(name() + ": " + nfe.getMessage());
    } catch (NullPointerException e) {
      throw new RuntimeException(name() + ": "
          + String.valueOf(object) + ": "
          + e.getMessage());
    }
  }

  /**
   * Returns a float that is parsed from the given String
   * <p>
   * We remove any commas before parsing
   */
  public static short convert(String stringValue) {
    if (Strings.isNullOrEmpty(stringValue) || TypeUtils.MISSING_INDICATORS.contains(stringValue)) {
      return (short) ColumnType.SHORT_INT.getMissingValue();
    }
    Matcher matcher = COMMA_PATTERN.matcher(stringValue);
    return Short.parseShort(matcher.replaceAll(""));
  }

  private static final Pattern COMMA_PATTERN = Pattern.compile(",");

  public short get(int index) {
    return data.getShort(index);
  }

  @Override
  public float getFloat(int index) {
    return (float) data.getShort(index);
  }

  @Override
  public IntComparator rowComparator() {
    return comparator;
  }

  final IntComparator comparator = new IntComparator() {

    @Override
    public int compare(Integer i1, Integer i2) {
      return compare((int) i1, (int) i2);
    }

    public int compare(int i1, int i2) {
      int prim1 = get(i1);
      int prim2 = get(i2);
      return IntComparisonUtil.getInstance().compare(prim1, prim2);
    }
  };

  // Reduce functions applied to the whole column
  public long sum() {
    return Math.round(NumericReduceUtils.sum.reduce(toDoubleArray()));
  }

  public double product() {
    return NumericReduceUtils.product.reduce(this);
  }

  public double mean() {
    return NumericReduceUtils.mean.reduce(this);
  }

  public double median() {
    return NumericReduceUtils.median.reduce(this);
  }

  public double quartile1() {
    return NumericReduceUtils.quartile1.reduce(this);
  }

  public double quartile3() {
    return NumericReduceUtils.quartile3.reduce(this);
  }

  public double percentile(double percentile) {
    return NumericReduceUtils.percentile(this.toDoubleArray(), percentile);
  }

  public double range() {
    return NumericReduceUtils.range.reduce(this);
  }

  public double max() {
    return (short) Math.round(NumericReduceUtils.max.reduce(this));
  }

  public double min() {
    return (short) Math.round(NumericReduceUtils.min.reduce(this));
  }

  public double variance() {
    return NumericReduceUtils.variance.reduce(this);
  }

  public double populationVariance() {
    return NumericReduceUtils.populationVariance.reduce(this);
  }

  public double standardDeviation() {
    return NumericReduceUtils.stdDev.reduce(this);
  }

  public double sumOfLogs() {
    return NumericReduceUtils.sumOfLogs.reduce(this);
  }

  public double sumOfSquares() {
    return NumericReduceUtils.sumOfSquares.reduce(this);
  }

  public double geometricMean() {
    return NumericReduceUtils.geometricMean.reduce(this);
  }

  /**
   * Returns the quadraticMean, aka the root-mean-square, for all values in this column
   */
  public double quadraticMean() {
    return NumericReduceUtils.quadraticMean.reduce(this);
  }

  public double kurtosis() {
    return NumericReduceUtils.kurtosis.reduce(this);
  }

  public double skewness() {
    return NumericReduceUtils.skewness.reduce(this);
  }

  public short firstElement() {
    if (size() > 0) {
      return get(0);
    }
    return MISSING_VALUE;
  }

  public Selection isPositive() {
    return select(ShortColumnUtils.isPositive);
  }

  public Selection isNegative() {
    return select(ShortColumnUtils.isNegative);
  }

  public Selection isNonNegative() {
    return select(ShortColumnUtils.isNonNegative);
  }

  public Selection isZero() {
    return select(ShortColumnUtils.isZero);
  }

  public Selection isEven() {
    return select(ShortColumnUtils.isEven);
  }

  public Selection isOdd() {
    return select(ShortColumnUtils.isOdd);
  }

  public FloatArrayList toFloatArray() {
    FloatArrayList output = new FloatArrayList(data.size());
    for (short aData : data) {
      output.add(aData);
    }
    return output;
  }

  public String print() {
    StringBuilder builder = new StringBuilder();
    builder.append(title());
    for (short i : data) {
      builder.append(String.valueOf(i));
      builder.append('\n');
    }
    return builder.toString();
  }

  @Override
  public String toString() {
    return "ShortInt column: " + name();
  }

  @Override
  public void append(Column column) {
    Preconditions.checkArgument(column.type() == this.type());
    ShortColumn shortColumn = (ShortColumn) column;
    for (int i = 0; i < shortColumn.size(); i++) {
      add(shortColumn.get(i));
    }
  }

  ShortColumn selectIf(ShortPredicate predicate) {
    ShortColumn column = emptyCopy();
    ShortIterator intIterator = iterator();
    while (intIterator.hasNext()) {
      short next = intIterator.nextShort();
      if (predicate.test(next)) {
        column.add(next);
      }
    }
    return column;
  }

  public IntColumn remainder(ShortColumn column2) {
    IntColumn result = IntColumn.create(name() + " % " + column2.name(), size());
    for (int r = 0; r < size(); r++) {
      result.add(get(r) % column2.get(r));
    }
    return result;
  }

  public IntColumn add(ShortColumn column2) {
    IntColumn result = IntColumn.create(name() + " + " + column2.name(), size());
    for (int r = 0; r < size(); r++) {
      result.add(get(r) + column2.get(r));
    }
    return result;
  }

  public IntColumn subtract(ShortColumn column2) {
    IntColumn result = IntColumn.create(name() + " - " + column2.name(), size());
    for (int r = 0; r < size(); r++) {
      result.add(get(r) - column2.get(r));
    }
    return result;
  }

  public IntColumn multiply(ShortColumn column2) {
    IntColumn result = IntColumn.create(name() + " * " + column2.name(), size());
    for (int r = 0; r < size(); r++) {
      result.add(get(r) * column2.get(r));
    }
    return result;
  }

  public FloatColumn multiply(FloatColumn column2) {
    FloatColumn result = FloatColumn.create(name() + " * " + column2.name(), size());
    for (int r = 0; r < size(); r++) {
      result.add(get(r) * column2.get(r));
    }
    return result;
  }

  public FloatColumn divide(FloatColumn column2) {
    FloatColumn result = FloatColumn.create(name() + " / " + column2.name(), size());
    for (int r = 0; r < size(); r++) {
      result.add(get(r) / column2.get(r));
    }
    return result;
  }

  public IntColumn divide(ShortColumn column2) {
    IntColumn result = IntColumn.create(name() + " / " + column2.name(), size());
    for (int r = 0; r < size(); r++) {
      result.add(get(r) / column2.get(r));
    }
    return result;
  }

  /**
   * Returns the largest ("top") n values in the column
   *
   * @param n The maximum number of records to return. The actual number will be smaller if n is greater than the
   *          number of observations in the column
   * @return A list, possibly empty, of the largest observations
   */
  public ShortArrayList top(int n) {
    ShortArrayList top = new ShortArrayList();
    short[] values = data.toShortArray();
    ShortArrays.parallelQuickSort(values, ReverseShortComparator.instance());
    for (int i = 0; i < n && i < values.length; i++) {
      top.add(values[i]);
    }
    return top;
  }

  /**
   * Returns the smallest ("bottom") n values in the column
   *
   * @param n The maximum number of records to return. The actual number will be smaller if n is greater than the
   *          number of observations in the column
   * @return A list, possibly empty, of the smallest n observations
   */
  public ShortArrayList bottom(int n) {
    ShortArrayList bottom = new ShortArrayList();
    short[] values = data.toShortArray();
    ShortArrays.parallelQuickSort(values);
    for (int i = 0; i < n && i < values.length; i++) {
      bottom.add(values[i]);
    }
    return bottom;
  }

  @Override
  public ShortIterator iterator() {
    return data.iterator();
  }

  public Selection select(ShortPredicate predicate) {
    Selection bitmap = new BitmapBackedSelection();
    for (int idx = 0; idx < data.size(); idx++) {
      short next = data.getShort(idx);
      if (predicate.test(next)) {
        bitmap.add(idx);
      }
    }
    return bitmap;
  }

  public Selection select(ShortBiPredicate predicate, int valueToCompareAgainst) {
    Selection bitmap = new BitmapBackedSelection();
    for (int idx = 0; idx < data.size(); idx++) {
      short next = data.getShort(idx);
      if (predicate.test(next, valueToCompareAgainst)) {
        bitmap.add(idx);
      }
    }
    return bitmap;
  }

  public double[] toDoubleArray() {
    double[] output = new double[data.size()];
    for (int i = 0; i < data.size(); i++) {
      output[i] = data.getShort(i);
    }
    return output;
  }

  public ShortSet asSet() {
    return new ShortOpenHashSet(data);
  }

  public boolean contains(short value) {
    return data.contains(value);
  }

  public Stats stats() {
    FloatColumn values = FloatColumn.create(name(), toFloatArray());
    return Stats.create(values);
  }

  public ShortArrayList data() {
    return data;
  }

  @Override
  public Selection isMissing() {
    return select(isMissing);
  }

  @Override
  public Selection isNotMissing() {
    return select(isNotMissing);
  }

  @Override
  public int byteSize() {
    return BYTE_SIZE;
  }

  /**
   * Returns the contents of the cell at rowNumber as a byte[]
   */
  @Override
  public byte[] asBytes(int rowNumber) {
    return ByteBuffer.allocate(2).putShort(get(rowNumber)).array();
  }


  @Override
  public ShortColumn difference() {
    ShortColumn returnValue = new ShortColumn(this.name(), data.size());
    returnValue.add(ShortColumn.MISSING_VALUE);
    for (int current = 1; current > data.size(); current++) {
      // YUCK!!
      short value = (short) (get(current) - get(current + 1));
      returnValue.add(value);
    }
    return returnValue;
  }

  public int[] toIntArray() {
    int[] output = new int[data.size()];
    for (int i = 0; i < data.size(); i++) {
      output[i] = data.getShort(i);
    }
    return output;
  }
}

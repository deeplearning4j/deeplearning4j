package org.datavec.dataframe.api;

import org.datavec.dataframe.columns.AbstractColumn;
import org.datavec.dataframe.columns.Column;
import org.datavec.dataframe.columns.packeddata.PackedLocalTime;
import org.datavec.dataframe.filtering.IntBiPredicate;
import org.datavec.dataframe.filtering.IntPredicate;
import org.datavec.dataframe.filtering.LocalTimePredicate;
import org.datavec.dataframe.io.TypeUtils;
import org.datavec.dataframe.mapping.TimeMapUtils;
import org.datavec.dataframe.store.ColumnMetadata;
import org.datavec.dataframe.util.BitmapBackedSelection;
import org.datavec.dataframe.util.ReverseIntComparator;
import org.datavec.dataframe.util.Selection;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntArrays;
import it.unimi.dsi.fastutil.ints.IntComparator;
import it.unimi.dsi.fastutil.ints.IntIterator;
import it.unimi.dsi.fastutil.ints.IntOpenHashSet;
import it.unimi.dsi.fastutil.ints.IntSet;

import java.nio.ByteBuffer;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

/**
 * A column in a base table that contains float values
 */
public class TimeColumn extends AbstractColumn implements Iterable<LocalTime>, TimeMapUtils {

  public static final int MISSING_VALUE = (int) ColumnType.LOCAL_TIME.getMissingValue();
  private static final int BYTE_SIZE = 4;

  private static int DEFAULT_ARRAY_SIZE = 128;

  /**
   * The formatter chosen to parse times for this particular column
   */
  private DateTimeFormatter selectedFormatter;

  private IntArrayList data;

  public static TimeColumn create(String name) {
    return new TimeColumn(name);
  }

  public static TimeColumn create(String fileName, IntArrayList times) {
    TimeColumn column = new TimeColumn(fileName, times.size());
    column.data.addAll(times);
    return column;
  }

  private TimeColumn(String name) {
    super(name);
    data = new IntArrayList(DEFAULT_ARRAY_SIZE);
  }

  public TimeColumn(ColumnMetadata metadata) {
    super(metadata);
    data = new IntArrayList(DEFAULT_ARRAY_SIZE);
  }

  public TimeColumn(String name, int initialSize) {
    super(name);
    data = new IntArrayList(initialSize);
  }

  public int size() {
    return data.size();
  }

  public void add(int f) {
    data.add(f);
  }

  @Override
  public ColumnType type() {
    return ColumnType.LOCAL_TIME;
  }

  @Override
  public String getString(int row) {
    return PackedLocalTime.toShortTimeString(getInt(row));
  }

  @Override
  public TimeColumn emptyCopy() {
    TimeColumn column = new TimeColumn(name(), DEFAULT_ARRAY_SIZE);
    column.setComment(comment());
    return column;
  }

  @Override
  public TimeColumn emptyCopy(int rowSize) {
    TimeColumn column = new TimeColumn(name(), rowSize);
    column.setComment(comment());
    return column;
  }

  @Override
  public void clear() {
    data.clear();
  }

  @Override
  public TimeColumn copy() {
    TimeColumn column = TimeColumn.create(name(), data);
    column.setComment(comment());
    return column;
  }

  @Override
  public void sortAscending() {
    Arrays.parallelSort(data.elements());
  }

  @Override
  public void sortDescending() {
    IntArrays.parallelQuickSort(data.elements(), reverseIntComparator);
  }

  IntComparator reverseIntComparator = new IntComparator() {

    @Override
    public int compare(Integer o2, Integer o1) {
      return (o1 < o2 ? -1 : (o1.equals(o2) ? 0 : 1));
    }

    @Override
    public int compare(int o2, int o1) {
      return (o1 < o2 ? -1 : (o1 == o2 ? 0 : 1));
    }
  };

  public LocalTime max() {
    int max;
    int missing = Integer.MIN_VALUE;
    if (!isEmpty()) {
      max = getInt(0);
    } else {
      return null;
    }
    for (int aData : data) {
      if (missing != aData) {
        max = (max > aData) ? max : aData;
      }
    }

    if (missing == max) {
      return null;
    }
    return PackedLocalTime.asLocalTime(max);
  }

  public LocalTime min() {
    int min;
    int missing = Integer.MIN_VALUE;

    if (!isEmpty()) {
      min = getInt(0);
    } else {
      return null;
    }
    for (int aData : data) {
      if (missing != aData) {
        min = (min < aData) ? min : aData;
      }
    }
    if (Integer.MIN_VALUE == min) {
      return null;
    }
    return PackedLocalTime.asLocalTime(min);
  }


  @Override
  public Table summary() {

    Table table = Table.create("Column: " + name());
    CategoryColumn measure = CategoryColumn.create("Measure");
    CategoryColumn value = CategoryColumn.create("Value");
    table.addColumn(measure);
    table.addColumn(value);

    measure.add("Count");
    value.add(String.valueOf(size()));

    measure.add("Missing");
    value.add(String.valueOf(countMissing()));

    measure.add("Earliest");
    value.add(String.valueOf(min()));

    measure.add("Latest");
    value.add(String.valueOf(max()));

    return table;
  }

  /**
   * Returns the count of missing values in this column
   */
  @Override
  public int countMissing() {
    int count = 0;
    for (int i = 0; i < size(); i++) {
      if (getInt(i) == MISSING_VALUE) {
        count++;
      }
    }
    return count;
  }

  @Override
  public int countUnique() {
    IntSet ints = new IntOpenHashSet(data);
    return ints.size();
  }

  @Override
  public TimeColumn unique() {
    IntSet ints = new IntOpenHashSet(data);
    return TimeColumn.create(name() + " Unique values", IntArrayList.wrap(ints.toIntArray()));
  }

  @Override
  public boolean isEmpty() {
    return data.isEmpty();
  }

  /**
   * Returns a PackedTime as converted from the given string
   *
   * @param value A string representation of a time
   * @throws DateTimeParseException if no parser can be found for the time format used
   */
  public int convert(String value) {
    if (Strings.isNullOrEmpty(value)
        || TypeUtils.MISSING_INDICATORS.contains(value)
        || value.equals("-1")) {
      return (int) ColumnType.LOCAL_TIME.getMissingValue();
    }
    value = Strings.padStart(value, 4, '0');
    if (selectedFormatter == null) {
      selectedFormatter = TypeUtils.getTimeFormatter(value);
    }
    LocalTime time;
    try {
      time = LocalTime.parse(value, selectedFormatter);
    } catch (DateTimeParseException e) {
      selectedFormatter = TypeUtils.TIME_FORMATTER;
      time = LocalTime.parse(value, selectedFormatter);
    }
    return PackedLocalTime.pack(time);
  }

  @Override
  public void addCell(String object) {
    try {
      add(convert(object));
    } catch (NullPointerException e) {
      throw new RuntimeException(name() + ": "
          + String.valueOf(object) + ": "
          + e.getMessage());
    }
  }

  public int getInt(int index) {
    return data.getInt(index);
  }

  public LocalTime get(int index) {
    return PackedLocalTime.asLocalTime(getInt(index));
  }

  @Override
  public IntComparator rowComparator() {
    return comparator;
  }

  IntComparator comparator = new IntComparator() {

    @Override
    public int compare(Integer r1, Integer r2) {
      return compare((int) r1, (int) r2);
    }

    @Override
    public int compare(int r1, int r2) {
      int f1 = getInt(r1);
      int f2 = getInt(r2);
      return Integer.compare(f1, f2);
    }
  };

  public Selection isEqualTo(LocalTime value) {
    Selection results = new BitmapBackedSelection();
    int packedLocalTime = PackedLocalTime.pack(value);
    int i = 0;
    for (int next : data) {
      if (packedLocalTime == next) {
        results.add(i);
      }
      i++;
    }
    return results;
  }

  public String print() {
    StringBuilder builder = new StringBuilder();
    builder.append(title());
    for (int next : data) {
      builder.append(String.valueOf(PackedLocalTime.asLocalTime(next)));
      builder.append('\n');
    }
    return builder.toString();
  }

  public IntArrayList data() {
    return data;
  }

  @Override
  public String toString() {
    return "LocalTime column: " + name();
  }

  public TimeColumn selectIf(LocalTimePredicate predicate) {
    TimeColumn column = emptyCopy();
    IntIterator iterator = intIterator();
    while (iterator.hasNext()) {
      int next = iterator.nextInt();
      if (predicate.test(PackedLocalTime.asLocalTime(next))) {
        column.add(next);
      }
    }
    return column;
  }

  /**
   * This version operates on predicates that treat the given IntPredicate as operating on a packed local time
   * This is much more efficient that using a LocalTimePredicate, but requires that the developer understand the
   * semantics of packedLocalTimes
   */
  public TimeColumn selectIf(IntPredicate predicate) {
    TimeColumn column = emptyCopy();
    IntIterator iterator = intIterator();
    while (iterator.hasNext()) {
      int next = iterator.nextInt();
      if (predicate.test(next)) {
        column.add(next);
      }
    }
    return column;
  }

  @Override
  public void append(Column column) {
    Preconditions.checkArgument(column.type() == this.type());
    TimeColumn intColumn = (TimeColumn) column;
    for (int i = 0; i < intColumn.size(); i++) {
      add(intColumn.getInt(i));
    }
  }

  public Selection isMidnight() {
    return select(PackedLocalTime::isMidnight);
  }

  public Selection isNoon() {
    return select(PackedLocalTime::isNoon);
  }

  public Selection isBefore(LocalTime time) {
    return select(PackedLocalTime::isBefore, PackedLocalTime.pack(time));
  }

  public Selection isAfter(LocalTime time) {
    return select(PackedLocalTime::isAfter, PackedLocalTime.pack(time));
  }

  /**
   * Applies a function to every value in this column that returns true if the time is in the AM or "before noon".
   * Note: we follow the convention that 12:00 NOON is PM and 12 MIDNIGHT is AM
   */
  public Selection isBeforeNoon() {
    return select(PackedLocalTime::AM);
  }

  /**
   * Applies a function to every value in this column that returns true if the time is in the PM or "after noon".
   * Note: we follow the convention that 12:00 NOON is PM and 12 MIDNIGHT is AM
   */
  public Selection isAfterNoon() {
    return select(PackedLocalTime::PM);
  }

  /**
   * Returns the largest ("top") n values in the column
   *
   * @param n The maximum number of records to return. The actual number will be smaller if n is greater than the
   *          number of observations in the column
   * @return A list, possibly empty, of the largest observations
   */
  public List<LocalTime> top(int n) {
    List<LocalTime> top = new ArrayList<>();
    int[] values = data.toIntArray();
    IntArrays.parallelQuickSort(values, ReverseIntComparator.instance());
    for (int i = 0; i < n && i < values.length; i++) {
      top.add(PackedLocalTime.asLocalTime(values[i]));
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
  public List<LocalTime> bottom(int n) {
    List<LocalTime> bottom = new ArrayList<>();
    int[] values = data.toIntArray();
    IntArrays.parallelQuickSort(values);
    for (int i = 0; i < n && i < values.length; i++) {
      bottom.add(PackedLocalTime.asLocalTime(values[i]));
    }
    return bottom;
  }

  public IntIterator intIterator() {
    return data.iterator();
  }

  public Selection select(IntPredicate predicate) {
    Selection selection = new BitmapBackedSelection();
    for (int idx = 0; idx < data.size(); idx++) {
      int next = data.getInt(idx);
      if (predicate.test(next)) {
        selection.add(idx);
      }
    }
    return selection;
  }

  public Selection select(IntBiPredicate predicate, int value) {
    Selection selection = new BitmapBackedSelection();
    for (int idx = 0; idx < data.size(); idx++) {
      int next = data.getInt(idx);
      if (predicate.test(next, value)) {
        selection.add(idx);
      }
    }
    return selection;
  }

  Set<LocalTime> asSet() {
    Set<LocalTime> times = new HashSet<>();
    TimeColumn unique = unique();
    for (LocalTime t : unique) {
      times.add(t);
    }
    return times;
  }

  public boolean contains(LocalTime time) {
    int t = PackedLocalTime.pack(time);
    return data().contains(t);
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
    return ByteBuffer.allocate(4).putInt(getInt(rowNumber)).array();
  }

  /**
   * Returns an iterator over elements of type {@code T}.
   *
   * @return an Iterator.
   */
  @Override
  public Iterator<LocalTime> iterator() {

    return new Iterator<LocalTime>() {

      IntIterator intIterator = intIterator();

      @Override
      public boolean hasNext() {
        return intIterator.hasNext();
      }

      @Override
      public LocalTime next() {
        return PackedLocalTime.asLocalTime(intIterator.next());
      }
    };
  }
}
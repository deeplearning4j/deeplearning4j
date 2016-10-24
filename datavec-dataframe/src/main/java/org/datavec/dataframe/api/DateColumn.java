package org.datavec.dataframe.api;

import org.datavec.dataframe.columns.AbstractColumn;
import org.datavec.dataframe.columns.Column;
import org.datavec.dataframe.columns.DateColumnUtils;
import org.datavec.dataframe.columns.IntColumnUtils;
import org.datavec.dataframe.columns.packeddata.PackedLocalDate;
import org.datavec.dataframe.columns.packeddata.PackedLocalDateTime;
import org.datavec.dataframe.columns.packeddata.PackedLocalTime;
import org.datavec.dataframe.filtering.IntBiPredicate;
import org.datavec.dataframe.filtering.IntPredicate;
import org.datavec.dataframe.filtering.LocalDatePredicate;
import org.datavec.dataframe.io.TypeUtils;
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
import java.time.LocalDate;
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
public class DateColumn extends AbstractColumn implements DateColumnUtils {

  public static final int MISSING_VALUE = (int) ColumnType.LOCAL_DATE.getMissingValue();

  private static final int DEFAULT_ARRAY_SIZE = 128;

  private static final int BYTE_SIZE = 4;

  private IntArrayList data;

  /**
   * The formatter chosen to parse dates for this particular column
   */
  private DateTimeFormatter selectedFormatter;

  private DateColumn(String name) {
    super(name);
    data = new IntArrayList(DEFAULT_ARRAY_SIZE);
  }

  public DateColumn(ColumnMetadata metadata) {
    super(metadata);
    data = new IntArrayList(DEFAULT_ARRAY_SIZE);
  }

  private DateColumn(String name, int initialSize) {
    super(name);
    data = new IntArrayList(initialSize);
  }

  public int size() {
    return data.size();
  }

  @Override
  public ColumnType type() {
    return ColumnType.LOCAL_DATE;
  }

  public void add(int f) {
    data.add(f);
  }

  public IntArrayList data() {
    return data;
  }

  public void set(int index, int value) {
    data.set(index, value);
  }

  public void add(LocalDate f) {
    add(PackedLocalDate.pack(f));
  }

  @Override
  public String getString(int row) {
    return PackedLocalDate.toDateString(getInt(row));
  }

  @Override
  public DateColumn emptyCopy() {
    DateColumn column = DateColumn.create(name());
    column.setComment(comment());
    return column;
  }

  @Override
  public DateColumn emptyCopy(int rowSize) {
    DateColumn column = new DateColumn(name(), rowSize);
    column.setComment(comment());
    return column;
  }

  @Override
  public void clear() {
    data.clear();
  }

  @Override
  public DateColumn copy() {
    DateColumn column = DateColumn.create(name(), data);
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

  @Override
  public int countUnique() {
    IntSet ints = new IntOpenHashSet(size());
    for (int i = 0; i < size(); i++) {
      ints.add(data.getInt(i));
    }
    return ints.size();
  }

  @Override
  public DateColumn unique() {
    IntSet ints = new IntOpenHashSet(data.size());
    for (int i = 0; i < size(); i++) {
      ints.add(data.getInt(i));
    }
    return DateColumn.create(name() + " Unique values", IntArrayList.wrap(ints.toIntArray()));
  }

  public LocalDate firstElement() {
    if (isEmpty()) {
      return null;
    }
    return PackedLocalDate.asLocalDate(getInt(0));
  }

  public LocalDate max() {
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
    return PackedLocalDate.asLocalDate(max);
  }

  public LocalDate min() {
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
    return PackedLocalDate.asLocalDate(min);
  }

  public CategoryColumn dayOfWeek() {
    CategoryColumn newColumn = CategoryColumn.create(this.name() + " day of week");
    for (int r = 0; r < this.size(); r++) {
      int c1 = this.getInt(r);
      if (c1 == DateColumn.MISSING_VALUE) {
        newColumn.add(null);
      } else {
        newColumn.add(PackedLocalDate.getDayOfWeek(c1).toString());
      }
    }
    return newColumn;
  }

  public ShortColumn dayOfWeekValue() {
    ShortColumn newColumn = ShortColumn.create(this.name() + " day of week", this.size());
    for (int r = 0; r < this.size(); r++) {
      int c1 = this.getInt(r);
      if (c1 == (DateColumn.MISSING_VALUE)) {
        newColumn.set(r, ShortColumn.MISSING_VALUE);
      } else {
        newColumn.add((short) PackedLocalDate.getDayOfWeek(c1).getValue());
      }
    }
    return newColumn;
  }


  public ShortColumn dayOfMonth() {
    ShortColumn newColumn = ShortColumn.create(this.name() + " day of month");
    for (int r = 0; r < this.size(); r++) {
      int c1 = this.getInt(r);
      if (c1 == DateColumn.MISSING_VALUE) {
        newColumn.add(ShortColumn.MISSING_VALUE);
      } else {
        newColumn.add(PackedLocalDate.getDayOfMonth(c1));
      }
    }
    return newColumn;
  }

  public ShortColumn dayOfYear() {
    ShortColumn newColumn = ShortColumn.create(this.name() + " day of month");
    for (int r = 0; r < this.size(); r++) {
      int c1 = this.getInt(r);
      if (c1 == DateColumn.MISSING_VALUE) {
        newColumn.add(ShortColumn.MISSING_VALUE);
      } else {
        newColumn.add((short) PackedLocalDate.getDayOfYear(c1));
      }
    }
    return newColumn;
  }

  public ShortColumn monthValue() {
    ShortColumn newColumn = ShortColumn.create(this.name() + " month");

    for (int r = 0; r < this.size(); r++) {
      int c1 = this.getInt(r);
      if (c1 == DateColumn.MISSING_VALUE) {
        newColumn.add(ShortColumn.MISSING_VALUE);
      } else {
        newColumn.add(PackedLocalDate.getMonthValue(c1));
      }
    }
    return newColumn;
  }

  public CategoryColumn month() {
    CategoryColumn newColumn = CategoryColumn.create(this.name() + " month");

    for (int r = 0; r < this.size(); r++) {
      int c1 = this.getInt(r);
      if (c1 == DateColumn.MISSING_VALUE) {
        newColumn.add(CategoryColumn.MISSING_VALUE);
      } else {
        newColumn.add(PackedLocalDate.getMonth(c1).name());
      }
    }
    return newColumn;
  }

  public ShortColumn year() {
    ShortColumn newColumn = ShortColumn.create(this.name() + " year");
    for (int r = 0; r < this.size(); r++) {
      int c1 = this.getInt(r);
      if (c1 == MISSING_VALUE) {
        newColumn.add(ShortColumn.MISSING_VALUE);
      } else {
        newColumn.add(PackedLocalDateTime.getYear(PackedLocalDateTime.date(c1)));
      }
    }
    return newColumn;
  }

  public LocalDate get(int index) {
    return PackedLocalDate.asLocalDate(getInt(index));
  }

  public static DateColumn create(String name) {
    return new DateColumn(name);
  }

  @Override
  public boolean isEmpty() {
    return data.isEmpty();
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

  public static DateColumn create(String columnName, IntArrayList dates) {
    DateColumn column = new DateColumn(columnName, dates.size());
    column.data.addAll(dates);
    return column;
  }

  /**
   * Returns a PackedDate as converted from the given string
   *
   * @param value A string representation of a date
   * @throws DateTimeParseException if no parser can be found for the date format
   */
  public int convert(String value) {
    if (Strings.isNullOrEmpty(value) || TypeUtils.MISSING_INDICATORS.contains(value) || value.equals("-1")) {
      return (int) ColumnType.LOCAL_DATE.getMissingValue();
    }
    value = Strings.padStart(value, 4, '0');

    if (selectedFormatter == null) {
      selectedFormatter = TypeUtils.getDateFormatter(value);
    }
    LocalDate date;
    try {
      date = LocalDate.parse(value, selectedFormatter);
    } catch (DateTimeParseException e) {
      selectedFormatter = TypeUtils.DATE_FORMATTER;
      date = LocalDate.parse(value, selectedFormatter);
    }
    return PackedLocalDate.pack(date);
  }

  public void addCell(String string) {
    try {
      add(convert(string));
    } catch (NullPointerException e) {
      throw new RuntimeException(name() + ": " + string + ": " + e.getMessage());
    }
  }

  public int getInt(int index) {
    return data.getInt(index);
  }

  public Selection isEqualTo(LocalDate value) {
    int packed = PackedLocalDate.pack(value);
    return select(IntColumnUtils.isEqualTo, packed);
  }

  /**
   * Returns a bitmap flagging the records for which the value in this column is equal to the value in the given
   * column
   * Columnwise isEqualTo.
   */
  public Selection isEqualTo(DateColumn column) {
    Selection results = new BitmapBackedSelection();
    int i = 0;
    IntIterator intIterator = column.intIterator();
    for (int next : data) {
      if (next == intIterator.nextInt()) {
        results.add(i);
      }
      i++;
    }
    return results;
  }

  /**
   * Returns a table of dates and the number of observations of those dates
   */
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
   * Returns a DateTime column where each value consists of the dates from this column combined with the corresponding
   * times from the other column
   */
  public DateTimeColumn atTime(TimeColumn c) {
    DateTimeColumn newColumn = DateTimeColumn.create(this.name() + " " + c.name());
    for (int r = 0; r < this.size(); r++) {
      int c1 = this.getInt(r);
      int c2 = c.getInt(r);
      if (c1 == MISSING_VALUE || c2 == TimeColumn.MISSING_VALUE) {
        newColumn.add(DateTimeColumn.MISSING_VALUE);
      } else {
        LocalDate value1 = PackedLocalDate.asLocalDate(c1);
        LocalTime time = PackedLocalTime.asLocalTime(c2);
        newColumn.add(PackedLocalDateTime.pack(value1, time));
      }
    }
    return newColumn;
  }

  public Selection isAfter(int value) {
    return select(PackedLocalDate::isAfter, value);
  }

  public Selection isAfter(LocalDate value) {
    int packed = PackedLocalDate.pack(value);
    return select(PackedLocalDate::isAfter, packed);
  }

  public Selection isBefore(int value) {
    return select(PackedLocalDate::isBefore, value);
  }

  public Selection isBefore(LocalDate value) {
    int packed = PackedLocalDate.pack(value);
    return select(PackedLocalDate::isBefore, packed);
  }

  public Selection isOnOrBefore(LocalDate value) {
    int packed = PackedLocalDate.pack(value);
    return select(PackedLocalDate::isOnOrBefore, packed);
  }

  public Selection isOnOrBefore(int value) {
    return select(PackedLocalDate::isOnOrBefore, value);
  }

  public Selection isOnOrAfter(LocalDate value) {
    int packed = PackedLocalDate.pack(value);
    return select(PackedLocalDate::isOnOrAfter, packed);
  }

  public Selection isOnOrAfter(int value) {
    return select(PackedLocalDate::isOnOrAfter, value);
  }

  public Selection isMonday() {
    return select(PackedLocalDate::isMonday);
  }

  public Selection isTuesday() {
    return select(PackedLocalDate::isTuesday);
  }

  public Selection isWednesday() {
    return select(PackedLocalDate::isWednesday);
  }

  public Selection isThursday() {
    return select(PackedLocalDate::isThursday);
  }

  public Selection isFriday() {
    return select(PackedLocalDate::isFriday);
  }

  public Selection isSaturday() {
    return select(PackedLocalDate::isSaturday);
  }

  public Selection isSunday() {
    return select(PackedLocalDate::isSunday);
  }

  public Selection isInJanuary() {
    return select(PackedLocalDate::isInJanuary);
  }

  public Selection isInFebruary() {
    return select(PackedLocalDate::isInFebruary);
  }

  public Selection isInMarch() {
    return select(PackedLocalDate::isInMarch);
  }

  public Selection isInApril() {
    return select(PackedLocalDate::isInApril);
  }

  public Selection isInMay() {
    return select(PackedLocalDate::isInMay);
  }

  public Selection isInJune() {
    return select(PackedLocalDate::isInJune);
  }

  public Selection isInJuly() {
    return select(PackedLocalDate::isInJuly);
  }

  public Selection isInAugust() {
    return select(PackedLocalDate::isInAugust);
  }

  public Selection isInSeptember() {
    return select(PackedLocalDate::isInSeptember);
  }

  public Selection isInOctober() {
    return select(PackedLocalDate::isInOctober);
  }

  public Selection isInNovember() {
    return select(PackedLocalDate::isInNovember);
  }

  public Selection isInDecember() {
    return select(PackedLocalDate::isInDecember);
  }

  public Selection isFirstDayOfMonth() {
    return select(PackedLocalDate::isFirstDayOfMonth);
  }

  public Selection isLastDayOfMonth() {
    return select(PackedLocalDate::isLastDayOfMonth);
  }

  public Selection isInQ1() {
    return select(PackedLocalDate::isInQ1);
  }

  public Selection isInQ2() {
    return select(PackedLocalDate::isInQ2);
  }

  public Selection isInQ3() {
    return select(PackedLocalDate::isInQ3);
  }

  public Selection isInQ4() {
    return select(PackedLocalDate::isInQ4);
  }

  public Selection isInYear(int year) {
    return select(PackedLocalDate::isInYear, year);
  }

  public String print() {
    StringBuilder builder = new StringBuilder();
    builder.append(title());
    for (int next : data) {
      builder.append(String.valueOf(PackedLocalDate.asLocalDate(next)));
      builder.append('\n');
    }
    return builder.toString();
  }

  @Override
  public Selection isMissing() {
    return select(isMissing);
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
  public Selection isNotMissing() {
    return select(isNotMissing);
  }

  @Override
  public String toString() {
    return "LocalDate column: " + name();
  }

  @Override
  public void append(Column column) {
    Preconditions.checkArgument(column.type() == this.type());
    DateColumn intColumn = (DateColumn) column;
    for (int i = 0; i < intColumn.size(); i++) {
      add(intColumn.getInt(i));
    }
  }

  public DateColumn selectIf(LocalDatePredicate predicate) {
    DateColumn column = emptyCopy();
    IntIterator iterator = intIterator();
    while (iterator.hasNext()) {
      int next = iterator.nextInt();
      if (predicate.test(PackedLocalDate.asLocalDate(next))) {
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
  public DateColumn selectIf(IntPredicate predicate) {
    DateColumn column = emptyCopy();
    IntIterator iterator = intIterator();
    while (iterator.hasNext()) {
      int next = iterator.nextInt();
      if (predicate.test(next)) {
        column.add(next);
      }
    }
    return column;
  }

  /**
   * Returns the largest ("top") n values in the column
   *
   * @param n The maximum number of records to return. The actual number will be smaller if n is greater than the
   *          number of observations in the column
   * @return A list, possibly empty, of the largest observations
   */
  public List<LocalDate> top(int n) {
    List<LocalDate> top = new ArrayList<>();
    int[] values = data.toIntArray();
    IntArrays.parallelQuickSort(values, ReverseIntComparator.instance());
    for (int i = 0; i < n && i < values.length; i++) {
      top.add(PackedLocalDate.asLocalDate(values[i]));
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
  public List<LocalDate> bottom(int n) {
    List<LocalDate> bottom = new ArrayList<>();
    int[] values = data.toIntArray();
    IntArrays.parallelQuickSort(values);
    for (int i = 0; i < n && i < values.length; i++) {
      bottom.add(PackedLocalDate.asLocalDate(values[i]));
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

  public Set<LocalDate> asSet() {
    Set<LocalDate> dates = new HashSet<>();
    DateColumn unique = unique();
    for (LocalDate d : unique) {
      dates.add(d);
    }
    return dates;
  }

  public DateTimeColumn with(TimeColumn timeColumn) {
    String dateTimeColumnName = name() + " : " + timeColumn.name();
    DateTimeColumn dateTimeColumn = new DateTimeColumn(dateTimeColumnName, size());
    for (int row = 0; row < size(); row++) {
      int date = getInt(row);
      int time = timeColumn.getInt(row);
      long packedLocalDateTime = PackedLocalDateTime.create(date, time);
      dateTimeColumn.add(packedLocalDateTime);
    }
    return dateTimeColumn;
  }

  public boolean contains(LocalDate localDate) {
    int date = PackedLocalDate.pack(localDate);
    return data().contains(date);
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
  public Iterator<LocalDate> iterator() {

    return new Iterator<LocalDate>() {

      IntIterator intIterator = intIterator();

      @Override
      public boolean hasNext() {
        return intIterator.hasNext();
      }

      @Override
      public LocalDate next() {
        return PackedLocalDate.asLocalDate(intIterator.next());
      }
    };
  }

  @Override
  public DateColumn difference() {
    throw new UnsupportedOperationException("DateTimeColumn.difference() currently not supported");
/*
        DateColumn returnValue = new DateColumn(this.name(), data.size());
        returnValue.add(DateColumn.MISSING_VALUE);
        for (int current = 1; current > data.size(); current++) {
            LocalDate currentValue = get(current);
            LocalDate nextValue = get(current+1);
            Duration duration = Duration.between(currentValue, nextValue);
            LocalDateTime date =
                    LocalDateTime.ofInstant(Instant.ofEpochMilli(duration.toMillis()), ZoneId.systemDefault());
            returnValue.add(date.toLocalDate());
        }
        return returnValue;
  */
  }

}
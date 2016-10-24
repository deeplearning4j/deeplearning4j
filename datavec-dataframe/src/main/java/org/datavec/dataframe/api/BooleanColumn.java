package org.datavec.dataframe.api;

import org.datavec.dataframe.columns.AbstractColumn;
import org.datavec.dataframe.columns.Column;
import org.datavec.dataframe.filtering.BooleanPredicate;
import org.datavec.dataframe.io.TypeUtils;
import org.datavec.dataframe.mapping.BooleanMapUtils;
import org.datavec.dataframe.store.ColumnMetadata;
import org.datavec.dataframe.util.BitmapBackedSelection;
import org.datavec.dataframe.util.Selection;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import it.unimi.dsi.fastutil.booleans.BooleanOpenHashSet;
import it.unimi.dsi.fastutil.booleans.BooleanSet;
import it.unimi.dsi.fastutil.bytes.Byte2IntMap;
import it.unimi.dsi.fastutil.bytes.Byte2IntOpenHashMap;
import it.unimi.dsi.fastutil.bytes.ByteArrayList;
import it.unimi.dsi.fastutil.bytes.ByteArrays;
import it.unimi.dsi.fastutil.bytes.ByteComparator;
import it.unimi.dsi.fastutil.bytes.ByteIterator;
import it.unimi.dsi.fastutil.bytes.ByteOpenHashSet;
import it.unimi.dsi.fastutil.bytes.ByteSet;
import it.unimi.dsi.fastutil.ints.IntComparator;
import it.unimi.dsi.fastutil.ints.IntIterator;

import java.util.Iterator;
import java.util.Map;

import static org.datavec.dataframe.columns.BooleanColumnUtils.isMissing;
import static org.datavec.dataframe.columns.BooleanColumnUtils.isNotMissing;

/**
 * A column in a base table that contains float values
 */
public class BooleanColumn extends AbstractColumn implements BooleanMapUtils {

  public static final byte MISSING_VALUE = Byte.MIN_VALUE;

  private static final int BYTE_SIZE = 1;

  private static int DEFAULT_ARRAY_SIZE = 128;

  private ByteArrayList data;

  public static BooleanColumn create(String name) {
    return new BooleanColumn(name);
  }

  public static BooleanColumn create(String name, int rowSize) {
    return new BooleanColumn(name, rowSize);
  }

  public static BooleanColumn create(String name, Selection selection, int rowSize) {
    return new BooleanColumn(name, selection, rowSize);
  }

  public BooleanColumn(ColumnMetadata metadata) {
    super(metadata);
    data = new ByteArrayList(DEFAULT_ARRAY_SIZE);
  }

  private BooleanColumn(String name) {
    super(name);
    data = new ByteArrayList(DEFAULT_ARRAY_SIZE);
  }

  public BooleanColumn(String name, int initialSize) {
    super(name);
    data = new ByteArrayList(initialSize);
  }

  private BooleanColumn(String name, ByteArrayList values) {
    super(name);
    data = values;
  }

  public BooleanColumn(String name, Selection hits, int columnSize) {
    super(name);
    if (columnSize == 0) {
      return;
    }
    ByteArrayList data = new ByteArrayList(columnSize);

    for (int i = 0; i < columnSize; i++) {
      data.add((byte) 0);
    }

    IntIterator intIterator = hits.iterator();
    while (intIterator.hasNext()) {
      byte b = (byte) 1;
      int i = intIterator.next();
      data.set(i, b);
    }
    this.data = data;
  }

  public int size() {
    return data.size();
  }

  @Override
  public Table summary() {

    Byte2IntMap counts = new Byte2IntOpenHashMap(3);
    counts.put((byte) 0, 0);
    counts.put((byte) 1, 0);

    for (byte next : data) {
      counts.put(next, counts.get(next) + 1);
    }

    Table table = Table.create(name());

    BooleanColumn booleanColumn = BooleanColumn.create("Value");
    IntColumn countColumn = IntColumn.create("Count");
    table.addColumn(booleanColumn);
    table.addColumn(countColumn);

    for (Map.Entry<Byte, Integer> entry : counts.entrySet()) {
      booleanColumn.add(entry.getKey());
      countColumn.add(entry.getValue());
    }
    return table;
  }

  /**
   * Returns the count of missing values in this column
   */
  @Override
  public int countMissing() {
    int count = 0;
    for (int i = 0; i < size(); i++) {
      if (getByte(i) == MISSING_VALUE) {
        count++;
      }
    }
    return count;
  }

  @Override
  public int countUnique() {
    ByteSet count = new ByteOpenHashSet(3);
    for (byte next : data) {
      count.add(next);
    }
    return count.size();
  }

  @Override
  public BooleanColumn unique() {
    ByteSet count = new ByteOpenHashSet(3);
    for (byte next : data) {
      count.add(next);
    }
    ByteArrayList list = new ByteArrayList(count);
    return new BooleanColumn(name() + " Unique values", list);
  }

  @Override
  public ColumnType type() {
    return ColumnType.BOOLEAN;
  }

  public void add(boolean b) {
    if (b) {
      data.add((byte) 1);
    } else {
      data.add((byte) 0);
    }
  }

  public void add(byte b) {
    data.add(b);
  }

  @Override
  public String getString(int row) {
    return String.valueOf(get(row));
  }

  @Override
  public BooleanColumn emptyCopy() {
    BooleanColumn column = BooleanColumn.create(name());
    column.setComment(comment());
    return column;
  }

  @Override
  public BooleanColumn emptyCopy(int rowSize) {
    BooleanColumn column = BooleanColumn.create(name(), rowSize);
    column.setComment(comment());
    return column;
  }

  @Override
  public void clear() {
    data.clear();
  }

  @Override
  public BooleanColumn copy() {
    BooleanColumn column = BooleanColumn.create(name(), data);
    column.setComment(comment());
    return column;
  }

  @Override
  public void sortAscending() {
    ByteArrays.mergeSort(data.elements());
  }

  @Override
  public void sortDescending() {
    ByteArrays.mergeSort(data.elements(), reverseByteComparator);
  }

  ByteComparator reverseByteComparator = new ByteComparator() {

    @Override
    public int compare(Byte o1, Byte o2) {
      return Byte.compare(o2, o1);
    }

    @Override
    public int compare(byte o1, byte o2) {
      return Byte.compare(o2, o1);
    }
  };

  public static boolean convert(String stringValue) {
    if (Strings.isNullOrEmpty(stringValue) || TypeUtils.MISSING_INDICATORS.contains(stringValue)) {
      return (boolean) ColumnType.BOOLEAN.getMissingValue();
    } else if (TypeUtils.TRUE_STRINGS.contains(stringValue)) {
      return true;
    } else if (TypeUtils.FALSE_STRINGS.contains(stringValue)) {
      return false;
    } else {
      throw new IllegalArgumentException("Attempting to convert non-boolean value " +
          stringValue + " to Boolean");
    }
  }

  public void addCell(String object) {
    try {
      add(convert(object));
    } catch (NullPointerException e) {
      throw new RuntimeException(name() + ": "
          + String.valueOf(object) + ": "
          + e.getMessage());
    }
  }

  /**
   * Returns the value in row i as a Boolean
   * @param i the row number
   * @return  A Boolean object (may be null)
   */
  public Boolean get(int i) {
    byte b = data.getByte(i);
    if (b == 1) {
      return Boolean.TRUE;
    }
    if (b == 0) {
      return Boolean.FALSE;
    }
    return null;
  }

  /**
   * Returns the value in row i as a byte (0, 1, or Byte.MIN_VALUE representing missing data)
   * @param i the row number
   */
  public byte getByte(int i) {
    return data.getByte(i);
  }

  @Override
  public boolean isEmpty() {
    return data.isEmpty();
  }

  public static BooleanColumn create(String fileName, ByteArrayList bools) {
    BooleanColumn booleanColumn = new BooleanColumn(fileName, bools.size());
    booleanColumn.data.addAll(bools);
    return booleanColumn;
  }

  public int countTrue() {
    int count = 0;
    for (byte b : data) {
      if (b == 1) {
        count++;
      }
    }
    return count;
  }

  public int countFalse() {
    int count = 0;
    for (byte b : data) {
      if (b == 0) {
        count++;
      }
    }
    return count;
  }

  public Selection isFalse() {
    Selection results = new BitmapBackedSelection();
    int i = 0;
    for (byte next : data) {
      if (next == 0) {
        results.add(i);
      }
      i++;
    }
    return results;
  }

  public Selection isTrue() {
    Selection results = new BitmapBackedSelection();
    int i = 0;
    for (byte next : data) {
      if (next == 1) {
        results.add(i);
      }
      i++;
    }
    return results;
  }

  public Selection isEqualTo(BooleanColumn other) {
    Selection results = new BitmapBackedSelection();
    int i = 0;
    ByteIterator booleanIterator = other.byteIterator();
    for (byte next : data) {
      if (next == booleanIterator.nextByte()) {
        results.add(i);
      }
      i++;
    }
    return results;
  }

  /**
   * Returns a ByteArrayList containing 0 (false), 1 (true) or Byte.MIN_VALUE (missing)
   */
  public ByteArrayList data() {
    return data;
  }

  public void set(int i, boolean b) {
    data.set(i, b ? (byte) 1 : (byte) 0);
  }

  @Override
  public IntComparator rowComparator() {
    return comparator;
  }

  @Override
  public void append(Column column) {
    Preconditions.checkArgument(column.type() == this.type());
    BooleanColumn booleanColumn = (BooleanColumn) column;
    for (int i = 0; i < booleanColumn.size(); i++) {
      add(booleanColumn.get(i));
    }
  }

  IntComparator comparator = new IntComparator() {

    @Override
    public int compare(Integer r1, Integer r2) {
      return compare((int) r1, (int) r2);
    }

    @Override
    public int compare(int r1, int r2) {
      boolean f1 = get(r1);
      boolean f2 = get(r2);
      return Boolean.compare(f1, f2);
    }
  };

  // TODO(lwhite): this won't scale
  public String print() {
    StringBuilder builder = new StringBuilder();
    builder.append(title());
    for (byte next : data) {
      if (next == (byte) 0) {
        builder.append(String.valueOf(false));
      } else if (next == (byte) 1) {
        builder.append(String.valueOf(true));
      } else {
        builder.append(String.valueOf("NA"));
      }
      builder.append('\n');
    }
    return builder.toString();
  }

  @Override
  public Selection isMissing() {  //TODO
    return select(isMissing);
  }

  @Override
  public Selection isNotMissing() { //TODO
    return select(isNotMissing);
  }

  public Iterator<Boolean> iterator() {
    return new BooleanColumnIterator(this.byteIterator());
  }

  public ByteIterator byteIterator() {
    return data.iterator();
  }

  @Override
  public String toString() {
    return "Boolean column: " + name();
  }

  public BooleanSet asSet() {
    BooleanSet set = new BooleanOpenHashSet(3);
    BooleanColumn unique = unique();
    for (int i = 0; i < unique.size(); i++) {
      set.add(unique.get(i));
    }
    return set;
  }

  public boolean contains(boolean aBoolean) {
    return data().contains(aBoolean);
  }

  @Override
  public int byteSize() {
    return BYTE_SIZE;
  }

  @Override
  public byte[] asBytes(int row) {
    byte[] result = new byte[1];
    result[0] = (byte) (get(row) ? 1 : 0);
    return result;
  }

  public Selection select(BooleanPredicate predicate) {
    Selection selection = new BitmapBackedSelection();
    for (int idx = 0; idx < data.size(); idx++) {
      byte next = data.getByte(idx);
      if (predicate.test(next)) {
        selection.add(idx);
      }
    }
    return selection;
  }

  public int[] toIntArray() {
    int[] output = new int[data.size()];
    for (int i = 0; i < data.size(); i++) {
      output[i] = data.getByte(i);
    }
    return output;
  }

  public IntColumn toIntColumn() {
    IntColumn intColumn = IntColumn.create(this.name() + ": ints", size());
    ByteArrayList data = data();
    for (int i = 0; i < size(); i++) {
      intColumn.add(data.getByte(i));
    }
    return intColumn;
  }

  static class BooleanColumnIterator implements Iterator<Boolean> {

    final ByteIterator iterator;

    public BooleanColumnIterator(ByteIterator iterator) {
      this.iterator = iterator;
    }

    /**
     * Returns {@code true} if the iteration has more elements.
     * (In other words, returns {@code true} if {@link #next} would
     * return an element rather than throwing an exception.)
     *
     * @return {@code true} if the iteration has more elements
     */
    @Override
    public boolean hasNext() {
      return iterator.hasNext();
    }

    /**
     * Returns the next element in the iteration.
     *
     * @return the next element in the iteration
     * @throws java.util.NoSuchElementException if the iteration has no more elements
     */
    @Override
    public Boolean next() {
      byte b = iterator.next();
      if (b == (byte) 0) {
        return false;
      }
      if (b == (byte) 1) {
        return true;
      }
      return null;
    }
  }

}

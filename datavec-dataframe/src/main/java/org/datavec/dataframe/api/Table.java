package org.datavec.dataframe.api;

import org.datavec.dataframe.columns.Column;
import org.datavec.dataframe.filtering.Filter;
import org.datavec.dataframe.io.csv.CsvReader;
import org.datavec.dataframe.io.csv.CsvWriter;
import org.datavec.dataframe.io.html.HtmlTableWriter;
import org.datavec.dataframe.io.jdbc.SqlResultSetReader;
import org.datavec.dataframe.reducing.NumericReduceFunction;
import org.datavec.dataframe.reducing.functions.Count;
import org.datavec.dataframe.reducing.functions.Maximum;
import org.datavec.dataframe.reducing.functions.Mean;
import org.datavec.dataframe.reducing.functions.Median;
import org.datavec.dataframe.reducing.functions.Minimum;
import org.datavec.dataframe.reducing.functions.StandardDeviation;
import org.datavec.dataframe.reducing.functions.Sum;
import org.datavec.dataframe.reducing.functions.SummaryFunction;
import org.datavec.dataframe.reducing.functions.Variance;
import org.datavec.dataframe.sorting.Sort;
import org.datavec.dataframe.store.StorageManager;
import org.datavec.dataframe.store.TableMetadata;
import org.datavec.dataframe.table.Projection;
import org.datavec.dataframe.table.Relation;
import org.datavec.dataframe.table.Rows;
import org.datavec.dataframe.table.ViewGroup;
import org.datavec.dataframe.util.BitmapBackedSelection;
import org.datavec.dataframe.util.IntComparatorChain;
import org.datavec.dataframe.util.ReversingIntComparator;
import org.datavec.dataframe.util.Selection;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntArrays;
import it.unimi.dsi.fastutil.ints.IntComparator;
import it.unimi.dsi.fastutil.ints.IntIterable;
import it.unimi.dsi.fastutil.ints.IntIterator;
import org.apache.commons.lang3.RandomUtils;

import java.io.IOException;
import java.io.InputStream;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * A table of data, consisting of some number of columns, each of which has the same number of rows.
 * All the data in a column has the same type: integer, float, category, etc., but a table may contain an arbitrary
 * number of columns of any type.
 * <p>
 * Tables are the main data-type and primary focus of Tablesaw.
 */
public class Table implements Relation, IntIterable {

  /**
   * The name of the table
   */
  private String name;

  /**
   * The columns that hold the data in this table
   */
  private final List<Column> columnList = new ArrayList<>();

  /**
   * Returns a new table initialized with the given name
   */
  private Table(String name) {
    this.name = name;
  }

  /**
   * Returns a new table initialized with data from the given TableMetadata object
   * <p>
   * The metadata is used by the storage module to save tables and read their data from disk
   */
  private Table(TableMetadata metadata) {
    this.name = metadata.getName();
  }

  /**
   * Returns a new Table initialized with the given names and columns
   *
   * @param name    The name of the table
   * @param columns One or more columns, all of which must have either the same length or size 0
   */
  protected Table(String name, Column... columns) {
    this(name);
    for (Column column : columns) {
      this.addColumn(column);
    }
  }

  /**
   * Returns a new, empty table (without rows or columns) with the given name
   */
  public static Table create(String tableName) {
    return new Table(tableName);
  }


  /**
   * Returns a new, empty table constructed according to the given metadata
   */
  public static Table create(TableMetadata metadata) {
    return new Table(metadata);
  }

  /**
   * Returns a new table with the given columns and given name
   *
   * @param columns One or more columns, all of the same @code{column.size()}
   */
  public static Table create(String tableName, Column... columns) {
    return new Table(tableName, columns);
  }

  /**
   * Adds the given column to this table
   */
  @Override
  public void addColumn(Column... cols) {
    for (Column c : cols) {
      validateColumn(c);
      columnList.add(c);
    }
  }

  /**
   * Throws a runtime exception if a column with the given name is already in the table
   */
  private void validateColumn(Column newColumn) {
    Preconditions.checkNotNull(newColumn, "Attempted to add a null to the columns in table " + name);
    List<String> stringList = new ArrayList<>();
    for (String name : columnNames()) {
      stringList.add(name.toLowerCase());
    }
    if (stringList.contains(newColumn.name().toLowerCase())) {
      String message = String.format("Cannot add column with duplicate name %s to table %s", newColumn, name);
      throw new RuntimeException(message);
    }
  }

  /**
   * Adds the given column to this table at the given position in the column list
   *
   * @param index  Zero-based index into the column list
   * @param column Column to be added
   */
  public void addColumn(int index, Column column) {
    validateColumn(column);
    columnList.add(index, column);
  }

  /**
   * Sets the name of the table
   */
  @Override
  public void setName(String name) {
    this.name = name;
  }

  /**
   * Returns the column at the given index in the column list
   *
   * @param columnIndex an integer at least 0 and less than number of columns in the table
   */
  @Override
  public Column column(int columnIndex) {
    return columnList.get(columnIndex);
  }

  /**
   * Returns the number of columns in the table
   */
  @Override
  public int columnCount() {
    return columnList.size();
  }

  /**
   * Returns the number of rows in the table
   */
  @Override
  public int rowCount() {
    int result = 0;
    if (!columnList.isEmpty()) {
      // all the columns have the same number of elements, so we can check any of them
      result = columnList.get(0).size();
    }
    return result;
  }

  /**
   * Returns the list of columns
   */
  @Override
  public List<Column> columns() {
    return columnList;
  }


  /**
   * Returns only the columns whose names are given in the input array
   */
  public List<Column> columns(String... columnNames) {
    List<Column> columns = new ArrayList<>();
    for (String columnName : columnNames) {
      columns.add(column(columnName));
    }
    return columns;
  }

  /**
   * Returns the index of the column with the given name
   *
   * @throws IllegalArgumentException if the input string is not the name of any column in the table
   */
  public int columnIndex(String columnName) {
    int columnIndex = -1;
    for (int i = 0; i < columnList.size(); i++) {
      if (columnList.get(i).name().equalsIgnoreCase(columnName)) {
        columnIndex = i;
        break;
      }
    }
    if (columnIndex == -1) {
      throw new IllegalArgumentException(String.format("Column %s is not present in table %s", columnName, name));
    }
    return columnIndex;
  }

  /**
   * Returns the index of the given column (its position in the list of columns)
   * <p>
   *
   * @throws IllegalArgumentException if the column is not present in this table
   */
  public int columnIndex(Column column) {
    int columnIndex = -1;
    for (int i = 0; i < columnList.size(); i++) {
      if (columnList.get(i).equals(column)) {
        columnIndex = i;
        break;
      }
    }
    if (columnIndex == -1) {
      throw new IllegalArgumentException(
          String.format("Column %s is not present in table %s", column.name(), name));
    }
    return columnIndex;
  }

  /**
   * Returns the name of the table
   */
  @Override
  public String name() {
    return name;
  }

  /**
   * Returns a List of the names of all the columns in this table
   */
  public List<String> columnNames() {
    List<String> names = new ArrayList<>(columnList.size());
    names.addAll(columnList.stream().map(Column::name).collect(Collectors.toList()));
    return names;
  }

  /**
   * Returns a string representation of the value at the given row and column indexes
   *
   * @param c the column index, 0 based
   * @param r the row index, 0 based
   */
  @Override
  public String get(int c, int r) {
    Column column = column(c);
    return column.getString(r);
  }

  /**
   * Returns a table with the same columns as this table, but no data
   */
  public Table emptyCopy() {
    Table copy = new Table(name);
    for (Column column : columnList) {
      copy.addColumn(column.emptyCopy());
    }
    return copy;
  }

  /**
   * Returns a table with the same columns as this table, but no data, initialized to the given row size
   */
  public Table emptyCopy(int rowSize) {
    Table copy = new Table(name);
    for (Column column : columnList) {
      copy.addColumn(column.emptyCopy(rowSize));
    }
    return copy;
  }

  /**
   * Splits the table into two, randomly assigning records to each according to the proportion given in
   * trainingProportion
   *
   * @param table1Proportion The proportion to go in the first table
   * @return An array two tables, with the first table having the proportion specified in the method parameter,
   * and the second table having the balance of the rows
   */
  public Table[] sampleSplit(double table1Proportion) {
    Table[] tables = new Table[2];
    int table1Count = (int) Math.round(rowCount() * table1Proportion);

    Selection table2Selection = new BitmapBackedSelection();
    for (int i = 0; i < rowCount(); i++) {
      table2Selection.add(i);
    }
    Selection table1Selection = new BitmapBackedSelection();

    int[] table1Records = generateUniformBitmap(table1Count, rowCount());
    for (int i = 0; i < table1Records.length; i++) {
      table1Selection.add(table1Records[i]);
    }
    table2Selection.andNot(table1Selection);
    tables[0] = selectWhere(table1Selection);
    tables[1] = selectWhere(table2Selection);
    return tables;
  }

  /**
   * Returns a table consisting of randomly selected records from this table. The sample size is based on the
   * given proportion
   *
   * @param proportion The proportion to go in the sample
   */
  public Table sample(double proportion) {

    int tableCount = (int) Math.round(rowCount() * proportion);

    Selection table1Selection = new BitmapBackedSelection();
    int[] selectedRecords = generateUniformBitmap(tableCount, rowCount());
    for (int selectedRecord : selectedRecords) {
      table1Selection.add(selectedRecord);
    }
    return selectWhere(table1Selection);
  }

  /**
   * Clears all the data from this table
   */
  @Override
  public void clear() {
    columnList.forEach(Column::clear);
  }

  /**
   * Returns a new table containing the first {@code nrows} of data in this table
   */
  public Table first(int nRows) {
    nRows = Math.min(nRows, rowCount());
    Table newTable = emptyCopy(nRows);
    Rows.head(nRows, this, newTable);
    return newTable;
  }

  /**
   * Returns a new table containing the last {@code nrows} of data in this table
   */
  public Table last(int nRows) {
    nRows = Math.min(nRows, rowCount());
    Table newTable = emptyCopy(nRows);
    Rows.tail(nRows, this, newTable);
    return newTable;
  }

  /**
   * Returns a sort Key that can be used for simple or chained comparator sorting
   * <p>
   * You can extend the sort key by using .next() to fill more columns to the sort order
   */
  private static Sort first(String columnName, Sort.Order order) {
    return Sort.on(columnName, order);
  }

  /**
   * Returns a copy of this table sorted on the given column names, applied in order,
   * <p>
   * if column name starts with - then sort that column descending otherwise sort ascending
   */
  public Table sortOn(String... columnNames) {

    Sort key = null;
    Sort.Order order;
    List<String> names = new ArrayList<>();
    for (String name : columnNames()) {
      names.add(name.toUpperCase());
    }

    for (String columnName : columnNames) {
      if (names.contains(columnName.toUpperCase())) {
        // the column name has not been annotated with a prefix.
        order = Sort.Order.ASCEND;
      } else {

        // get the prefix which could be - or +
        String prefix = columnName.substring(0, 1);

        // remove - prefix so provided name matches actual column name
        columnName = columnName.substring(1, columnName.length());

        switch (prefix) {
          case "+":
            order = Sort.Order.ASCEND;
            break;
          case "-":
            order = Sort.Order.DESCEND;
            break;
          default:
            throw new IllegalStateException("Column prefix: " + prefix + " is unknown.");
        }
      }

      if (key == null) { // key will be null the first time through
        key = first(columnName, order);
      } else {
        key.next(columnName, order);
      }
    }
    return sortOn(key);
  }

  /**
   * Returns a copy of this table sorted in the order of the given column names, in ascending order
   */
  public Table sortAscendingOn(String... columnNames) {
    return this.sortOn(columnNames);
  }

  /**
   * Returns a copy of this table sorted on the given column names, applied in order, descending
   */
  public Table sortDescendingOn(String... columnNames) {
    Sort key = getSort(columnNames);
    return sortOn(key);
  }

  /**
   * Returns an object that can be used to sort this table in the order specified for by the given column names
   */
  @VisibleForTesting
  public static Sort getSort(String... columnNames) {
    Sort key = null;
    for (String s : columnNames) {
      if (key == null) {
        key = first(s, Sort.Order.DESCEND);
      } else {
        key.next(s, Sort.Order.DESCEND);
      }
    }
    return key;
  }

  /**
   */
  public Table sortOn(Sort key) {
    Preconditions.checkArgument(!key.isEmpty());
    if (key.size() == 1) {
      IntComparator comparator = getComparator(key);
      return sortOn(comparator);
    }
    IntComparatorChain chain = getChain(key);
    return sortOn(chain);
  }

  /**
   * Returns a comparator that can be used to sort the records in this table according to the given sort key
   */
  public IntComparator getComparator(Sort key) {
    Iterator<Map.Entry<String, Sort.Order>> entries = key.iterator();
    Map.Entry<String, Sort.Order> sort = entries.next();
    IntComparator comparator;
    if (sort.getValue() == Sort.Order.ASCEND) {
      comparator = rowComparator(sort.getKey(), false);
    } else {
      comparator = rowComparator(sort.getKey(), true);
    }
    return comparator;
  }

  /**
   * Returns a comparator chain for sorting according to the given key
   */
  private IntComparatorChain getChain(Sort key) {
    Iterator<Map.Entry<String, Sort.Order>> entries = key.iterator();
    Map.Entry<String, Sort.Order> sort = entries.next();

    IntComparator comparator;
    if (sort.getValue() == Sort.Order.ASCEND) {
      comparator = rowComparator(sort.getKey(), false);
    } else {
      comparator = rowComparator(sort.getKey(), true);
    }

    IntComparatorChain chain = new IntComparatorChain(comparator);
    while (entries.hasNext()) {
      sort = entries.next();
      if (sort.getValue() == Sort.Order.ASCEND) {
        chain.addComparator(rowComparator(sort.getKey(), false));
      } else {
        chain.addComparator(rowComparator(sort.getKey(), true));
      }
    }
    return chain;
  }

  /**
   * Returns a copy of this table sorted using the given comparator
   */
  public Table sortOn(IntComparator rowComparator) {
    Table newTable = emptyCopy(rowCount());

    int[] newRows = rows();
    IntArrays.parallelQuickSort(newRows, rowComparator);

    Rows.copyRowsToTable(IntArrayList.wrap(newRows), this, newTable);
    return newTable;
  }

  /**
   * Returns an array of ints of the same number of rows as the table
   */
  @VisibleForTesting
  public int[] rows() {
    int[] rowIndexes = new int[rowCount()];
    for (int i = 0; i < rowCount(); i++) {
      rowIndexes[i] = i;
    }
    return rowIndexes;
  }

  /**
   * Returns a comparator for the column matching the specified name
   *
   * @param columnName The name of the column to sort
   * @param reverse    {@code true} if the column should be sorted in reverse
   */
  private IntComparator rowComparator(String columnName, boolean reverse) {

    Column column = this.column(columnName);
    IntComparator rowComparator = column.rowComparator();

    if (reverse) {
      return ReversingIntComparator.reverse(rowComparator);
    } else {
      return rowComparator;
    }
  }

  public Table selectWhere(Selection selection) {
    Table newTable = this.emptyCopy(selection.size());
    Rows.copyRowsToTable(selection, this, newTable);
    return newTable;
  }

  public BooleanColumn selectIntoColumn(String newColumnName, Selection selection) {
    return BooleanColumn.create(newColumnName, selection, rowCount());
  }

  public Table selectWhere(Filter filter) {
    Selection map = filter.apply(this);
    Table newTable = this.emptyCopy(map.size());
    Rows.copyRowsToTable(map, this, newTable);
    return newTable;
  }

  public BooleanColumn selectIntoColumn(String newColumnName, Filter filter) {
    return BooleanColumn.create(newColumnName, filter.apply(this), rowCount());
  }

  public ViewGroup splitOn(Column... columns) {
    return new ViewGroup(this, columns);
  }

  public String printHtml() {
    return HtmlTableWriter.write(this, "");
  }

  public Table structure() {
    Table t = new Table("Structure of " + name());
    IntColumn index = new IntColumn("Index", columnCount());
    CategoryColumn columnName = new CategoryColumn("Column Name", columnCount());
    CategoryColumn columnType = new CategoryColumn("Column Type", columnCount());
    t.addColumn(index);
    t.addColumn(columnName);
    t.addColumn(columnType);
    columnName.addAll(columnNames());
    for (int i = 0; i < columnCount(); i++) {
      Column column = columnList.get(i);
      index.add(i);
      columnType.add(column.type().name());
    }
    return t;
  }

  /**
   * Returns the unique records in this table
   * Note: Uses a lot of memory for a sort
   */
  public Table uniqueRecords() {

    IntArrayList uniqueRows = new IntArrayList();
    Table sorted = this.sortOn(columnNames().toArray(new String[columns().size()]));
    Table temp = emptyCopy();

    for (int row = 0; row < rowCount(); row++) {
      if (temp.isEmpty() || !Rows.compareRows(row, sorted, temp)) {
        uniqueRows.add(row);
        Rows.appendRowToTable(row, sorted, temp);
      }
    }
    return temp;
  }

  public Projection select(String... columnName) {
    return new Projection(this, columnName);
  }

  /**
   * Removes the given columns
   */
  @Override
  public void removeColumns(Column... columns) {
    for (Column c : columns) {
      columnList.remove(c);
    }
  }

  /**
   * Removes the given columns
   */
  public void retainColumns(Column... columns) {
    List<Column> retained = Arrays.asList(columns);
    columnList.retainAll(retained);
  }

  public void retainColumns(String... columnNames) {
    columnList.retainAll(columns(columnNames));
  }

  public Sum sum(String numericColumnName) {
    return new Sum(this, numericColumnName);
  }

  public Mean mean(String numericColumnName) {
    return new Mean(this, numericColumnName);
  }

  public Median median(String numericColumnName) {
    return new Median(this, numericColumnName);
  }

  public Variance variance(String numericColumnName) {
    return new Variance(this, numericColumnName);
  }

  public StandardDeviation stdDev(String numericColumnName) {
    return new StandardDeviation(this, numericColumnName);
  }

  public Count count(String numericColumnName) {
    return new Count(this, numericColumnName);
  }

  public Maximum max(String numericColumnName) {
    return new Maximum(this, numericColumnName);
  }

  public Minimum minimum(String numericColumnName) {
    return new Minimum(this, numericColumnName);
  }

  public void append(Table tableToAppend) {
    for (Column column : columnList) {
      Column columnToAppend = tableToAppend.column(column.name());
      column.append(columnToAppend);
    }
  }

  /**
   * Exports this table as a CSV file with the name (and path) of the given file
   *
   * @param fileNameWithPath The name of the file to save to. By default, it writes to the working directory,
   *                         but you can specify a different folder by providing the path (e.g. mydata/myfile.csv)
   */
  public void exportToCsv(String fileNameWithPath) {
    try {
      CsvWriter.write(fileNameWithPath, this);
    } catch (IOException e) {
      System.err.println("Unable to export table as CSV file");
      e.printStackTrace();
    }
  }

  public String save(String folder) {
    String storageFolder = "";
    try {
      storageFolder = StorageManager.saveTable(folder, this);
    } catch (IOException e) {
      System.err.println("Unable to save table in Tablesaw format");
      e.printStackTrace();
    }
    return storageFolder;
  }

  public static Table readTable(String tableNameAndPath) {
    Table t;
    try {
      t = StorageManager.readTable(tableNameAndPath);
    } catch (IOException e) {
      System.err.println("Unable to load table from Tablesaw table format");
      e.printStackTrace();
      return null;
    }
    return t;
  }

  /**
   * Returns the result of applying the given function to the specified column
   *
   * @param numericColumnName The name of a numeric (integer, float, etc.) column in this table
   * @param function          A numeric reduce function
   * @return the function result
   * @throws IllegalArgumentException if numericColumnName doesn't name a numeric column in this table
   */
  public double reduce(String numericColumnName, NumericReduceFunction function) {
    Column column = column(numericColumnName);
    return function.reduce(column.toDoubleArray());
  }

  public SummaryFunction summarize(String numericColumnName, NumericReduceFunction function) {
    return new SummaryFunction(this, numericColumnName) {
      @Override
      public NumericReduceFunction function() {
        return function;
      }
    };
  }

  public Table countBy(CategoryColumn column) {
    return column.countByCategory();
  }

  /**
   * Returns a new table constructed from a character delimited (aka CSV) text file
   * <p>
   * It is assumed that the file is truly comma-separated, and that the file has a one-line header,
   * which is used to populate the column names
   *
   * @param csvFileName The name of the file to import
   * @throws IOException
   */
  public static Table createFromCsv(String csvFileName) throws IOException {
    return CsvReader.read(csvFileName, true, ',');
  }

  /**
   * Returns a new table constructed from a character delimited (aka CSV) text file
   * <p>
   * It is assumed that the file is truly comma-separated, and that the file has a one-line header,
   * which is used to populate the column names
   *
   * @param csvFileName The name of the file to import
   * @param header      True if the file has a single header row. False if it has no header row.
   *                    Multi-line headers are not supported
   * @throws IOException
   */
  public static Table createFromCsv(String csvFileName, boolean header) throws IOException {
    return CsvReader.read(csvFileName, header, ',');
  }

  /**
   * Returns a new table constructed from a character delimited (aka CSV) text file
   * <p>
   * It is assumed that the file is truly comma-separated, and that the file has a one-line header,
   * which is used to populate the column names
   *
   * @param csvFileName The name of the file to import
   * @param header      True if the file has a single header row. False if it has no header row.
   *                    Multi-line headers are not supported
   * @param delimiter   a char that divides the columns in the source file, often a comma or tab
   * @throws IOException
   */
  public static Table createFromCsv(String csvFileName, boolean header, char delimiter) throws IOException {
    return CsvReader.read(csvFileName, header, delimiter);
  }

  /**
   * Returns a new table constructed from a character delimited (aka CSV) text file
   * <p>
   * It is assumed that the file is truly comma-separated, and that the file has a one-line header,
   * which is used to populate the column names
   *
   * @param types       The column types
   * @param csvFileName The name of the file to import
   * @throws IOException
   */
  public static Table createFromCsv(ColumnType[] types, String csvFileName) throws IOException {
    return CsvReader.read(types, true, ',', csvFileName);
  }

  /**
   * Returns a new table constructed from a character delimited (aka CSV) text file
   * <p>
   * It is assumed that the file is truly comma-separated
   *
   * @param types       The column types
   * @param header      True if the file has a single header row. False if it has no header row.
   *                    Multi-line headers are not supported
   * @param csvFileName the name of the file to import
   * @throws IOException
   */
  public static Table createFromCsv(ColumnType[] types, String csvFileName, boolean header) throws IOException {
    return CsvReader.read(types, header, ',', csvFileName);
  }

  /**
   * Returns a new table constructed from a character delimited (aka CSV) text file
   *
   * @param types       The column types
   * @param header      true if the file has a single header row. False if it has no header row.
   *                    Multi-line headers are not supported
   * @param delimiter   a char that divides the columns in the source file, often a comma or tab
   * @param csvFileName the name of the file to import
   * @throws IOException
   */
  public static Table createFromCsv(ColumnType[] types, String csvFileName, boolean header, char delimiter)
      throws IOException {
    return CsvReader.read(types, header, delimiter, csvFileName);
  }

  /**
   * Returns a new table constructed from a character delimited (aka CSV) text file
   *
   * @param types     The column types
   * @param header    true if the file has a single header row. False if it has no header row.
   *                  Multi-line headers are not supported
   * @param delimiter a char that divides the columns in the source file, often a comma or tab
   * @param stream    an InputStream from a file, URL, etc.
   * @param tableName the name of the resulting table
   * @throws IOException
   */
  public static Table createFromStream(ColumnType[] types, boolean header, char delimiter, InputStream stream,
                                       String tableName) throws IOException {
    return CsvReader.read(tableName, types, header, delimiter, stream);
  }

  /**
   * Returns a new Table with the given name, and containing the data in the given result set
   */
  public static Table create(ResultSet resultSet, String tableName) throws SQLException {
    return SqlResultSetReader.read(resultSet, tableName);
  }

/*
  */
/**
   * Joins together this table and another table on the given column names. All the records of this table are included
   * @return   A new table derived from combining this table with {@code other} table
   *//*

  public Table innerJoin(Table other, String columnName, String otherColumnName) {
    // create a new table like this one
    Table table = new Table(this.name());
    // add the columns from this table
    for (Column column : columns()) {
      table.addColumn(column);
    }
    // add the columns from the other table, but leave the data out for now
    for (Column column : other.columns()) {
      if (!column.name().equals(otherColumnName)) {
        table.addColumn(column.emptyCopy());
      }
    }

    // iterate over the rows in the new table, fetching rows from the other table that match on
    Column joinColumn = column(columnName);
    Column otherJoinColumn = other.column(otherColumnName);
    for (int row : table) {
       joinColumn.getString(row))

     // Row otherRow = other.getFirst(otherColumnName, comparable);
      if (otherRow != null) {
        // fill in the values of other tables columns for that row.
        for (Column c : other.columns()) {
          if (!c.name().equals(otherColumnName)) {
            row.set(c.name(), otherRow.get(c.name()));
          }
        }
      }
    }
    return table;
  }
*/

  @Override
  public String toString() {
    return "Table " + name + ": Size = " + rowCount() + " x " + columnCount();
  }

  @Override
  public IntIterator iterator() {

    return new IntIterator() {

      private int i = 0;

      @Override
      public int nextInt() {
        return i++;
      }

      @Override
      public int skip(int k) {
        return i + k;
      }

      @Override
      public boolean hasNext() {
        return i < rowCount();
      }

      @Override
      public Integer next() {
        return i++;
      }
    };
  }

  /**
   * Returns an randomly generated array of ints of size N where Max is the largest possible value
   */
  static int[] generateUniformBitmap(int N, int Max) {

    if (N > Max) throw new RuntimeException("not possible");

    int[] ans = new int[N];
    if (N == Max) {
      for (int k = 0; k < N; ++k)
        ans[k] = k;
      return ans;
    }

    BitSet bs = new BitSet(Max);
    int cardinality = 0;
    while (cardinality < N) {
      int v = RandomUtils.nextInt(0, Max);
      if (!bs.get(v)) {
        bs.set(v);
        cardinality++;
      }
    }
    int pos = 0;
    for (int i = bs.nextSetBit(0); i >= 0; i = bs.nextSetBit(i + 1)) {
      ans[pos++] = i;
    }
    return ans;
  }
}

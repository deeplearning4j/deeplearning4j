package org.datavec.dataframe.table;

import org.datavec.dataframe.api.BooleanColumn;
import org.datavec.dataframe.api.CategoryColumn;
import org.datavec.dataframe.api.DateColumn;
import org.datavec.dataframe.api.DateTimeColumn;
import org.datavec.dataframe.api.FloatColumn;
import org.datavec.dataframe.api.IntColumn;
import org.datavec.dataframe.api.LongColumn;
import org.datavec.dataframe.api.NumericColumn;
import org.datavec.dataframe.api.ShortColumn;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.api.TimeColumn;
import org.datavec.dataframe.columns.Column;
import org.datavec.dataframe.reducing.NumericReduceFunction;
import org.datavec.dataframe.util.BitmapBackedSelection;
import org.datavec.dataframe.util.Selection;
import it.unimi.dsi.fastutil.ints.IntIterable;
import it.unimi.dsi.fastutil.ints.IntIterator;
import org.apache.commons.lang3.StringUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * A TemporaryView is a facade around a Relation that acts as a filtering.
 * Requests for data are forwarded to the underlying table.
 * <p>
 * The view is only good until the structure of the underlying table changes, after which it is marked 'stale'.
 * At that point, it's operations will return an error.
 * <p>
 * View is something of a misnomer, as it is not like a database view, which is merely a query masquerading as a table,
 * nor is it like a materialized database view, which is like a real table.
 */
public class TemporaryView implements Relation, IntIterable {

  private String name;
  private Table table;
  private final Selection rowMap;

  /**
   * Returns a new View constructed from the given table, containing only the rows represented by the bitmap
   */
  public TemporaryView(Table table, Selection rowSelection) {
    this.name = table.name();
    this.rowMap = rowSelection;
    this.table = table;
  }

  @Override
  public Column column(int columnIndex) {
    return table.column(columnIndex);
  }

  @Override
  public int columnCount() {
    return table.columnCount();
  }

  @Override
  public int rowCount() {
    return rowMap.size();
  }

  @Override
  public List<Column> columns() {
    List<Column> columns = new ArrayList<>();
    for (int i = 0; i < columnCount(); i++) {
      columns.add(column(i));
    }
    return columns;
  }

  @Override
  public int columnIndex(Column column) {
    return table.columnIndex(column);
  }

  @Override
  public String get(int c, int r) {
    return table.get(c, r);
  }

  @Override
  public void addColumn(Column... column) {
    throw new UnsupportedOperationException("TemporaryView does not support the addColumn operation");
  }

  @Override
  public String name() {
    return name;
  }

  /**
   * Clears all rows from this View, leaving the structure in place
   */
  @Override
  public void clear() {
    rowMap.clear();
  }

  @Override
  public List<String> columnNames() {
    return table.columnNames();
  }

  @Override
  public void removeColumns(Column... columns) {
    throw new UnsupportedOperationException("TemporaryView does not support the removeColumns operation");
  }

  @Override
  public Table first(int nRows) {
    Selection newMap = new BitmapBackedSelection();
    int count = 0;
    IntIterator it = intIterator();
    while (it.hasNext() && count < nRows) {
      int row = it.next();
      newMap.add(row);
      count++;
    }
    return table.selectWhere(newMap);
  }

  @Override
  public void setName(String name) {
    this.name = name;
  }


  @Override
  public String print() {
    StringBuilder buf = new StringBuilder();

    int[] colWidths = colWidths();
    buf.append(name()).append('\n');
    List<String> names = this.columnNames();

    for (int colNum = 0; colNum < columnCount(); colNum++) {
      buf.append(
          StringUtils.rightPad(
              StringUtils.defaultString(String.valueOf(names.get(colNum))), colWidths[colNum]));
      buf.append(' ');
    }
    buf.append('\n');
    IntIterator iterator = intIterator();
    while (iterator.hasNext()) {
      int r = iterator.next();
      for (int i = 0; i < columnCount(); i++) {
        String cell = StringUtils.rightPad(get(i, r), colWidths[i]);
        buf.append(cell);
        buf.append(' ');
      }
      buf.append('\n');
    }
    return buf.toString();
  }

  /**
   * Returns an array of column widths for printing tables
   */
  @Override
  public int[] colWidths() {

    int cols = columnCount();
    int[] widths = new int[cols];
    List<String> columnNames = columnNames();

    for (int i = 0; i < columnCount(); i++) {
      widths[i] = columnNames.get(i).length();
    }

    for (int rowNum = 0; rowNum < rowCount(); rowNum++) {
      for (int colNum = 0; colNum < cols; colNum++) {
        String value = get(colNum, rowNum);
        widths[colNum]
            = Math.max(widths[colNum], StringUtils.length(value));
      }
    }
    return widths;
  }

  public Table asTable() {
    Table table = Table.create(this.name());
    for (Column column : columns()) {
      table.addColumn(column.subset(rowMap));
    }
    return table;
  }

  IntIterator intIterator() {
    return rowMap.iterator();
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
    return function.reduce(column.subset(rowMap).toDoubleArray());
  }

  public String toString() {
    return "View " + name() + ": Size = " + rowCount() + " x " + columns().size();
  }

  public BooleanColumn booleanColumn(int columnIndex) {
    return (BooleanColumn) column(columnIndex).subset(rowMap);
  }

  public BooleanColumn booleanColumn(String columnName) {
    return (BooleanColumn) column(columnName).subset(rowMap);
  }

  public FloatColumn floatColumn(int columnIndex) {
    return (FloatColumn) column(columnIndex).subset(rowMap);
  }

  public FloatColumn floatColumn(String columnName) {
    return (FloatColumn) column(columnName).subset(rowMap);
  }

  public IntColumn intColumn(String columnName) {
    return (IntColumn) column(columnName).subset(rowMap);
  }

  public IntColumn intColumn(int columnIndex) {
    return (IntColumn) column(columnIndex).subset(rowMap);
  }

  public ShortColumn shortColumn(String columnName) {
    return (ShortColumn) column(columnName).subset(rowMap);
  }

  public ShortColumn shortColumn(int columnIndex) {
    return (ShortColumn) column(columnIndex).subset(rowMap);
  }

  public LongColumn longColumn(String columnName) {
    return (LongColumn) column(columnName).subset(rowMap);
  }

  public LongColumn longColumn(int columnIndex) {
    return (LongColumn) column(columnIndex).subset(rowMap);
  }

  public DateColumn dateColumn(int columnIndex) {
    return (DateColumn) column(columnIndex).subset(rowMap);
  }

  public DateColumn dateColumn(String columnName) {
    return (DateColumn) column(columnName).subset(rowMap);
  }

  public TimeColumn timeColumn(String columnName) {
    return (TimeColumn) column(columnName).subset(rowMap);
  }

  public TimeColumn timeColumn(int columnIndex) {
    return (TimeColumn) column(columnIndex).subset(rowMap);
  }

  public DateTimeColumn dateTimeColumn(int columnIndex) {
    return (DateTimeColumn) column(columnIndex).subset(rowMap);
  }

  public DateTimeColumn dateTimeColumn(String columnName) {
    return (DateTimeColumn) column(columnName).subset(rowMap);
  }

  public CategoryColumn categoryColumn(String columnName) {
    return (CategoryColumn) column(columnName).subset(rowMap);
  }

  public CategoryColumn categoryColumn(int columnIndex) {
    return (CategoryColumn) column(columnIndex).subset(rowMap);
  }

  public NumericColumn numericColumn(int columnIndex) {
    return (NumericColumn) column(columnIndex).subset(rowMap);
  }

  public NumericColumn numericColumn(String columnName) {
    return (NumericColumn) column(columnName).subset(rowMap);
  }

  @Override
  public it.unimi.dsi.fastutil.ints.IntIterator iterator() {

    return new it.unimi.dsi.fastutil.ints.IntIterator() {

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
}
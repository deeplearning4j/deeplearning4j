package org.datavec.dataframe.table;

import org.datavec.dataframe.reducing.NumericReduceFunction;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.api.CategoryColumn;
import org.datavec.dataframe.columns.Column;
import org.datavec.dataframe.api.FloatColumn;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * A group of tables formed by performing splitting operations on an original table
 */
public class TableGroup implements Iterable<SubTable> {

  private static final String SPLIT_STRING = "~~~";
  private static final Splitter SPLITTER = Splitter.on(SPLIT_STRING);
  private final Table original;

  private final List<SubTable> subTables;

  // the name(s) of the column(s) we're splitting the table on
  private String[] splitColumnNames;

  public TableGroup(Table original, String... splitColumnNames) {
    this.original = original.sortOn(splitColumnNames);
    this.subTables = splitOn(splitColumnNames);
    Preconditions.checkState(!subTables.isEmpty());
    this.splitColumnNames = splitColumnNames;
  }

  public TableGroup(Table original, Column... columns) {
    splitColumnNames = new String[columns.length];
    for (int i = 0; i < columns.length; i++) {
      splitColumnNames[i] = columns[i].name();
    }
    this.original = original.sortOn(splitColumnNames);
    this.subTables = splitOn(splitColumnNames);
    Preconditions.checkState(!subTables.isEmpty());
  }

  /**
   * Splits the original table into sub-tables, grouping on the columns whose names are given in splitColumnNames
   */
  private List<SubTable> splitOn(String... columnNames) {

    int columnCount = columnNames.length;
    List<Column> columns = original.columns(columnNames);
    List<SubTable> tables = new ArrayList<>();

    int[] columnIndices = new int[columnCount];
    for (int i = 0; i < columnCount; i++) {
      columnIndices[i] = original.columnIndex(columnNames[i]);
    }

    Table empty = original.emptyCopy();

    SubTable newView = new SubTable(empty);
    String lastKey = "";
    newView.setName(lastKey);

    for (int row = 0; row < original.rowCount(); row++) {

      String newKey = "";
      List<String> values = new ArrayList<>();

      for (int col = 0; col < columnCount; col++) {
        if (col > 0)
          newKey = newKey + SPLIT_STRING;

        String groupKey = original.get(columnIndices[col], row);
        newKey = newKey + groupKey;
        values.add(groupKey);
      }

      if (!newKey.equals(lastKey)) {
        if (!newView.isEmpty()) {
          tables.add(newView);
        }

        newView = new SubTable(empty);
        newView.setName(newKey);
        newView.setValues(values);
        lastKey = newKey;
      }
      newView.addRow(row, original);
    }

    if (!tables.contains(newView) && !newView.isEmpty()) {
      if (columnCount == 1) {
        tables.add(newView);
      } else {
        tables.add(newView);
      }
    }
    return tables;
  }

  private SubTable splitGroupingColumn(SubTable subTable, List<Column> columns) {

    List<Column> newColumns = new ArrayList<>();

    for (Column column : columns) {
      Column newColumn = column.emptyCopy();
      newColumns.add(newColumn);
    }
    // iterate through the rows in the table and split each of the grouping columns into multiple columns
    for (int row = 0; row < subTable.rowCount(); row++) {
      List<String> strings = SPLITTER.splitToList(subTable.name());
      for (int col = 0; col < newColumns.size(); col++) {
        newColumns.get(col).addCell(strings.get(col));
      }
    }
    for (Column c : newColumns) {
      subTable.addColumn(c);
    }
    return subTable;
  }

  public List<SubTable> getSubTables() {
    return subTables;
  }

  public int size() {
    return subTables.size();
  }

  public Table reduce(String numericColumnName, NumericReduceFunction function) {
    Preconditions.checkArgument(!subTables.isEmpty());
    Table t = Table.create(original.name() + " summary");
    CategoryColumn groupColumn = new CategoryColumn("Group", subTables.size());
    FloatColumn resultColumn = new FloatColumn(function.functionName(), subTables.size());
    t.addColumn(groupColumn);
    t.addColumn(resultColumn);

    for (SubTable subTable : subTables) {
      double result = subTable.reduce(numericColumnName, function);
      groupColumn.add(subTable.name().replace(SPLIT_STRING, " * "));
      resultColumn.add((float) result);
    }
    return t;
  }

  /**
   * Returns an iterator over elements of type {@code T}.
   *
   * @return an Iterator.
   */
  @Override
  public Iterator<SubTable> iterator() {
    return subTables.iterator();
  }
}

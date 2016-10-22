package org.datavec.dataframe.table;

import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.columns.Column;
import org.datavec.dataframe.filtering.Filter;
import org.datavec.dataframe.util.Selection;

/**
 * A table projection, i.e. the subset of columns in a table that should be returned in a query
 */
public class Projection {

  private final Table table;
  private final Column[] columns;

  public Projection(Table table, String[] columnNames) {
    this.table = table;
    columns = new Column[columnNames.length];
    for (int i = 0; i < columnNames.length; i++) {
      String name = columnNames[i];
      columns[i] = table.column(name);
    }
  }

  public Table where(Filter filter) {
    Table projectedTable = Table.create(table.name(), columns);
    Table newTable = projectedTable.emptyCopy();
    Selection map = filter.apply(table);
    Rows.copyRowsToTable(map, projectedTable, newTable);
    return newTable;
  }
}

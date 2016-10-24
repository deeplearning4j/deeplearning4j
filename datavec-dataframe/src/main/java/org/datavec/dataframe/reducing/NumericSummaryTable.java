package org.datavec.dataframe.reducing;

import org.datavec.dataframe.api.Table;

/**
 * NumericSummaryTable is a standard table, but one with a specific format:
 * It has two columns, the first a category column and the second a numeric column,
 * so that it is appropriate for managing data that summarizes numeric variables by groups
 */
public class NumericSummaryTable extends Table {

  /**
   * Returns a new, empty table (without rows or columns) with the given name
   */
  public static NumericSummaryTable create(String tableName) {
    return new NumericSummaryTable(tableName);
  }

  /**
   * Returns a new Table initialized with the given names and columns
   *
   * @param name    The name of the table
   */
   private NumericSummaryTable(String name) {
    super(name);
  }
}

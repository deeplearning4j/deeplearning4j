package org.datavec.dataframe.filtering.ints;

import org.datavec.dataframe.api.IntColumn;
import org.datavec.dataframe.api.LongColumn;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.columns.ColumnReference;
import org.datavec.dataframe.filtering.ColumnFilter;
import org.datavec.dataframe.util.Selection;

/**
 */
public class IntLessThan extends ColumnFilter {

  private int value;

  public IntLessThan(ColumnReference reference, int value) {
    super(reference);
    this.value = value;
  }

  public Selection apply(Table relation) {
    IntColumn longColumn = (IntColumn) relation.column(getColumnReference().getColumnName());
    return longColumn.isLessThan(value);
  }
}

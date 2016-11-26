package org.datavec.dataframe.filtering.ints;

import org.datavec.dataframe.api.IntColumn;
import org.datavec.dataframe.api.LongColumn;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.columns.ColumnReference;
import org.datavec.dataframe.filtering.ColumnFilter;
import org.datavec.dataframe.util.Selection;

/**
 */
public class IntLessThanOrEqualTo extends ColumnFilter {

  private int value;

  public IntLessThanOrEqualTo(ColumnReference reference, int value) {
    super(reference);
    this.value = value;
  }

  public Selection apply(Table relation) {
    IntColumn longColumn = (IntColumn) relation.column(getColumnReference().getColumnName());
    return longColumn.isLessThanOrEqualTo(value);
  }
}

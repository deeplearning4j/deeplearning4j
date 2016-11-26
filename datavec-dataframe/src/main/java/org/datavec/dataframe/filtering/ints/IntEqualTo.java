package org.datavec.dataframe.filtering.ints;

import org.datavec.dataframe.api.IntColumn;
import org.datavec.dataframe.api.LongColumn;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.columns.ColumnReference;
import org.datavec.dataframe.filtering.ColumnFilter;
import org.datavec.dataframe.util.Selection;

/**
 *
 */
public class IntEqualTo extends ColumnFilter {

  private int value;

  public IntEqualTo(ColumnReference reference, int value) {
    super(reference);
    this.value = value;
  }

  public Selection apply(Table relation) {
    IntColumn intColumn = (IntColumn) relation.column(getColumnReference().getColumnName());
    return intColumn.isEqualTo(value);
  }
}

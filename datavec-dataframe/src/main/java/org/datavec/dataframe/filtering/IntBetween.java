package org.datavec.dataframe.filtering;

import org.datavec.dataframe.api.IntColumn;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.columns.ColumnReference;
import org.datavec.dataframe.util.Selection;

/**
 */
public class IntBetween extends ColumnFilter {
  private int low;
  private int high;

  public IntBetween(ColumnReference reference, int lowValue, int highValue) {
    super(reference);
    this.low = lowValue;
    this.high = highValue;
  }

  public Selection apply(Table relation) {
    IntColumn intColumn = (IntColumn) relation.column(columnReference.getColumnName());
    Selection matches = intColumn.isGreaterThan(low);
    matches.toBitmap().and(intColumn.isLessThan(high).toBitmap());
    return matches;
  }
}

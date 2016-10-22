package org.datavec.dataframe.filtering;

import org.datavec.dataframe.api.DateColumn;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.columns.ColumnReference;
import org.datavec.dataframe.util.Selection;

import java.time.LocalDate;

/**
 */
public class LocalDateBetween extends ColumnFilter {
  private LocalDate low;
  private LocalDate high;

  public LocalDateBetween(ColumnReference reference, LocalDate lowValue, LocalDate highValue) {
    super(reference);
    this.low = lowValue;
    this.high = highValue;
  }

  public Selection apply(Table relation) {
    DateColumn column = (DateColumn) relation.column(columnReference.getColumnName());
    Selection matches = column.isAfter(low);
    matches.and(column.isBefore(high));
    return matches;
  }
}

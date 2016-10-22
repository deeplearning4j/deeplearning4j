package org.datavec.dataframe.filtering.times;

import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.api.TimeColumn;
import org.datavec.dataframe.columns.ColumnReference;
import org.datavec.dataframe.filtering.ColumnFilter;
import org.datavec.dataframe.util.Selection;

import java.time.LocalTime;

/**
 *
 */
public class IsAfter extends ColumnFilter {

  private LocalTime value;

  public IsAfter(ColumnReference reference, LocalTime value) {
    super(reference);
    this.value = value;
  }

  public Selection apply(Table relation) {
    TimeColumn timeColumn = (TimeColumn) relation.column(columnReference().getColumnName());
    return timeColumn.isAfter(value);
  }
}

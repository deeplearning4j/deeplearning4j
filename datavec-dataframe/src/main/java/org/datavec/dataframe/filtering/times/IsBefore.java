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
public class IsBefore extends ColumnFilter {

  private LocalTime value;

  public IsBefore(ColumnReference reference, LocalTime value) {
    super(reference);
    this.value = value;
  }

  public Selection apply(Table relation) {
    TimeColumn timeColumn = (TimeColumn) relation.column(columnReference().getColumnName());
    return timeColumn.isBefore(value);
  }
}

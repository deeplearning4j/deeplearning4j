package org.datavec.dataframe.filtering;

import org.datavec.dataframe.api.DateTimeColumn;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.columns.ColumnReference;
import org.datavec.dataframe.util.Selection;

import java.time.LocalDateTime;

/**
 */
public class DateTimeEqualTo extends ColumnFilter {

  LocalDateTime value;

  public DateTimeEqualTo(ColumnReference reference, LocalDateTime value) {
    super(reference);
    this.value = value;
  }

  public Selection apply(Table relation) {
    DateTimeColumn dateColumn = (DateTimeColumn) relation.column(columnReference.getColumnName());
    return dateColumn.isEqualTo(value);
  }
}

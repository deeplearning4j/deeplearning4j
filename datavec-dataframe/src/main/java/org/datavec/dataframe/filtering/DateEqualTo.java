package org.datavec.dataframe.filtering;

import org.datavec.dataframe.api.DateColumn;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.columns.ColumnReference;
import org.datavec.dataframe.util.Selection;

import java.time.LocalDate;

/**
 */
public class DateEqualTo extends ColumnFilter {

  LocalDate value;

  public DateEqualTo(ColumnReference reference, LocalDate value) {
    super(reference);
    this.value = value;
  }

  public Selection apply(Table relation) {
    DateColumn dateColumn = (DateColumn) relation.column(columnReference.getColumnName());
    return dateColumn.isEqualTo(value);
  }
}

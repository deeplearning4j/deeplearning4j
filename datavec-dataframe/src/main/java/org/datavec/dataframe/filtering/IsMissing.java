package org.datavec.dataframe.filtering;

import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.columns.Column;
import org.datavec.dataframe.columns.ColumnReference;
import org.datavec.dataframe.util.Selection;

/**
 * A filtering that matches all missing values in a column
 */
public class IsMissing extends ColumnFilter {

  public IsMissing(ColumnReference reference) {
    super(reference);
  }

  public Selection apply(Table relation) {
    Column column = relation.column(columnReference.getColumnName());
    return column.isMissing();
  }
}

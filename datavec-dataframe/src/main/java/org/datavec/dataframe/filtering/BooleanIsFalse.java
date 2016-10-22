package org.datavec.dataframe.filtering;

import org.datavec.dataframe.api.BooleanColumn;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.columns.ColumnReference;
import org.datavec.dataframe.util.Selection;

/**
 *
 */
public class BooleanIsFalse extends ColumnFilter {

  public BooleanIsFalse(ColumnReference reference) {
    super(reference);
  }

  public Selection apply(Table relation) {
    BooleanColumn booleanColumn = (BooleanColumn) relation.column(columnReference.getColumnName());
    return booleanColumn.isFalse();
  }
}
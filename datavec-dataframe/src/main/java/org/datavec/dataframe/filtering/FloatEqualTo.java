package org.datavec.dataframe.filtering;

import org.datavec.dataframe.api.FloatColumn;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.columns.ColumnReference;
import org.datavec.dataframe.util.Selection;

/**
 *
 */
public class FloatEqualTo extends ColumnFilter {

  private float value;

  public FloatEqualTo(ColumnReference reference, float value) {
    super(reference);
    this.value = value;
  }

  public Selection apply(Table relation) {
    FloatColumn floatColumn = (FloatColumn) relation.column(columnReference.getColumnName());
    return floatColumn.isEqualTo(value);
  }
}

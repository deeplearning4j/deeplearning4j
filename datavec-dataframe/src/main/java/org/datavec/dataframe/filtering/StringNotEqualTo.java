package org.datavec.dataframe.filtering;

import org.datavec.dataframe.api.CategoryColumn;
import org.datavec.dataframe.api.ColumnType;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.columns.Column;
import org.datavec.dataframe.columns.ColumnReference;
import org.datavec.dataframe.util.Selection;

/**
 * Implements EqualTo testing for Category and Text Columns
 */
public class StringNotEqualTo extends ColumnFilter {

  private String value;

  public StringNotEqualTo(ColumnReference reference, String value) {
    super(reference);
    this.value = value;
  }

  public Selection apply(Table relation) {
    Column column = relation.column(columnReference.getColumnName());
    ColumnType type = column.type();
    switch (type) {
      case CATEGORY: {
        CategoryColumn categoryColumn = (CategoryColumn) relation.column(columnReference.getColumnName());
        return categoryColumn.isNotEqualTo(value);
      }
      default:
        throw new UnsupportedOperationException(
            String.format("ColumnType %s does not support equalTo on a String value", type));
    }
  }
}

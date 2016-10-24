package org.datavec.dataframe.filtering;

import org.datavec.dataframe.api.ColumnType;
import org.datavec.dataframe.api.IntColumn;
import org.datavec.dataframe.api.LongColumn;
import org.datavec.dataframe.api.ShortColumn;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.columns.Column;
import org.datavec.dataframe.columns.ColumnReference;
import org.datavec.dataframe.util.Selection;

/**
 *
 */
public class IntGreaterThan extends ColumnFilter {

  private int value;

  public IntGreaterThan(ColumnReference reference, int value) {
    super(reference);
    this.value = value;
  }

  public Selection apply(Table relation) {
    String name = columnReference.getColumnName();
    Column column = relation.column(name);
    ColumnType type = column.type();
    switch (type) {
      case INTEGER:
        IntColumn intColumn = relation.intColumn(name);
        return intColumn.isGreaterThan(value);
      case LONG_INT:
        LongColumn longColumn = relation.longColumn(name);
        return longColumn.isGreaterThan(value);
      case SHORT_INT:
        ShortColumn shortColumn = relation.shortColumn(name);
        return shortColumn.isGreaterThan(value);
      default:
        throw new UnsupportedOperationException("Columns of type " + type.name() + " do not support the operation "
            + "greaterThan(anInt) ");
    }
  }
}

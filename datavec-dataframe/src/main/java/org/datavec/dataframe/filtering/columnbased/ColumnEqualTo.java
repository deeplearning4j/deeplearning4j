package org.datavec.dataframe.filtering.columnbased;

import org.datavec.dataframe.api.ColumnType;
import org.datavec.dataframe.api.IntColumn;
import org.datavec.dataframe.api.LongColumn;
import org.datavec.dataframe.api.ShortColumn;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.columns.Column;
import org.datavec.dataframe.columns.ColumnReference;
import org.datavec.dataframe.filtering.ColumnFilter;
import org.datavec.dataframe.util.Selection;
import com.google.common.base.Preconditions;

/**
 *
 */
public class ColumnEqualTo extends ColumnFilter {

  private final ColumnReference otherColumn;

  public ColumnEqualTo(ColumnReference a, ColumnReference b) {
    super(a);
    otherColumn = b;
  }

  public Selection apply(Table relation) {

    Column column = relation.column(columnReference().getColumnName());
    Column other = relation.column(otherColumn.getColumnName());

    Preconditions.checkArgument(column.type() == other.type());

    if (column.type() == ColumnType.INTEGER)
      return apply((IntColumn) column, (IntColumn) other);

    if (column.type() == ColumnType.LONG_INT)
      return apply((LongColumn) column, (LongColumn) other);

    if (column.type() == ColumnType.SHORT_INT)
      return apply((ShortColumn) column, (ShortColumn) other);

    throw new UnsupportedOperationException("Not yet implemented for this column type");
  }

  private static Selection apply(IntColumn column1, IntColumn column2) {
    return column1.isEqualTo(column2);
  }

  private static Selection apply(ShortColumn column1, ShortColumn column2) {
    return column1.isEqualTo(column2);
  }

  private static Selection apply(LongColumn column1, LongColumn column2) {
    return column1.isEqualTo(column2);
  }
}

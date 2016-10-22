package org.datavec.dataframe.mapping;

import org.datavec.dataframe.api.IntColumn;
import org.datavec.dataframe.columns.ShortColumnUtils;
import org.datavec.dataframe.columns.Column;
import org.datavec.dataframe.api.FloatColumn;
import org.datavec.dataframe.api.ShortColumn;

/**
 *
 */
public interface ShortMapUtils extends ShortColumnUtils {

  default IntColumn plus(ShortColumn... columns) {

    // TODO(lwhite): Assert all columns are the same size.
    String nString = names(columns);
    String name = String.format("sum(%s)", nString);
    IntColumn newColumn = IntColumn.create(name);

    for (int r = 0; r < columns[0].size(); r++) {
      int result = 0;
      for (ShortColumn column : columns) {
        result = result + column.get(r);
      }
      newColumn.add(result);
    }
    return newColumn;
  }

  // TODO(lwhite): make this a shared utility
  default String names(ShortColumn[] columns) {
    StringBuilder builder = new StringBuilder();
    int count = 0;
    for (Column column : columns) {
      builder.append(column.name());
      if (count < columns.length - 1) {
        builder.append(", ");
      }
      count++;
    }
    return builder.toString();
  }

  /**
   * Return the elements of this column as the ratios of their value and the sum of all
   * elements
   */
  default FloatColumn asRatio() {
    FloatColumn pctColumn = new FloatColumn(name() + " percents");
    float total = sum();
    for (short next : this) {
      if (total != 0) {
        pctColumn.add((float) next / total);
      } else {
        pctColumn.add(FloatColumn.MISSING_VALUE);
      }

    }
    return pctColumn;
  }

  /**
   * Return the elements of this column as the percentages of their value relative to the sum of all
   * elements
   */
  default FloatColumn asPercent() {
    FloatColumn pctColumn = new FloatColumn(name() + " percents");
    float total = sum();
    for (short next : this) {
      if (total != 0) {
        pctColumn.add(((float) next / total) * 100);
      } else {
        pctColumn.add(FloatColumn.MISSING_VALUE);
      }
    }
    return pctColumn;
  }

  long sum();
}

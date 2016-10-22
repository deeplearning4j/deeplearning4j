package org.datavec.dataframe.mapping;

import org.datavec.dataframe.api.IntColumn;
import org.datavec.dataframe.columns.IntColumnUtils;
import org.datavec.dataframe.columns.Column;
import org.datavec.dataframe.api.FloatColumn;

/**
 *
 */
public interface IntMapUtils extends IntColumnUtils {

  default IntColumn plus(IntColumn... columns) {

    // TODO(lwhite): Assert all columns are the same size.
    String nString = names(columns);
    String name = String.format("sum(%s)", nString);
    IntColumn newColumn = IntColumn.create(name);

    for (int r = 0; r < columns[0].size(); r++) {
      int result = 0;
      for (IntColumn column : columns) {
        result = result + column.get(r);
      }
      newColumn.add(result);
    }
    return newColumn;
  }

  default IntColumn plus(int value) {

    // TODO(lwhite): Assert all columns are the same size.
    String name = name() + " + " + value;
    IntColumn newColumn = IntColumn.create(name);

    for (int r = 0; r < size(); r++) {
      newColumn.add(get(r) + value);
    }
    return newColumn;
  }

  default IntColumn multiply(int value) {

    // TODO(lwhite): Assert all columns are the same size.
    String name = name() + " * " + value;
    IntColumn newColumn = IntColumn.create(name);

    for (int r = 0; r < size(); r++) {
      newColumn.add(get(r) * value);
    }
    return newColumn;
  }

  default FloatColumn multiply(double value) {

    // TODO(lwhite): Assert all columns are the same size.
    String name = name() + " * " + value;
    FloatColumn newColumn = FloatColumn.create(name);

    for (int r = 0; r < size(); r++) {
      newColumn.add(get(r) * (float) value);
    }
    return newColumn;
  }

  default FloatColumn divide(int value) {

    // TODO(lwhite): Assert all columns are the same size.
    String name = name() + " / " + value;
    FloatColumn newColumn = FloatColumn.create(name);

    for (int r = 0; r < size(); r++) {
      newColumn.add(get(r) / (value * 1.0f));
    }
    return newColumn;
  }

  default FloatColumn divide(double value) {

    // TODO(lwhite): Assert all columns are the same size.
    String name = name() + " / " + value;
    FloatColumn newColumn = FloatColumn.create(name);

    for (int r = 0; r < size(); r++) {
      newColumn.add(get(r) / value);
    }
    return newColumn;
  }

  default FloatColumn divide(IntColumn divisor) {

    // TODO(lwhite): Assert all columns are the same size.
    String name = name() + " / " + divisor.name();
    FloatColumn newColumn = FloatColumn.create(name);

    for (int r = 0; r < size(); r++) {
      newColumn.add(get(r) / (divisor.get(r) * 1.0f));
    }
    return newColumn;
  }

  // TODO(lwhite): make this a shared utility
  default String names(IntColumn[] columns) {
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
    for (int next : this) {
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
    for (int next : this) {
      if (total != 0) {
        pctColumn.add(((float) next / total) * 100);
      } else {
        pctColumn.add(FloatColumn.MISSING_VALUE);
      }
    }
    return pctColumn;
  }

  long sum();

  int get(int index);

  default IntColumn difference(IntColumn column2) {
    IntColumn result = IntColumn.create(name() + " - " + column2.name());
    for (int r = 0; r < size(); r++) {
      result.set(r, get(r) - column2.get(r));
    }
    return result;
  }

  default IntColumn difference(int value) {
    IntColumn result = IntColumn.create(name() + " - " + value);
    for (int r = 0; r < size(); r++) {
      result.set(r, get(r) - value);
    }
    return result;
  }
}
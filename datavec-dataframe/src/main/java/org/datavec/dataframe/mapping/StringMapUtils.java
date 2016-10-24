package org.datavec.dataframe.mapping;

import org.datavec.dataframe.api.CategoryColumn;
import org.datavec.dataframe.api.FloatColumn;
import org.datavec.dataframe.columns.Column;
import com.google.common.base.Strings;
import org.apache.commons.lang3.StringUtils;

/**
 *
 */
public interface StringMapUtils extends Column {
  /**
   * String utility functions. Each function takes one or more String columns as input and produces
   * another Column as output. The resulting column need not be a string column.
   */

  default CategoryColumn upperCase() {
    CategoryColumn newColumn = CategoryColumn.create(this.name() + "[ucase]");

    for (int r = 0; r < size(); r++) {
      String value = getString(r);
      if (value == null) {
        newColumn.set(r, null);
      } else {
        newColumn.set(r, value.toUpperCase());
      }
    }
    return newColumn;
  }

  default CategoryColumn lowerCase() {

    CategoryColumn newColumn = CategoryColumn.create(name() + "[lcase]");

    for (int r = 0; r < size(); r++) {
      String value = getString(r);
      newColumn.set(r, value.toLowerCase());
    }
    return newColumn;
  }

  default CategoryColumn trim() {

    CategoryColumn newColumn = CategoryColumn.create(name() + "[trim]");

    for (int r = 0; r < size(); r++) {
      String value = getString(r);
      newColumn.set(r, value.trim());
    }
    return newColumn;
  }

  default CategoryColumn replaceAll(String regex, String replacement) {

    CategoryColumn newColumn = CategoryColumn.create(name() + "[repl]");

    for (int r = 0; r < size(); r++) {
      String value = getString(r);
      newColumn.set(r, value.replaceAll(regex, replacement));
    }
    return newColumn;
  }

  default CategoryColumn replaceFirst(String regex, String replacement) {

    CategoryColumn newColumn = CategoryColumn.create(name() + "[repl]");

    for (int r = 0; r < size(); r++) {
      String value = getString(r);
      newColumn.set(r, value.replaceFirst(regex, replacement));
    }
    return newColumn;
  }

  default CategoryColumn substring(int start, int end) {

    CategoryColumn newColumn = CategoryColumn.create(name() + "[sub]");

    for (int r = 0; r < size(); r++) {
      String value = getString(r);
      newColumn.set(r, value.substring(start, end));
    }
    return newColumn;
  }


  default CategoryColumn substring(int start) {

    CategoryColumn newColumn = CategoryColumn.create(name() + "[sub]");

    for (int r = 0; r < size(); r++) {
      String value = getString(r);
      newColumn.set(r, value.substring(start));
    }
    return newColumn;
  }

  default CategoryColumn abbreviate(int maxWidth) {

    CategoryColumn newColumn = CategoryColumn.create(name() + "[abbr]");

    for (int r = 0; r < size(); r++) {
      String value = getString(r);
      newColumn.set(r, StringUtils.abbreviate(value, maxWidth));
    }
    return newColumn;
  }

  default CategoryColumn padEnd(int minLength, char padChar) {

    CategoryColumn newColumn = CategoryColumn.create(name() + "[pad]");

    for (int r = 0; r < size(); r++) {
      String value = getString(r);
      newColumn.set(r, Strings.padEnd(value, minLength, padChar));
    }
    return newColumn;
  }

  default CategoryColumn padStart(int minLength, char padChar) {

    CategoryColumn newColumn = CategoryColumn.create(name() + "[pad]");

    for (int r = 0; r < size(); r++) {
      String value = getString(r);
      newColumn.set(r, Strings.padStart(value, minLength, padChar));
    }
    return newColumn;
  }

  default CategoryColumn commonPrefix(Column column2) {

    CategoryColumn newColumn = CategoryColumn.create(name() + column2.name() + "[prefix]");

    for (int r = 0; r < size(); r++) {
      String value1 = getString(r);
      String value2 = column2.getString(r);
      newColumn.set(r, Strings.commonPrefix(value1, value2));
    }
    return newColumn;
  }

  default CategoryColumn commonSuffix(Column column2) {

    CategoryColumn newColumn = CategoryColumn.create(name() + column2.name() + "[suffix]");

    for (int r = 0; r < size(); r++) {
      String value1 = getString(r);
      String value2 = column2.getString(r);
      newColumn.set(r, Strings.commonSuffix(value1, value2));
    }
    return newColumn;
  }

  /**
   * Returns a column containing the levenshtein distance between the two given string columns
   */
  default Column distance(Column column2) {

    FloatColumn newColumn = FloatColumn.create(name() + column2.name() + "[distance]");

    for (int r = 0; r < size(); r++) {
      String value1 = getString(r);
      String value2 = column2.getString(r);
      newColumn.set(r, StringUtils.getLevenshteinDistance(value1, value2));
    }
    return newColumn;
  }

  default CategoryColumn join(Column column2, String delimiter) {

    CategoryColumn newColumn = CategoryColumn.create(name() + column2.name() + "[join]");

    for (int r = 0; r < size(); r++) {
      String[] values = new String[2];
      values[0] = getString(r);
      values[1] = column2.getString(r);
      newColumn.set(r, StringUtils.join(values, delimiter));
    }
    return newColumn;
  }
}
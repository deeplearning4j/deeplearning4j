package org.datavec.dataframe.io.html;

import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.columns.Column;
import com.google.common.annotations.VisibleForTesting;
import org.apache.commons.lang3.StringUtils;

import java.util.List;

/**
 * Static utility that Writes outlier tables in html table format for display
 */
final public class HtmlTableWriter {

  /**
   * Private constructor to prevent instantiation
   */
  private HtmlTableWriter() {}

  public static String write(Table table, String missing) {

    StringBuilder builder = new StringBuilder();
    builder.append(header(table.columnNames()));
    builder.append("<tbody>")
        .append('\n');
    for (int row : table.rows()) {
      builder.append(row(row, table, missing));
    }
    builder.append("</tbody>");
    return builder.toString();
  }

  /**
   * Returns a string containing the html output of one table row
   */
  @VisibleForTesting
  static String row(int row, Table table, String missing) {
    StringBuilder builder = new StringBuilder()
        .append("<tr>");

    for (Column col : table.columns()) {
      builder
          .append("<td>")
          .append(String.valueOf(col.getString(row)))
          .append("</td>");
      }
    builder
        .append("</tr>")
        .append('\n');
    return builder.toString();
  }

  @VisibleForTesting
  static String header(List<String> columnNames) {
    StringBuilder builder = new StringBuilder()
        .append("<thead>")
        .append('\n')
        .append("<tr>");
    for (String name : columnNames) {
      builder
          .append("<th>")
          .append(splitCamelCase(splitOnUnderscore(name)))
          .append("</th>");
    }
    builder
        .append("</tr>")
        .append('\n')
        .append("</thead>")
        .append('\n');

    return builder.toString();
  }

  // todo move to utils
  private static String splitCamelCase(String s) {
    return StringUtils.join(
        StringUtils.splitByCharacterTypeCamelCase(s),
        ' '
    );
  }

  // todo move to utils
  static String splitOnUnderscore(String s) {
    return StringUtils.join(
        StringUtils.split(s, '_'),
        ' '
    );
  }
}
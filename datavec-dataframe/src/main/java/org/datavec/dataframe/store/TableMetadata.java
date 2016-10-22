package org.datavec.dataframe.store;

import org.datavec.dataframe.table.Relation;
import org.datavec.dataframe.columns.Column;
import com.google.gson.Gson;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Data about a specific physical table used in it's persistence
 */
public class TableMetadata {

  private static final Gson GSON = new Gson();

  private final String name;

  private final int rowCount;

  private final List<ColumnMetadata> columnMetadataList = new ArrayList<>();

  public TableMetadata(Relation table) {
    this.name = table.name();
    this.rowCount = table.rowCount();
    for (Column column : table.columns()) {
      columnMetadataList.add(new ColumnMetadata(column));
    }
  }

  public String toJson() {
    return GSON.toJson(this);
  }

  public static TableMetadata fromJson(String jsonString) {
    return GSON.fromJson(jsonString, TableMetadata.class);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    TableMetadata that = (TableMetadata) o;
    return rowCount == that.rowCount &&
        Objects.equals(name, that.name) &&
        Objects.equals(columnMetadataList, that.columnMetadataList);
  }

  @Override
  public int hashCode() {
    return Objects.hash(name, rowCount, columnMetadataList);
  }

  public String getName() {
    return name;
  }

  public int getRowCount() {
    return rowCount;
  }

  public List<ColumnMetadata> getColumnMetadataList() {
    return columnMetadataList;
  }
}

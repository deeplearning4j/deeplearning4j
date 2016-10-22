package org.datavec.dataframe.store;

import org.datavec.dataframe.columns.Column;
import org.datavec.dataframe.api.ColumnType;
import com.google.gson.Gson;

/**
 * Data about a specific column used in it's persistence
 */
public class ColumnMetadata {

  static final Gson GSON = new Gson();

  private final String id;

  private final String name;

  private final ColumnType type;

  private final int size;

  public ColumnMetadata(Column column) {
    this.id = column.id();
    this.name = column.name();
    this.type = column.type();
    this.size = column.size();
  }

  public String toJson() {
    return GSON.toJson(this);
  }

  public static ColumnMetadata fromJson(String jsonString) {
    return GSON.fromJson(jsonString, ColumnMetadata.class);
  }

  @Override
  public String toString() {
    return "ColumnMetadata{" +
        "id='" + id + '\'' +
        ", name='" + name + '\'' +
        ", type=" + type +
        ", size=" + size +
        '}';
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;

    ColumnMetadata that = (ColumnMetadata) o;

    if (size != that.size) return false;
    if (!id.equals(that.id)) return false;
    if (!name.equals(that.name)) return false;
    return type == that.type;
  }

  @Override
  public int hashCode() {
    int result = id.hashCode();
    result = 31 * result + name.hashCode();
    result = 31 * result + type.hashCode();
    result = 31 * result + size;
    return result;
  }

  public String getId() {
    return id;
  }

  public String getName() {
    return name;
  }

  public ColumnType getType() {
    return type;
  }

  public int getSize() {
    return size;
  }
}

package org.datavec.dataframe.filtering;

import lombok.Data;
import org.datavec.dataframe.columns.ColumnReference;

/**
 */
@Data
public abstract class ColumnFilter extends Filter {


  ColumnReference columnReference;

  public ColumnFilter(ColumnReference columnReference) {
    this.columnReference = columnReference;
  }

  public ColumnReference columnReference() {
    return columnReference;
  }

  public ColumnReference getColumnReference() {
    return columnReference;
  }

}

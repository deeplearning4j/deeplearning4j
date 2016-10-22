package org.datavec.dataframe.filtering;

import org.datavec.dataframe.api.IntColumn;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.columns.ColumnReference;
import org.datavec.dataframe.util.Selection;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntSet;

/**
 */
public class IntIsIn extends ColumnFilter {

  private IntColumn filterColumn;

  public IntIsIn(ColumnReference reference, IntColumn filterColumn) {
    super(reference);
    this.filterColumn = filterColumn;
  }

  public IntIsIn(ColumnReference reference, int ... ints) {
    super(reference);
    this.filterColumn = IntColumn.create("temp", new IntArrayList(ints));
  }

  public Selection apply(Table relation) {
    IntColumn intColumn = (IntColumn) relation.column(columnReference.getColumnName());
    IntSet firstSet = intColumn.asSet();
    firstSet.retainAll(filterColumn.data());
    return intColumn.select(firstSet::contains);
  }
}

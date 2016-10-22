package org.datavec.dataframe.reducing.functions;

import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.reducing.NumericReduceFunction;
import org.datavec.dataframe.reducing.NumericSummaryTable;
import org.datavec.dataframe.table.ViewGroup;

/**
 *
 */
public abstract class SummaryFunction {

  private final Table original;
  private final String summarizedColumnName;

  public SummaryFunction(Table original, String summarizedColumnName) {
    this.original = original;
    this.summarizedColumnName = summarizedColumnName;
  }

  public String summarizedColumnName() {
    return summarizedColumnName;
  }

  Table original() {
    return original;
  }

  public NumericSummaryTable by(String... columnNames) {
    ViewGroup group = ViewGroup.create(original(), columnNames);
    return group.reduce(summarizedColumnName(), function());
  }

  /**
   * Returns the result of applying to the function to all the values in the appropriate column
   */
  public double get() {
    return original.reduce(summarizedColumnName, function());
  }

  public abstract NumericReduceFunction function();
}

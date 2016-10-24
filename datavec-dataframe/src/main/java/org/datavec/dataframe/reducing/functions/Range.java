package org.datavec.dataframe.reducing.functions;

import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.reducing.NumericReduceFunction;

import static org.datavec.dataframe.reducing.NumericReduceUtils.range;

/**
 *
 */
public class Range extends SummaryFunction {

  public Range(Table original, String summarizedColumnName) {
    super(original, summarizedColumnName);
  }

  @Override
  public NumericReduceFunction function() {
    return range;
  }
}

package org.datavec.dataframe.reducing.functions;


import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.reducing.NumericReduceFunction;
import org.datavec.dataframe.reducing.NumericReduceUtils;

/**
 *
 */
public class SumOfSquares extends SummaryFunction {

  public SumOfSquares(Table original, String summarizedColumnName) {
    super(original, summarizedColumnName);
  }

  @Override
  public NumericReduceFunction function() {
    return NumericReduceUtils.sumOfSquares;
  }
}

package org.datavec.dataframe.api;

import org.datavec.dataframe.columns.Column;

/**
 * Functionality common to all numeric column types
 */
public interface NumericColumn extends Column {

  double[] toDoubleArray();

  float getFloat(int index);

  double max();
  
  double min();

  double product();

  double mean();

  double median();

  double quartile1();

  double quartile3();

  double percentile(double percentile);

  double range();

  double variance();

  double populationVariance();

  double standardDeviation();

  double sumOfLogs();

  double sumOfSquares();

  double geometricMean();

  /**
   * Returns the quadraticMean, aka the root-mean-square, for all values in this column
   */
  double quadraticMean();

  double kurtosis();

  double skewness() ;
}

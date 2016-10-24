package org.datavec.dataframe.reducing;

import org.datavec.dataframe.api.FloatColumn;
import org.datavec.dataframe.api.IntColumn;
import org.datavec.dataframe.api.LongColumn;
import org.datavec.dataframe.api.ShortColumn;

/**
 * Functions that calculate values over the data of an entire column, such as sum, mean, std. dev, etc.
 */
public interface NumericReduceFunction {

  String functionName();

  double reduce(double[] data);

  default double reduce(FloatColumn data) {
    return this.reduce(data.toDoubleArray());
  }

  default double reduce(IntColumn data) {
    return this.reduce(data.toDoubleArray());
  }

  default double reduce(ShortColumn data) {
    return this.reduce(data.toDoubleArray());
  }

  default double reduce(LongColumn data) {
    return this.reduce(data.toDoubleArray());
  }
}

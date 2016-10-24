package org.datavec.dataframe.mapping;

import org.apache.commons.math3.stat.StatUtils;

/**
 *
 */
public class NumericMapUtils {


  public double[] normalize(double[] data) {
    return StatUtils.normalize(data);
  }


}

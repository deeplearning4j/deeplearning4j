package org.datavec.dataframe.util;

import org.datavec.dataframe.api.NumericColumn;
import org.datavec.dataframe.table.TemporaryView;
import org.datavec.dataframe.table.ViewGroup;
import com.google.common.base.Preconditions;

/**
 *  Utility functions for creating 2D double arrays from columns and other arrays
 */
public class DoubleArrays {

  public static double[] toN(int n) {
    double[] result = new double[n];
    for (int i = 0; i < n; i++) {
      result[i] = i;
    }
    return result;
  }

  public static double[][] to2dArray(NumericColumn... columns) {
    Preconditions.checkArgument(columns.length >= 1);
    int obs = columns[0].size();
    double[][] allVals = new double[obs][columns.length];

    for( int r = 0; r < obs; r++ ) {
      for (int c = 0; c < columns.length; c++) {
        allVals[r][c] = columns[c].getFloat(r);
        allVals[r][c] = columns[c].getFloat(r);
      }
    }
    return allVals;
  }

  public static double[][] to2dArray(ViewGroup views, int columnNumber) {

    int viewCount = views.size();

    double[][] allVals = new double[viewCount][];
    for(int viewNumber = 0; viewNumber < viewCount; viewNumber++ ) {
      TemporaryView view = views.get(viewNumber);
      allVals[viewNumber] = new double[view.rowCount()];
      NumericColumn numericColumn = view.numericColumn(columnNumber);
      for (int r = 0; r < view.rowCount(); r++) {
        allVals[viewNumber][r] = numericColumn.getFloat(r);
      }
    }
    return allVals;
  }

  public static double[][] to2dArray(double[] x, double[] y) {
    double[][] allVals = new double[x.length][2];
    for( int i = 0; i < x.length; i++ ) {
      allVals[i][0] = x[i];
      allVals[i][1] = y[i];
    }
    return allVals;
  }

  public static double[][] to2dArray(NumericColumn x, NumericColumn y) {
    double[][] allVals = new double[x.size()][2];
    for( int i = 0; i < x.size(); i++ ) {
      allVals[i][0] = x.getFloat(i);
      allVals[i][1] = y.getFloat(i);
    }
    return allVals;
  }
}

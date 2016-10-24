package org.datavec.dataframe.sorting;

/**
 *
 */
public class LongComparisonUtil {

  private static LongComparisonUtil instance = new LongComparisonUtil();

  public static LongComparisonUtil getInstance() {
    return instance;
  }

  private LongComparisonUtil() {
  }

  public int compare(long a, long b) {
    if (a > b)
      return 1;
    if (b > a)
      return -1;
    return 0;
  }
}

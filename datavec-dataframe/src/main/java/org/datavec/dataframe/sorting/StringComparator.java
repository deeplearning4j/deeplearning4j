package org.datavec.dataframe.sorting;

/**
 *
 */
public class StringComparator {

  private static StringComparator instance = new StringComparator();

  public static StringComparator getInstance() {
    return instance;
  }

  private StringComparator() {
  }

  public int compare(String a, String b) {
    return a.compareTo(b);
  }
}

package org.datavec.dataframe.filtering.doubles;

/**
 *
 */
public interface DoubleBiPredicate {

  /**
   * Returns true if valueToTest meets the criteria of this predicate when valueToCompareAgainst is considered
   * <p>
   * Example (to compare all the values v in a column such that v is greater than 4.0, v is the value to test and 4.0
   * is the value to compare against
   *
   * @param valueToTest           the value you're checking. Often this is the value of a cell in a floatt column
   * @param valueToCompareAgainst the value to compare against. Often this is a single value for all comparisions
   */
  boolean test(double valueToTest, double valueToCompareAgainst);
}

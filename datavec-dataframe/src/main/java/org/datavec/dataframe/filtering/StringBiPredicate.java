package org.datavec.dataframe.filtering;

/**
 *
 */
public interface StringBiPredicate {

  /**
   * Returns true if valueToTest meets the criteria of this predicate when valueToCompareAgainst is considered
   * <p>
   * Example (to compare all the values v in a column such that v = "foo", v is the value to test and "foo" is the
   * value to compare against
   *
   * @param valueToTest           the value you're checking. Often this is the value of a cell in a short column
   * @param valueToCompareAgainst the value to compare against. Often this is a single value for all comparisions
   */
  boolean test(String valueToTest, String valueToCompareAgainst);
}

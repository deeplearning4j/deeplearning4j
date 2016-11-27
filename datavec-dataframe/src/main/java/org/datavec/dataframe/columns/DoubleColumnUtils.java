package org.datavec.dataframe.columns;

import it.unimi.dsi.fastutil.ints.IntIterable;
import org.datavec.dataframe.filtering.doubles.DoubleBiPredicate;
import org.datavec.dataframe.filtering.doubles.DoublePredicate;

/**
 *
 */
public interface DoubleColumnUtils extends Column, IntIterable {

  DoublePredicate isZero = i -> i == 0.0;

  DoublePredicate isNegative = i -> i < 0;

  DoublePredicate isPositive = i -> i > 0;

  DoublePredicate isNonNegative = i -> i >= 0;

  DoubleBiPredicate isGreaterThan = (valueToTest, valueToCompareAgainst) -> valueToTest > valueToCompareAgainst;

  DoubleBiPredicate isGreaterThanOrEqualTo = (valueToTest, valueToCompareAgainst) -> valueToTest >=
      valueToCompareAgainst;

  DoubleBiPredicate isLessThan = (valueToTest, valueToCompareAgainst) -> valueToTest < valueToCompareAgainst;

  DoubleBiPredicate isLessThanOrEqualTo = (valueToTest, valueToCompareAgainst) -> valueToTest <= valueToCompareAgainst;

  DoubleBiPredicate isEqualTo = (valueToTest, valueToCompareAgainst) -> valueToTest == valueToCompareAgainst;

  DoubleBiPredicate notIsEqualTo = (valueToTest, valueToCompareAgainst) -> valueToTest != valueToCompareAgainst;


  DoublePredicate isMissing = i -> i != i;

  DoublePredicate isNotMissing = i -> i == i;
}

package org.datavec.dataframe.columns;

import org.datavec.dataframe.filtering.FloatBiPredicate;
import org.datavec.dataframe.filtering.FloatPredicate;
import it.unimi.dsi.fastutil.ints.IntIterable;

/**
 *
 */
public interface FloatColumnUtils extends Column, IntIterable {

  FloatPredicate isZero = i -> i == 0.0f;

  FloatPredicate isNegative = i -> i < 0f;

  FloatPredicate isPositive = i -> i > 0f;

  FloatPredicate isNonNegative = i -> i >= 0f;

  FloatBiPredicate isGreaterThan = (valueToTest, valueToCompareAgainst) -> valueToTest > valueToCompareAgainst;

  FloatBiPredicate isGreaterThanOrEqualTo = (valueToTest, valueToCompareAgainst) -> valueToTest >=
      valueToCompareAgainst;

  FloatBiPredicate isLessThan = (valueToTest, valueToCompareAgainst) -> valueToTest < valueToCompareAgainst;

  FloatBiPredicate isLessThanOrEqualTo = (valueToTest, valueToCompareAgainst) -> valueToTest <= valueToCompareAgainst;

  FloatBiPredicate isEqualTo = (valueToTest, valueToCompareAgainst) -> valueToTest == valueToCompareAgainst;

  FloatBiPredicate notIsEqualTo = (valueToTest, valueToCompareAgainst) -> valueToTest != valueToCompareAgainst;


  FloatPredicate isMissing = i -> i != i;

  FloatPredicate isNotMissing = i -> i == i;
}

package org.datavec.dataframe.columns;

import org.datavec.dataframe.api.BooleanColumn;
import org.datavec.dataframe.filtering.BooleanPredicate;
import it.unimi.dsi.fastutil.ints.IntIterable;

/**
 *
 */
public interface BooleanColumnUtils extends Column, IntIterable {

  BooleanPredicate isMissing = i -> i == Byte.MIN_VALUE;

  BooleanPredicate isNotMissing = i -> i != BooleanColumn.MISSING_VALUE;
}

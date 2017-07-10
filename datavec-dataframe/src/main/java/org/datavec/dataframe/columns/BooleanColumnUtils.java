package org.datavec.dataframe.columns;

import it.unimi.dsi.fastutil.ints.IntIterable;
import org.datavec.dataframe.api.BooleanColumn;
import org.datavec.dataframe.filtering.BooleanPredicate;

/**
 *
 */
public interface BooleanColumnUtils extends Column, IntIterable {

    BooleanPredicate isMissing = i -> i == Byte.MIN_VALUE;

    BooleanPredicate isNotMissing = i -> i != BooleanColumn.MISSING_VALUE;
}

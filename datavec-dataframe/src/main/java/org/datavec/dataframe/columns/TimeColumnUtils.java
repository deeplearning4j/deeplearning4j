package org.datavec.dataframe.columns;

import org.datavec.dataframe.api.TimeColumn;
import org.datavec.dataframe.filtering.IntPredicate;
import it.unimi.dsi.fastutil.ints.IntArrayList;

import java.time.LocalTime;

/**
 *
 */
public interface TimeColumnUtils extends Column, Iterable<LocalTime> {

  IntArrayList data();

  IntPredicate isMissing = i -> i == TimeColumn.MISSING_VALUE;

  IntPredicate isNotMissing = i -> i != TimeColumn.MISSING_VALUE;
}

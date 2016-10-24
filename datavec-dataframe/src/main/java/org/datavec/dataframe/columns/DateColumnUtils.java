package org.datavec.dataframe.columns;

import org.datavec.dataframe.api.DateColumn;
import org.datavec.dataframe.filtering.IntPredicate;
import it.unimi.dsi.fastutil.ints.IntArrayList;

import java.time.LocalDate;

/**
 *
 */
public interface DateColumnUtils extends Column, Iterable<LocalDate> {

  IntArrayList data();

  IntPredicate isMissing = i -> i == DateColumn.MISSING_VALUE;

  IntPredicate isNotMissing = i -> i != DateColumn.MISSING_VALUE;
}

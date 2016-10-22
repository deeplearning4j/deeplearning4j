package org.datavec.dataframe.columns;

import org.datavec.dataframe.api.DateTimeColumn;
import org.datavec.dataframe.filtering.LongPredicate;
import it.unimi.dsi.fastutil.longs.LongArrayList;

/**
 *
 */
public interface DateTImeColumnUtils extends Column {

  LongArrayList data();

  LongPredicate isMissing = i -> i == DateTimeColumn.MISSING_VALUE;

  LongPredicate isNotMissing = i -> i != DateTimeColumn.MISSING_VALUE;
}

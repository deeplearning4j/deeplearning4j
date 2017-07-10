package org.datavec.dataframe.columns;

import it.unimi.dsi.fastutil.longs.LongArrayList;
import org.datavec.dataframe.api.DateTimeColumn;
import org.datavec.dataframe.filtering.LongPredicate;

/**
 *
 */
public interface DateTImeColumnUtils extends Column {

    LongArrayList data();

    LongPredicate isMissing = i -> i == DateTimeColumn.MISSING_VALUE;

    LongPredicate isNotMissing = i -> i != DateTimeColumn.MISSING_VALUE;
}

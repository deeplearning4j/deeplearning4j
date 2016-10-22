package org.datavec.dataframe.columns;

import org.datavec.dataframe.api.CategoryColumn;
import org.datavec.dataframe.filtering.StringPredicate;
import org.datavec.dataframe.mapping.StringMapUtils;
import org.datavec.dataframe.reducing.CategoryReduceUtils;
import org.datavec.dataframe.util.DictionaryMap;
import it.unimi.dsi.fastutil.ints.IntArrayList;

/**
 *
 */
public interface CategoryColumnUtils extends Column, StringMapUtils, CategoryReduceUtils, Iterable<String> {

  StringPredicate isMissing = i -> i.equals(CategoryColumn.MISSING_VALUE);
  StringPredicate isNotMissing = i -> !i.equals(CategoryColumn.MISSING_VALUE);

  DictionaryMap dictionaryMap();

  IntArrayList values();
}

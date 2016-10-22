package org.datavec.dataframe.filtering;

import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.util.Selection;

/**
 * A predicate applied to a Relation, to return a subset of the rows in that table
 */
public abstract class Filter {

  public abstract Selection apply(Table relation);
}

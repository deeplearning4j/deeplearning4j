package org.datavec.dataframe.filtering;

import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.util.Selection;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

/**
 * A composite filtering that only returns {@code true} if all component filters return true
 */
public class AnyOf extends CompositeFilter {

  private List<Filter> filterList = new ArrayList<>();

  AnyOf(Collection<Filter> filters) {

    this.filterList.addAll(filters);
  }

  public static AnyOf anyOf(Filter... filters) {
    List<Filter> filterList = new ArrayList<>();
    Collections.addAll(filterList, filters);
    return new AnyOf(filterList);
  }

  public static AnyOf anyOf(Collection<Filter> filters) {
    return new AnyOf(filters);
  }

  public Selection apply(Table relation) {
    Selection selection = null;
    for (Filter filter : filterList) {
      if (selection == null) {
        selection = filter.apply(relation);
      } else {
        selection.or(filter.apply(relation));
      }
    }
    return selection;
  }
}

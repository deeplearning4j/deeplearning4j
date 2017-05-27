package org.datavec.dataframe.filtering;

import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.util.BitmapBackedSelection;
import org.datavec.dataframe.util.Selection;

import net.jcip.annotations.Immutable;

/**
 * A boolean filtering, returns true if the filtering it wraps returns false, and vice-versa.
 */
@Immutable
public class IsFalse extends CompositeFilter {

    private final Filter filter;

    private IsFalse(Filter filter) {
        this.filter = filter;
    }

    public static IsFalse isFalse(Filter filter) {
        return new IsFalse(filter);
    }

    /**
     * Returns true if the element in the given row in my {@code column} is true
     */
    @Override
    public Selection apply(Table relation) {
        Selection selection = new BitmapBackedSelection();
        selection.addRange(0, relation.rowCount());
        selection.andNot(filter.apply(relation));
        return selection;
    }
}

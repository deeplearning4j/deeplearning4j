package org.datavec.dataframe.filtering;

import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.util.Selection;

import net.jcip.annotations.Immutable;

/**
 * A boolean filtering. For symmetry with IsFalse
 */
@Immutable
public class IsTrue extends CompositeFilter {

    private final Filter filter;

    private IsTrue(Filter filter) {
        this.filter = filter;
    }

    public static IsTrue isTrue(Filter filter) {
        return new IsTrue(filter);
    }

    /**
     * Returns true if the element in the given row in my {@code column} is true
     *
     * @param relation
     */
    @Override
    public Selection apply(Table relation) {
        return filter.apply(relation);
    }
}

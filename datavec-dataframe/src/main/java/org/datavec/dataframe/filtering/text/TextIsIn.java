package org.datavec.dataframe.filtering.text;

import org.datavec.dataframe.api.CategoryColumn;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.columns.Column;
import org.datavec.dataframe.columns.ColumnReference;
import org.datavec.dataframe.filtering.ColumnFilter;
import org.datavec.dataframe.util.Selection;

import net.jcip.annotations.Immutable;

/**
 * A filtering that selects cells in which the string value is in the given array of strings
 */
@Immutable
public class TextIsIn extends ColumnFilter {

    private String[] strings;

    public TextIsIn(ColumnReference reference, String... strings) {
        super(reference);
        this.strings = strings;
    }

    @Override
    public Selection apply(Table relation) {
        Column column = relation.column(columnReference().getColumnName());
        CategoryColumn textColumn = (CategoryColumn) column;
        return textColumn.isIn(strings);
    }
}

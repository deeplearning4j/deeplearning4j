package org.datavec.dataframe.filtering.datetimes;


import org.datavec.dataframe.api.DateTimeColumn;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.columns.ColumnReference;
import org.datavec.dataframe.filtering.ColumnFilter;
import org.datavec.dataframe.util.Selection;

import net.jcip.annotations.Immutable;

/**
 *
 */
@Immutable
public class DateIsOnOrAfter extends ColumnFilter {

    private int value;

    public DateIsOnOrAfter(ColumnReference reference, int value) {
        super(reference);
        this.value = value;
    }

    @Override
    public Selection apply(Table relation) {

        DateTimeColumn dateColumn = (DateTimeColumn) relation.column(columnReference().getColumnName());
        return dateColumn.isOnOrAfter(value);
    }
}

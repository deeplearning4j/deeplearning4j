package org.datavec.dataframe.filtering.datetimes;


import org.datavec.dataframe.api.DateTimeColumn;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.columns.ColumnReference;
import org.datavec.dataframe.filtering.ColumnFilter;
import org.datavec.dataframe.util.Selection;

import net.jcip.annotations.Immutable;
import java.time.LocalDateTime;

/**
 *
 */
@Immutable
public class DateTimeIsAfter extends ColumnFilter {

    private LocalDateTime value;

    public DateTimeIsAfter(ColumnReference reference, LocalDateTime value) {
        super(reference);
        this.value = value;
    }

    @Override
    public Selection apply(Table relation) {

        DateTimeColumn dateColumn = relation.dateTimeColumn(columnReference().getColumnName());
        return dateColumn.isAfter(value);
    }
}

package org.datavec.dataframe.columns.packeddata;

import com.google.common.annotations.Beta;
import org.datavec.dataframe.columns.DateIntervalColumn;

/**
 * EXPERIMENTAL
 */
@Beta
public abstract class PackedDateInterval {

    // boolean operations
    abstract boolean equals(DateIntervalColumn interval);

    abstract boolean before(DateIntervalColumn interval);

    abstract boolean after(DateIntervalColumn interval);
}

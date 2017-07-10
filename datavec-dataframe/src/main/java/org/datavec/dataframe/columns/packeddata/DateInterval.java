package org.datavec.dataframe.columns.packeddata;

import com.google.common.annotations.Beta;
import org.datavec.dataframe.columns.DateIntervalColumn;

/**
 * EXPERIMENTAL
 */
@Beta
public abstract class DateInterval {

    // boolean operations
    abstract boolean equals(DateIntervalColumn interval);

    abstract boolean before(DateIntervalColumn interval);

    abstract boolean after(DateIntervalColumn interval);

    abstract boolean contains(DateIntervalColumn interval);

    abstract boolean containedIn(DateIntervalColumn interval);

    abstract boolean meets(DateIntervalColumn interval);

    // combination operations
    abstract DateInterval union(DateInterval interval); // or

    abstract DateInterval intersect(DateInterval interval); // and

    abstract DateInterval minus(DateInterval interval); // and not

    abstract DateInterval gap(DateInterval interval); // the difference between two intervals

}

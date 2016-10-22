package org.datavec.dataframe.columns.packeddata;

import org.datavec.dataframe.columns.DateIntervalColumn;
import com.google.common.annotations.Beta;

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

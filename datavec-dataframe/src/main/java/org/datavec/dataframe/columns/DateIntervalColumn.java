package org.datavec.dataframe.columns;

import org.datavec.dataframe.api.DateColumn;
import org.datavec.dataframe.columns.packeddata.DateInterval;
import org.datavec.dataframe.columns.packeddata.PackedLocalDate;
import org.datavec.dataframe.util.Selection;
import com.google.common.annotations.Beta;

import java.util.List;


/**
 * EXPERIMENTAL
 */
@Beta
public abstract class DateIntervalColumn {

  /*-------------------------------------------------------*/
  // Column-wise boolean operations against individual values
  /*-------------------------------------------------------*/

  // boolean comparisons to other intervals
  abstract Selection equals(DateInterval interval);
  abstract Selection before(DateInterval interval);
  abstract Selection after(DateInterval interval);
  abstract Selection overlaps(DateInterval interval);

  /**
   * Returns a selection containing all cells whose interval is during (containedBy) the given interval
   */
  abstract Selection containedIn(DateInterval interval);
  abstract Selection contains(DateInterval interval);

  /**
   * Returns true if interval a end + 1 = interval b start; or vice versa
   */
  abstract Selection meets(DateInterval interval);

  // boolean comparisons to individual dates
  abstract Selection before(PackedLocalDate date);
  abstract Selection after(PackedLocalDate date);
  abstract Selection contains(PackedLocalDate date);
  abstract Selection meets(PackedLocalDate date);


  /*-------------------------------------------------------*/
  // Column-wise boolean operations against other columns
  /*-------------------------------------------------------*/

  abstract Selection equals(DateIntervalColumn interval);
  abstract Selection before(DateIntervalColumn interval);
  abstract Selection after(DateIntervalColumn interval);
  abstract Selection overlaps(DateIntervalColumn interval);

  /**
   * Returns a selection containing all cells whose interval is during (containedBy) the given interval
   */
  abstract Selection containedIn(DateIntervalColumn interval);
  abstract Selection contains(DateIntervalColumn interval);

  /**
   * Returns true if interval a end + 1 = interval b start; or vice versa
   */
  abstract Selection meets(DateIntervalColumn interval);

  // boolean comparisons to individual dates
  abstract Selection before(DateColumn column);
  abstract Selection after(DateColumn column);
  abstract Selection contains(DateColumn column);
  abstract Selection meets(DateColumn column);

  /**




  /*-------------------------------------------------------*/
  // reduction methods
  /*-------------------------------------------------------*/
  abstract int sumDuration();
  abstract int maxDuration();
  abstract int minDuration();
  abstract float meanDuration();
  abstract float medianDuration();
  abstract float durationVariance();
  abstract float durationStdDev();

  abstract PackedLocalDate earliestStart();
  abstract PackedLocalDate lastestEnd();
  abstract DateInterval span();

  /*-------------------------------------------------------*/
  // misc methods
  /*-------------------------------------------------------*/
  abstract List<PackedLocalDate> toDays();

}

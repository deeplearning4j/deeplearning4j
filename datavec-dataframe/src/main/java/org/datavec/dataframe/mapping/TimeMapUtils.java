package org.datavec.dataframe.mapping;

import org.datavec.dataframe.api.IntColumn;
import org.datavec.dataframe.api.LongColumn;
import org.datavec.dataframe.api.ShortColumn;
import org.datavec.dataframe.api.TimeColumn;
import org.datavec.dataframe.columns.TimeColumnUtils;
import org.datavec.dataframe.columns.packeddata.PackedLocalTime;

import java.time.LocalTime;
import java.time.temporal.ChronoUnit;

public interface TimeMapUtils extends TimeColumnUtils {

  default LongColumn differenceInMilliseconds(TimeColumn column2) {
    return difference(column2, ChronoUnit.MILLIS);
  }

  default LongColumn differenceInSeconds(TimeColumn column2) {
    return difference(column2, ChronoUnit.SECONDS);
  }

  default LongColumn differenceInMinutes(TimeColumn column2) {
    return difference(column2, ChronoUnit.MINUTES);
  }

  default LongColumn differenceInHours(TimeColumn column2) {
    return difference(column2, ChronoUnit.HOURS);
  }

  default LongColumn difference(TimeColumn column2, ChronoUnit unit) {
    LongColumn newColumn = LongColumn.create(name() + " - " + column2.name());

    for (int r = 0; r < size(); r++) {
      int c1 = this.getInt(r);
      int c2 = column2.getInt(r);
      if (c1 == TimeColumn.MISSING_VALUE || c2 == TimeColumn.MISSING_VALUE) {
        newColumn.add(IntColumn.MISSING_VALUE);
      } else {
        newColumn.add(difference(c1, c2, unit));
      }
    }
    return newColumn;
  }

  default long difference(int packedLocalTime1, int packedLocalTime2, ChronoUnit unit) {
    LocalTime value1 = PackedLocalTime.asLocalTime(packedLocalTime1);
    LocalTime value2 = PackedLocalTime.asLocalTime(packedLocalTime2);
    return unit.between(value1, value2);
  }

  default ShortColumn hour() {
    ShortColumn newColumn = ShortColumn.create(name() + "[" + "hour" + "]");
    for (int r = 0; r < size(); r++) {
      int c1 = getInt(r);
      if (c1 != TimeColumn.MISSING_VALUE) {
        newColumn.add(PackedLocalTime.getHour(c1));
      } else {
        newColumn.add(ShortColumn.MISSING_VALUE);
      }
    }
    return newColumn;
  }

  default IntColumn minuteOfDay() {
    IntColumn newColumn = IntColumn.create(name() + "[" + "minute-of-day" + "]");
    for (int r = 0; r < size(); r++) {
      int c1 = getInt(r);
      if (c1 != TimeColumn.MISSING_VALUE) {
        newColumn.add(PackedLocalTime.getMinuteOfDay(c1));
      } else {
        newColumn.add(IntColumn.MISSING_VALUE);
      }
    }
    return newColumn;
  }

  default IntColumn secondOfDay() {
    IntColumn newColumn = IntColumn.create(name() + "[" + "second-of-day" + "]");
    for (int r = 0; r < size(); r++) {
      int c1 = getInt(r);
      if (c1 != TimeColumn.MISSING_VALUE) {
        newColumn.add(PackedLocalTime.getSecondOfDay(c1));
      } else {
        newColumn.add(IntColumn.MISSING_VALUE);
      }
    }
    return newColumn;
  }

  LocalTime get(int r);

  int getInt(int r);
}

package org.datavec.dataframe.mapping;

import org.datavec.dataframe.columns.DateTImeColumnUtils;
import org.datavec.dataframe.api.DateTimeColumn;
import org.datavec.dataframe.api.IntColumn;
import org.datavec.dataframe.api.LongColumn;
import org.datavec.dataframe.api.ShortColumn;
import org.datavec.dataframe.columns.packeddata.PackedLocalDateTime;

import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;

public interface DateTimeMapUtils extends DateTImeColumnUtils {

  default LongColumn differenceInMilliseconds(DateTimeColumn column2) {
    return difference(column2, ChronoUnit.MILLIS);
  }

  default LongColumn differenceInSeconds(DateTimeColumn column2) {
    return difference(column2, ChronoUnit.SECONDS);
  }

  default LongColumn differenceInMinutes(DateTimeColumn column2) {
    return difference(column2, ChronoUnit.MINUTES);
  }

  default LongColumn differenceInHours(DateTimeColumn column2) {
    return difference(column2, ChronoUnit.HOURS);
  }

  default LongColumn differenceInDays(DateTimeColumn column2) {
    return difference(column2, ChronoUnit.DAYS);
  }

  default LongColumn differenceInYears(DateTimeColumn column2) {
    return difference(column2, ChronoUnit.YEARS);
  }

  default LongColumn difference(DateTimeColumn column2, ChronoUnit unit) {
    LongColumn newColumn = LongColumn.create(name() + " - " + column2.name());

    for (int r = 0; r < size(); r++) {
      long c1 = this.getLong(r);
      long c2 = column2.getLong(r);
      if (c1 == DateTimeColumn.MISSING_VALUE || c2 == DateTimeColumn.MISSING_VALUE) {
        newColumn.add(IntColumn.MISSING_VALUE);
      } else {
        newColumn.add(difference(c1, c2, unit));
      }
    }
    return newColumn;
  }

  default long difference(long packedLocalDateTime1, long packedLocalDateTime2, ChronoUnit unit) {
    LocalDateTime value1 = PackedLocalDateTime.asLocalDateTime(packedLocalDateTime1);
    LocalDateTime value2 = PackedLocalDateTime.asLocalDateTime(packedLocalDateTime2);
    return unit.between(value1, value2);
  }

  default ShortColumn hour() {
    ShortColumn newColumn = ShortColumn.create(name() + "[" + "hour" + "]");
    for (int r = 0; r < size(); r++) {
      long c1 = getLong(r);
      if (c1 != DateTimeColumn.MISSING_VALUE) {
        newColumn.add(PackedLocalDateTime.getHour(c1));
      } else {
        newColumn.add(ShortColumn.MISSING_VALUE);
      }
    }
    return newColumn;
  }

  default ShortColumn minuteOfDay() {
    ShortColumn newColumn = ShortColumn.create(name() + "[" + "minute-of-day" + "]");
    for (int r = 0; r < size(); r++) {
      long c1 = getLong(r);
      if (c1 != DateTimeColumn.MISSING_VALUE) {
        newColumn.add((short) PackedLocalDateTime.getMinuteOfDay(c1));
      } else {
        newColumn.add(ShortColumn.MISSING_VALUE);
      }
    }
    return newColumn;
  }

  default IntColumn secondOfDay() {
    IntColumn newColumn = IntColumn.create(name() + "[" + "second-of-day" + "]");
    for (int r = 0; r < size(); r++) {
      long c1 = getLong(r);
      if (c1 != DateTimeColumn.MISSING_VALUE) {
        newColumn.add(PackedLocalDateTime.getSecondOfDay(c1));
      } else {
        newColumn.add(IntColumn.MISSING_VALUE);
      }
    }
    return newColumn;
  }

  LocalDateTime get(int r);

  long getLong(int r);
}

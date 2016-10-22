package org.datavec.dataframe.splitting.dates;

import org.datavec.dataframe.columns.packeddata.PackedLocalDate;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.Month;
import java.time.temporal.ChronoField;
import java.util.function.Function;

/**
 *
 */
public class SplitUtils {

  public static LocalDateSplitter byYear = PackedLocalDate::getYear;

  public static LocalDateSplitter byMonth = PackedLocalDate::getMonthValue;

  public static LocalDateSplitter byDayOfMonth = PackedLocalDate::getDayOfMonth;

  public static LocalDateSplitter byDayOfYear = PackedLocalDate::getDayOfYear;

  public static LocalDateSplitter byDayOfWeek = packedLocalDate ->
      PackedLocalDate.getDayOfWeek(packedLocalDate).getValue();

  public static LocalDateSplitter byQuarter = PackedLocalDate::getQuarter;


  public static Function<Comparable, Object> byWeek = comparable -> {
    if (comparable instanceof LocalDate) {
      return ((LocalDate) comparable).get(ChronoField.ALIGNED_WEEK_OF_YEAR);
    } else if (comparable instanceof LocalDateTime) {
      return ((LocalDateTime) comparable).get(ChronoField.ALIGNED_WEEK_OF_YEAR);
    } else {
      throw new IllegalArgumentException("Date function called on non-date column");
    }
  };

  public static Function<Comparable, Object> byHour = comparable -> {
    if (comparable instanceof LocalDateTime) {
      return ((LocalDateTime) comparable).get(ChronoField.HOUR_OF_DAY);
    } else {
      throw new IllegalArgumentException("Time function called on non-time column");
    }
  };

  public static Function<Comparable, Object> bySecondOfMinute = comparable -> {
    if (comparable instanceof LocalDateTime) {
      return ((LocalDateTime) comparable).get(ChronoField.SECOND_OF_MINUTE);
    } else {
      throw new IllegalArgumentException("Time function called on non-time column");
    }
  };

  public static Function<Comparable, Object> bySecondOfDay = comparable -> {
    if (comparable instanceof LocalDateTime) {
      return ((LocalDateTime) comparable).get(ChronoField.SECOND_OF_DAY);
    } else {
      throw new IllegalArgumentException("Time function called on non-time column");
    }
  };

  public static Function<Comparable, Object> byMinuteOfHour = comparable -> {
    if (comparable instanceof LocalDateTime) {
      return ((LocalDateTime) comparable).get(ChronoField.MINUTE_OF_HOUR);
    } else {
      throw new IllegalArgumentException("Time function called on non-time column");
    }
  };

  public static Function<Comparable, Object> byMinuteOfDay = comparable -> {
    if (comparable instanceof LocalDateTime) {
      return ((LocalDateTime) comparable).get(ChronoField.MINUTE_OF_HOUR);
    } else {
      throw new IllegalArgumentException("Time function called on non-time column");
    }
  };

  private static int getQuarter(Month month) {
    int monthValue = month.getValue();
    if (monthValue <= 3) {
      return 1;
    } else if (monthValue <= 6) {
      return 2;
    } else if (monthValue <= 9) {
      return 3;
    } else {
      return 4;
    }
  }

/*
  BY_QUARTER_AND_YEAR,  // 1974-Q1; 1974-Q2; etc.
  BY_MONTH_AND_YEAR,    // 1974-01; 1974-02; 1974-03; etc.
  BY_WEEK_AND_YEAR,     // 1956-51; 1956-52;
  BY_DAY_AND_YEAR,      // 1990-364; 1990-365;
  BY_DAY_AND_MONTH,           // 12-03
  BY_DAY_AND_MONTH_AND_YEAR,  // 2003-04-15
  BY_DAY_AND_WEEK_AND_YEAR,   // 1993-48-6
  BY_DAY_AND_WEEK,            // 52-1 to 52-7
  BY_HOUR_AND_DAY,            //
  BY_MINUTE_AND_HOUR,         // 23-49
*/

}
